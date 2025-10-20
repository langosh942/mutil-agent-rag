from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    func,
    or_,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from .config import DatabaseConfig


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    source: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now())
    chunks: Mapped[List["DocumentChunk"]] = relationship(
        "DocumentChunk", back_populates="document", cascade="all, delete-orphan"
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(Integer, ForeignKey("documents.id"), nullable=False, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    metadata: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    keywords_json: Mapped[Optional[str]] = mapped_column("keywords", Text, nullable=True)

    document: Mapped[Document] = relationship("Document", back_populates="chunks")
    triples: Mapped[List["KnowledgeTriple"]] = relationship(
        "KnowledgeTriple", back_populates="chunk", cascade="all, delete-orphan"
    )


class KnowledgeTriple(Base):
    __tablename__ = "knowledge_triples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    subject: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    predicate: Mapped[str] = mapped_column(String(255), nullable=False, default="appears_in")
    object: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    chunk_id: Mapped[int] = mapped_column(Integer, ForeignKey("document_chunks.id"), nullable=False)

    chunk: Mapped[DocumentChunk] = relationship("DocumentChunk", back_populates="triples")


class DatabaseManager:
    """Lightweight wrapper around SQLAlchemy to manage document storage."""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.engine: Engine = create_engine(config.url, echo=config.echo, future=True)
        self._session_factory = sessionmaker(bind=self.engine, expire_on_commit=False, class_=Session, future=True)
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session_scope(self) -> Iterable[Session]:
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except SQLAlchemyError:
            session.rollback()
            raise
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------
    def ingest_document(self, title: str, payloads: List[Dict]) -> int:
        """Persist document, chunks and derived knowledge triples."""

        if not payloads:
            raise ValueError("payloads must not be empty")

        with self.session_scope() as session:
            document = Document(title=title, source=payloads[0].get("source"))
            session.add(document)
            session.flush()

            for payload in payloads:
                metadata = {
                    key: value
                    for key, value in payload.items()
                    if key not in {"text", "chunk_index", "keywords"}
                }
                chunk = DocumentChunk(
                    document_id=document.id,
                    chunk_index=int(payload.get("chunk_index", 0)),
                    text=payload["text"],
                    metadata=json.dumps(metadata, ensure_ascii=False) if metadata else None,
                    keywords_json=json.dumps(payload.get("keywords", []), ensure_ascii=False) or None,
                )
                session.add(chunk)
                session.flush()

                for keyword in payload.get("keywords", []):
                    triple = KnowledgeTriple(
                        subject=keyword,
                        predicate="appears_in",
                        object=title,
                        chunk_id=chunk.id,
                    )
                    session.add(triple)

            session.flush()
            document_id = document.id

        return document_id

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------
    def keyword_search(self, query: str, limit: int = 5) -> List[Dict]:
        if not query:
            return []

        pattern = f"%{query}%"
        with self.session_scope() as session:
            results = (
                session.query(DocumentChunk, Document)
                .join(Document, DocumentChunk.document_id == Document.id)
                .filter(DocumentChunk.text.ilike(pattern))
                .order_by(DocumentChunk.chunk_index.asc())
                .limit(limit)
                .all()
            )

            return [
                {
                    "document": document.title,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "source": document.source,
                    "keywords": self._deserialize_keywords(chunk.keywords_json),
                }
                for chunk, document in results
            ]

    def knowledge_graph_search(self, query: str, limit: int = 5) -> List[Dict]:
        if not query:
            return []

        pattern = f"%{query}%"
        with self.session_scope() as session:
            results = (
                session.query(KnowledgeTriple, DocumentChunk, Document)
                .join(DocumentChunk, KnowledgeTriple.chunk_id == DocumentChunk.id)
                .join(Document, DocumentChunk.document_id == Document.id)
                .filter(or_(KnowledgeTriple.subject.ilike(pattern), KnowledgeTriple.object.ilike(pattern)))
                .limit(limit)
                .all()
            )

            formatted: List[Dict] = []
            for triple, chunk, document in results:
                formatted.append(
                    {
                        "subject": triple.subject,
                        "predicate": triple.predicate,
                        "object": triple.object,
                        "document": document.title,
                        "chunk_index": chunk.chunk_index,
                        "snippet": chunk.text[:300],
                    }
                )
            return formatted

    def list_documents(self, limit: int = 50) -> List[Dict]:
        with self.session_scope() as session:
            rows = session.query(Document).order_by(Document.created_at.desc()).limit(limit).all()
            return [
                {
                    "id": row.id,
                    "title": row.title,
                    "source": row.source,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                    "chunk_count": len(row.chunks),
                }
                for row in rows
            ]

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _deserialize_keywords(raw: Optional[str]) -> List[str]:
        if not raw:
            return []
        try:
            value = json.loads(raw)
            if isinstance(value, list):
                return [str(item) for item in value]
        except json.JSONDecodeError:
            pass
        return []

from __future__ import annotations

import ast
import operator
import re
from typing import Dict, List, Optional, Tuple

from duckduckgo_search import DDGS

_MATH_PATTERN = re.compile(r"([\d\.\s\+\-\*\/\(\)]+)")
_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


class SafeEvaluator(ast.NodeVisitor):
    def visit(self, node):  # type: ignore[override]
        if isinstance(node, ast.Expression):
            return self.visit(node.body)
        if isinstance(node, ast.Num):  # type: ignore[attr-defined]
            return node.n
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
            operand = self.visit(node.operand)
            return _ALLOWED_OPERATORS[type(node.op)](operand)
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
            left = self.visit(node.left)
            right = self.visit(node.right)
            return _ALLOWED_OPERATORS[type(node.op)](left, right)
        raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def _safe_eval(expression: str) -> float:
    tree = ast.parse(expression, mode="eval")
    return SafeEvaluator().visit(tree)


def detect_math_expression(text: str) -> Optional[str]:
    if not text:
        return None
    match = _MATH_PATTERN.search(text)
    if not match:
        return None
    expression = match.group(1)
    if not expression:
        return None
    if re.search(r"[A-Za-z]", expression):
        return None
    return expression.strip()


def run_tooling(question: str) -> Tuple[List[Dict], Optional[str]]:
    """Attempt to execute lightweight tools based on the question content."""

    outputs: List[Dict] = []
    error: Optional[str] = None

    expression = detect_math_expression(question)
    if expression:
        try:
            result = _safe_eval(expression)
            outputs.append({"tool": "calculator", "input": expression, "output": result})
        except Exception as exc:
            error = f"Calculator error: {exc}"

    return outputs, error


def web_search(query: str, max_results: int = 3) -> Tuple[List[Dict], Optional[str]]:
    """Execute a DuckDuckGo search returning simplified results."""

    if not query:
        return [], None

    try:
        with DDGS() as ddgs:  # type: ignore[no-redef]
            results = ddgs.text(query, max_results=max_results)
            items: List[Dict] = []
            for item in results:
                items.append(
                    {
                        "title": item.get("title"),
                        "snippet": item.get("body"),
                        "url": item.get("href"),
                    }
                )
            return items, None
    except Exception as exc:  # pragma: no cover - network / environment dependent
        return [], str(exc)

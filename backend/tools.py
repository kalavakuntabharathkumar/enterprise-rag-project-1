"""Tool implementations available to the tool_agent node.

Each tool is a plain callable that takes a string input and returns a
string result.  New tools can be added to TOOL_REGISTRY without touching
the graph or the pipeline.
"""

import ast
import math
import operator as op
from typing import Any, Callable, Dict


# ---------------------------------------------------------------------------
# Safe arithmetic evaluator (no eval(), no exec())
# ---------------------------------------------------------------------------

_SAFE_OPS = {
    ast.Add:  op.add,
    ast.Sub:  op.sub,
    ast.Mult: op.mul,
    ast.Div:  op.truediv,
    ast.Pow:  op.pow,
    ast.Mod:  op.mod,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}

_SAFE_NAMES: Dict[str, Any] = {
    "pi":    math.pi,
    "e":     math.e,
    "sqrt":  math.sqrt,
    "abs":   abs,
    "round": round,
    "sin":   math.sin,
    "cos":   math.cos,
    "tan":   math.tan,
    "log":   math.log,
    "log2":  math.log2,
    "log10": math.log10,
    "ceil":  math.ceil,
    "floor": math.floor,
    "exp":   math.exp,
}


def _eval_node(node: ast.AST) -> Any:
    """Recursively evaluate a parsed AST node using only safe operations."""
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")
        return node.value

    if isinstance(node, ast.Name):
        if node.id in _SAFE_NAMES:
            return _SAFE_NAMES[node.id]
        raise ValueError(f"Unknown name: '{node.id}'")

    if isinstance(node, ast.BinOp):
        fn = _SAFE_OPS.get(type(node.op))
        if fn is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return fn(_eval_node(node.left), _eval_node(node.right))

    if isinstance(node, ast.UnaryOp):
        fn = _SAFE_OPS.get(type(node.op))
        if fn is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return fn(_eval_node(node.operand))

    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _SAFE_NAMES:
            fn = _SAFE_NAMES[node.func.id]
            if callable(fn):
                args = [_eval_node(a) for a in node.args]
                return fn(*args)
        raise ValueError(f"Unsupported function call: {ast.dump(node.func)}")

    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def calculator(expression: str) -> str:
    """Evaluate a safe arithmetic expression and return the result as a string.

    Supports the four basic operations, exponentiation, modulo, unary
    negation, and the constants/functions listed in ``_SAFE_NAMES``.
    All evaluation is done through Python's ``ast`` module — no ``eval``
    or ``exec`` calls are made.
    """
    expression = expression.strip()
    if not expression:
        return "Error: empty expression"
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
        # Return integer representation when the result is a whole number.
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return str(result)
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception as exc:
        return f"Error: {exc}"


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOL_REGISTRY: Dict[str, Callable[[str], str]] = {
    "calculator": calculator,
}


def dispatch_tool(tool_name: str, tool_input: str) -> str:
    """Look up *tool_name* in the registry and invoke it with *tool_input*.

    Returns the tool's string output, or an error message when the tool
    name is not recognised.
    """
    fn = TOOL_REGISTRY.get(tool_name)
    if fn is None:
        return f"Error: unknown tool '{tool_name}'"
    return fn(tool_input)

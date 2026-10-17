"""Every import in the docs' Python snippets must resolve.

docs/API_LOGIC.md had four snippets that raised on their first line (names
that are not root exports, a module that did not exist). Only the import
statements are executed; the rest of each snippet needs data the docs do not
define.
"""
import ast
import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"
_FENCE = re.compile(r"```python\n(.*?)```", re.S)
_OURS = {"mlxmolkit", "opencheese"}


def _doc_imports():
    for doc in sorted(DOCS.glob("*.md")):
        for block in _FENCE.findall(doc.read_text()):
            try:
                tree = ast.parse(block)
            except SyntaxError:
                continue  # pseudo-code, not runnable Python
            for node in tree.body:
                if isinstance(node, ast.ImportFrom):
                    root = (node.module or "").split(".")[0]
                elif isinstance(node, ast.Import):
                    root = node.names[0].name.split(".")[0]
                else:
                    continue
                if root in _OURS:
                    stmt = ast.unparse(node)
                    yield pytest.param(stmt, id=f"{doc.name}:{stmt}")


@pytest.mark.parametrize("statement", list(_doc_imports()))
def test_doc_import_resolves(statement):
    exec(statement, {})

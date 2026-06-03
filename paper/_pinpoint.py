"""Pinpoint module-level statement differences between two Python files.

Compares each top-level statement after stripping module / function /
class docstrings. For matching statements (same Python source after
ast.unparse), it confirms identity. For unmatched ones it prints both
sides with line numbers so a human can decide which side to keep.

Usage:
    python paper/_pinpoint.py <baseline.py> <current.py>
"""
import ast
import sys


def strip_docs(node):
    """
    Recursively strip leading docstring expressions from a parsed AST.

    Walks ``node`` in-place and removes the first body element whenever
    that element is a string-literal expression (the convention for
    module, class, and function docstrings). This lets us compare two
    Python files structurally without false positives caused by
    documentation differences.

    Parameters
    ----------
    node : ast.AST
        Any AST node; the function recurses into all child nodes.

    Returns
    -------
    None
        Mutates ``node`` in place.
    """
    if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        b = node.body
        if (b and isinstance(b[0], ast.Expr)
                and isinstance(b[0].value, ast.Constant)
                and isinstance(b[0].value.value, str)):
            node.body = b[1:]
    for c in ast.iter_child_nodes(node):
        strip_docs(c)


def top_level_codeprints(path):
    """Return list of (lineno, dump, source) for each top-level statement
    of the file, after stripping docstrings everywhere."""
    src = open(path, encoding='utf-8').read()
    tree = ast.parse(src)
    strip_docs(tree)
    out = []
    for node in tree.body:
        # Skip pure-string expression nodes (these are stripped doc/banners).
        if (isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            continue
        d = ast.dump(node)
        try:
            s = ast.unparse(node)
        except Exception:
            s = '<unparse failed>'
        out.append((node.lineno, d, s))
    return out


a, b = sys.argv[1], sys.argv[2]
A = top_level_codeprints(a)
B = top_level_codeprints(b)
A_dumps = {x[1] for x in A}
B_dumps = {x[1] for x in B}

only_a = [x for x in A if x[1] not in B_dumps]
only_b = [x for x in B if x[1] not in A_dumps]

print(f'\n=== {a}: top-level statements not present in {b} ===')
for lineno, _, s in only_a:
    snippet = s if len(s) < 200 else s[:200] + '...'
    print(f'\n[line {lineno}]\n{snippet}')

print(f'\n=== {b}: top-level statements not present in {a} ===')
for lineno, _, s in only_b:
    snippet = s if len(s) < 200 else s[:200] + '...'
    print(f'\n[line {lineno}]\n{snippet}')

print(f'\nSummary: {len(only_a)} in {a} only, {len(only_b)} in {b} only')

"""Parity guard for the verbatim copy in finder_ext.

finder_ext copies ``TransitFinder._process_cb_events`` from the installed
mono-cbp source with exactly one added line (marked CBP-VET). mono-cbp is
never edited, but it could be upgraded; if the installed source drifts from
the pinned copy, every column the benchmark's real negatives were built from
becomes suspect. The comparison is on the AST (docstrings and line-wrapping
whitespace are not semantics), so any BEHAVIORAL drift fails loudly while
cosmetic re-formatting does not.
"""

import ast
import inspect
import textwrap

from mono_cbp.transit_finding.finder import TransitFinder

from cbpvet.search.finder_ext import TransitFinderExt


class _DropMarked(ast.NodeTransformer):
    """Remove Expr statements whose dump contains the marker."""

    def __init__(self, marker):
        self.marker = marker
        self.dropped = 0

    def visit_Expr(self, node):
        if self.marker is not None and self.marker in ast.dump(node):
            self.dropped += 1
            return None
        return self.generic_visit(node)


def _canonical_ast(func, drop_marker=None):
    src = textwrap.dedent(inspect.getsource(func))
    fn = ast.parse(src).body[0]
    # A docstring is not semantics.
    if fn.body and isinstance(fn.body[0], ast.Expr) and isinstance(
            fn.body[0].value, ast.Constant) and isinstance(fn.body[0].value.value, str):
        fn.body = fn.body[1:]
    t = _DropMarked(drop_marker)
    fn = t.visit(fn)
    if drop_marker is not None:
        assert t.dropped == 1, (
            f"expected exactly one statement containing {drop_marker!r}, "
            f"dropped {t.dropped}")
    return ast.dump(fn)


def test_copy_is_stock_plus_one_added_statement():
    stock = _canonical_ast(TransitFinder._process_cb_events)
    copy = _canonical_ast(TransitFinderExt._process_cb_events,
                          drop_marker="n_detrend_detections")
    assert copy == stock, (
        "finder_ext's _process_cb_events no longer matches the installed "
        "mono-cbp source beyond the single CBP-VET line. If mono-cbp was "
        "upgraded, re-pin against the new source and re-verify the frozen "
        "search's columns before trusting any new run."
    )


def test_extra_column_is_trailing_only():
    # The stock results keys must all survive, with the extra count appended,
    # so every existing reader of detected_events.txt keeps working.
    stock_keys = list(TransitFinder._initialise_results().keys())
    ext_keys = list(TransitFinderExt._initialise_results().keys())
    assert ext_keys[: len(stock_keys)] == stock_keys
    assert ext_keys[len(stock_keys):] == ["n_detrend_detections"]

"""Cross-cutting rules for backtest/, checked by reading the source."""
import ast
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent / "backtest"
# settings.py holds the values; data.py holds the frozen data path, hash and row split.
EXEMPT = {"settings.py", "data.py"}


def _modules():
    return [(p.name, ast.parse(p.read_text())) for p in sorted(PKG.glob("*.py")) if p.name not in EXEMPT]


def test_no_numeric_module_constants():
    """Every tunable value lives in Settings: no module-level CONSTANT = <number>."""
    offenders = [
        f"{name}:{node.lineno} {t.id}"
        for name, tree in _modules()
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, (int, float))
        for t in node.targets
        if isinstance(t, ast.Name) and t.id.lstrip("_").isupper()
    ]
    assert offenders == []


def test_settings_is_never_a_default_argument():
    """Callers pass the shared Settings; a Settings() default would let them skip it silently."""
    offenders = [
        f"{name}:{d.lineno}"
        for name, tree in _modules()
        for fn in ast.walk(tree)
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
        for d in fn.args.defaults + [k for k in fn.args.kw_defaults if k is not None]
        if isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == "Settings"
    ]
    assert offenders == []


def test_barrier_function_is_the_only_barrier_logic():
    offenders = [p.name for p in PKG.glob("*.py")
                 if p.name not in {"settings.py", "simulator.py"}
                 and ("upper_barrier" in p.read_text() or "lower_barrier" in p.read_text())]
    assert offenders == []

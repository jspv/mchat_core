import importlib.util
import os
import pytest

def has_pkg(name: str) -> bool:
    return importlib.util.find_spec(name) is not None

def require_pkgs(names: list[str]):
    missing = [n for n in names if not has_pkg(n)]
    if missing:
        pytest.skip(f"Skipping optional tools test; missing packages: {', '.join(missing)}")

def require_env(var: str):
    if not os.environ.get(var):
        pytest.skip(f"Skipping optional tools test; missing env var: {var}")
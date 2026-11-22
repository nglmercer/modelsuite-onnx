"""Plugin registries. Adapters and exporters are discovered from (a) built-ins, (b) setuptools entry points,
(c) `--plugin`/config paths and modules, so custom ones never require editing this package."""
from __future__ import annotations

import importlib
import importlib.metadata as md
import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Callable, Generic, Iterator, TypeVar

from .errors import ConfigError, ModelSuiteError

log = logging.getLogger("modelsuite.registry")
T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str, entry_point_group: str, builtins: str | None = None):
        """`builtins`: dotted module whose import registers the built-in items; imported lazily on first lookup so that
        library users get the same registry contents as the CLI, regardless of what else they imported."""
        self.kind, self.group, self._builtins = kind, entry_point_group, builtins
        self._builtins_loaded = False
        self._items: dict[str, T] = {}
        self._origin: dict[str, str] = {}
        self._load_errors: dict[str, str] = {}
        self._entry_points_loaded = False

    # -- registration -------------------------------------------------------------------
    def register(self, name: str, obj: T, *, origin: str = "runtime", replace: bool = False) -> T:
        if name in self._items and not replace and self._items[name] is not obj:
            raise ConfigError(f"{self.kind} '{name}' is already registered (by {self._origin[name]})")
        self._items[name] = obj
        self._origin[name] = origin
        return obj

    def decorator(self, name: str | None = None, *, replace: bool = False) -> Callable[[T], T]:
        def wrap(obj: T) -> T:
            self.register(name or getattr(obj, "name"), obj, origin=getattr(obj, "__module__", "runtime"), replace=replace)
            return obj
        return wrap

    # -- lookup ---------------------------------------------------------------------------
    def load_entry_points(self) -> None:
        if not self._builtins_loaded and self._builtins:
            self._builtins_loaded = True
            importlib.import_module(self._builtins)
        if self._entry_points_loaded:
            return
        self._entry_points_loaded = True
        for ep in md.entry_points(group=self.group):
            if ep.name in self._items:
                continue
            try:
                self.register(ep.name, ep.load(), origin=f"entry-point:{ep.value}")
            except Exception as exc:  # a broken third-party plugin must not break the CLI
                self._load_errors[ep.name] = f"{type(exc).__name__}: {exc}"
                log.warning("could not load %s '%s': %s", self.kind, ep.name, exc)

    def get(self, name: str) -> T:
        self.load_entry_points()
        try:
            return self._items[name]
        except KeyError:
            known = ", ".join(sorted(self._items)) or "(none)"
            raise ModelSuiteError(f"unknown {self.kind} '{name}'. Registered: {known}") from None

    def names(self) -> list[str]:
        self.load_entry_points()
        return sorted(self._items)

    def items(self) -> Iterator[tuple[str, T]]:
        self.load_entry_points()
        yield from sorted(self._items.items())

    def origin(self, name: str) -> str:
        return self._origin.get(name, "unknown")

    @property
    def load_errors(self) -> dict[str, str]:
        self.load_entry_points()
        return dict(self._load_errors)

    def __contains__(self, name: str) -> bool:
        self.load_entry_points()
        return name in self._items


adapters: Registry = Registry("adapter", "modelsuite.adapters", builtins="modelsuite.adapters.builtin")
exporters: Registry = Registry("exporter", "modelsuite.exporters", builtins="modelsuite.exporters")


def register_adapter(cls=None, *, name: str | None = None, replace: bool = False):
    """Decorator: `@register_adapter` on an Adapter subclass."""
    def wrap(c):
        adapters.register(name or c.name, c, origin=c.__module__, replace=replace)
        return c
    return wrap(cls) if cls is not None else wrap


def register_exporter(cls=None, *, name: str | None = None, replace: bool = False):
    def wrap(c):
        exporters.register(name or c.name, c, origin=c.__module__, replace=replace)
        return c
    return wrap(cls) if cls is not None else wrap


_loaded_plugins: set[str] = set()


def load_plugin(spec: str) -> None:
    """Import a plugin so its decorators run. `spec` is a .py file path or an importable module name."""
    if spec in _loaded_plugins:
        return
    path = Path(spec).expanduser()
    if spec.endswith(".py") or path.is_file():
        if not path.is_file():
            raise ConfigError(f"plugin file not found: {spec}")
        mod_name = f"modelsuite_plugin_{path.stem}_{abs(hash(str(path.resolve()))) % 10**8}"
        module_spec = importlib.util.spec_from_file_location(mod_name, path)
        if module_spec is None or module_spec.loader is None:
            raise ConfigError(f"cannot import plugin file {spec}")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[mod_name] = module
        module_spec.loader.exec_module(module)
    else:
        try:
            importlib.import_module(spec)
        except ImportError as exc:
            raise ConfigError(f"cannot import plugin module '{spec}': {exc}") from exc
    _loaded_plugins.add(spec)


def load_plugins(specs: list[str] | None = None) -> None:
    """Load plugins from the given specs plus the MODELSUITE_PLUGINS environment variable (os.pathsep separated)."""
    env = [s for s in os.environ.get("MODELSUITE_PLUGINS", "").split(os.pathsep) if s]
    for spec in [*env, *(specs or [])]:
        load_plugin(spec)

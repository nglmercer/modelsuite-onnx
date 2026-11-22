"""Exception hierarchy. Everything the suite raises on purpose derives from ModelSuiteError."""
from __future__ import annotations


class ModelSuiteError(Exception):
    """Base class; the CLI prints these without a traceback."""


class ConfigError(ModelSuiteError):
    """Invalid user configuration."""


class DependencyMissingError(ModelSuiteError):
    def __init__(self, package: str, purpose: str = "", extra: str | None = None):
        self.package, self.purpose, self.extra = package, purpose, extra
        hint = f" Install it with: pip install 'modelsuite[{extra}]'" if extra else f" Install it with: pip install {package}"
        super().__init__(f"Optional dependency '{package}' is required{' for ' + purpose if purpose else ''}.{hint}")


class DiscoveryError(ModelSuiteError):
    """The model reference could not be resolved."""


class NoAdapterError(ModelSuiteError):
    """No registered adapter recognises the model."""


class UnsupportedModelError(ModelSuiteError):
    """The model was recognised but cannot be converted; the message says why."""


class ExportError(ModelSuiteError):
    """An exporter failed."""

    def __init__(self, message: str, attempts: list[dict] | None = None):
        super().__init__(message)
        self.attempts = attempts or []


class ProviderError(ModelSuiteError):
    """A requested execution provider is not usable. Never swallowed into a CPU fallback."""


class ArtifactError(ModelSuiteError):
    """Malformed or missing artifact / manifest."""


class LicenseError(ModelSuiteError):
    """An operation was refused because of a model license."""

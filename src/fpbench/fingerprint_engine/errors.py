from __future__ import annotations


class FingerprintEngineError(RuntimeError):
    """Base class for fingerprint engine failures."""


class ProviderUnavailableError(FingerprintEngineError):
    """Raised when a selected provider is registered but unavailable."""


class TemplateExtractionError(FingerprintEngineError):
    """Raised when template extraction fails."""


class VerificationError(FingerprintEngineError):
    """Raised when 1:1 verification fails."""


class IdentificationError(FingerprintEngineError):
    """Raised when 1:N identification fails."""


class UnsupportedEngineOperationError(FingerprintEngineError):
    """Raised when a provider does not support the requested operation."""


class InvalidTemplateError(FingerprintEngineError):
    """Raised when a template is malformed or from the wrong provider."""

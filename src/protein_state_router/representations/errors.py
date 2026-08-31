"""Named errors for embedding resolution, execution, and portable bundles."""


class EmbeddingError(RuntimeError):
    """Base error for this package's embedding workflow."""


class ProteinQueryError(EmbeddingError):
    """Raised when a protein input cannot be resolved unambiguously."""


class BackendCapabilityError(EmbeddingError):
    """Raised when an external backend cannot produce the requested embeddings."""


class EmbeddingBundleError(EmbeddingError):
    """Raised when a workspace/result archive is unsafe or invalid."""

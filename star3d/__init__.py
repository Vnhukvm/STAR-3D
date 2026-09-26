"""Minimal, dataset-independent STAR-3D test-time adaptation runtime."""

__all__ = ["Config", "FeatureEncoder", "adapt_stream"]


def __getattr__(name):
    if name in __all__:
        from . import tta_tempv3
        return getattr(tta_tempv3, name)
    raise AttributeError(name)

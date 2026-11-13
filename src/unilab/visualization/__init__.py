"""Visualization and playback helpers."""

from importlib import import_module

__all__ = ["render_play_mode"]


def __getattr__(name: str):
    if name != "render_play_mode":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(".playback", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

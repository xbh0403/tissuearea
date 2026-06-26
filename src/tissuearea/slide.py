"""Minimal read-only whole-slide-image reader.

A thin wrapper over OpenSlide exposing only what tissue-area estimation needs:
level-0 dimensions, MPP metadata, and an RGB thumbnail. It never writes to the
slide or its directory.
"""

from pathlib import Path
from typing import Mapping, Optional, Tuple, Union

import numpy as np
import openslide
from PIL import Image

__all__ = ["SlideReader", "resolve_mpp_xy"]


def resolve_mpp_xy(
    props: Mapping[str, str],
    fallback_mpp: Optional[float] = None,
) -> Tuple[float, float]:
    """Resolve level-0 ``(mpp_x, mpp_y)`` in microns/pixel from slide properties.

    Preference order:
      1. ``openslide.mpp-x`` / ``openslide.mpp-y`` (a missing axis mirrors the other).
      2. ``fallback_mpp`` for both axes.
      3. ``10 / objective-power`` for both axes.

    Raises:
        ValueError: if no usable resolution can be determined.
    """

    def _pos_float(value: Optional[str]) -> Optional[float]:
        try:
            x = float(value)
        except (TypeError, ValueError):
            return None
        return x if x > 0 else None

    mpp_x = _pos_float(props.get(openslide.PROPERTY_NAME_MPP_X))
    mpp_y = _pos_float(props.get(openslide.PROPERTY_NAME_MPP_Y))

    if mpp_x is None and mpp_y is not None:
        mpp_x = mpp_y
    if mpp_y is None and mpp_x is not None:
        mpp_y = mpp_x

    if mpp_x is None and mpp_y is None:
        fb = fallback_mpp if (fallback_mpp and fallback_mpp > 0) else None
        if fb is None:
            obj = _pos_float(props.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER))
            if obj is not None:
                fb = 10.0 / obj
        if fb is None:
            raise ValueError(
                "Cannot resolve MPP: no mpp-x/mpp-y, fallback, or objective-power."
            )
        mpp_x = mpp_y = fb

    return float(mpp_x), float(mpp_y)


class SlideReader:
    """Read-only handle on a WSI, backed by OpenSlide.

    Exposes ``slide_id``, ``dimensions`` (level-0 W, H), ``mpp``, the raw
    OpenSlide ``properties``, and ``get_thumbnail(scale)``. Usable as a context
    manager.
    """

    def __init__(self, path: Union[str, Path], slide_id: Optional[str] = None):
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"Slide file not found at: {self._path}")
        self.slide_id = slide_id or self._path.stem
        self.os_slide = openslide.OpenSlide(str(self._path))

    @property
    def path(self) -> Path:
        return self._path

    @property
    def dimensions(self) -> Tuple[int, int]:
        """Level-0 ``(width, height)`` in pixels."""
        return self.os_slide.dimensions

    @property
    def properties(self) -> Mapping[str, str]:
        return self.os_slide.properties

    @property
    def mpp(self) -> Optional[float]:
        """Level-0 microns-per-pixel averaged over X/Y, or ``None`` if unknown."""

        def _parse(value: Optional[str]) -> Optional[float]:
            try:
                x = float(value)
            except (TypeError, ValueError):
                return None
            return x if x > 0 else None

        mpp_x = _parse(self.properties.get(openslide.PROPERTY_NAME_MPP_X))
        mpp_y = _parse(self.properties.get(openslide.PROPERTY_NAME_MPP_Y))
        if mpp_x and mpp_y:
            return (mpp_x + mpp_y) / 2.0
        return mpp_x or mpp_y

    def get_thumbnail(self, scale: int = 32) -> Image.Image:
        """Return an RGB thumbnail downsampled by ``scale`` from level 0."""
        w, h = self.dimensions
        thumb_dims = (max(1, w // scale), max(1, h // scale))
        return self.os_slide.get_thumbnail(thumb_dims).convert("RGB")

    def get_thumbnail_array(self, scale: int = 32) -> np.ndarray:
        """Convenience: ``get_thumbnail`` as an RGB uint8 numpy array."""
        return np.array(self.get_thumbnail(scale))

    def close(self) -> None:
        if getattr(self, "os_slide", None) is not None:
            try:
                self.os_slide.close()
            finally:
                self.os_slide = None

    def __enter__(self) -> "SlideReader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        return f"SlideReader('{self.slide_id}', dimensions={self.dimensions})"

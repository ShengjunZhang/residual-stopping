"""Figure export helpers for Chinese labels in EPS submissions."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def save_png_backed_eps(fig, eps_path: Path, png_path: Path, dpi: int = 360, **savefig_kwargs) -> None:
    """Save a high-resolution PNG and wrap it as EPS.

    Matplotlib's PostScript backend may drop CJK text when EPS files are
    included by XeLaTeX/dvipdfmx.  Rendering the figure to PNG first keeps the
    Chinese labels stable while preserving the EPS file format required by the
    journal.
    """

    png_path = Path(png_path)
    eps_path = Path(eps_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    eps_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=dpi, facecolor="white", **savefig_kwargs)
    with Image.open(png_path) as image:
        image.convert("RGB").save(eps_path, format="EPS")

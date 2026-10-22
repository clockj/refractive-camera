"""One look for every demo's figures: surface, ink, grid, and one colour per method, the same in every chart.

The method colours are a colour-blind-safe categorical set (checked for adjacent pairs, light surface); the light
aqua and yellow sit below 3:1 against the surface, so every chart that uses them labels its marks or has a legend.
"""
import matplotlib as mpl

SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
DARK = "#1a1a19"                       # the surface behind camera frames and renders

# one colour per method, in every demo
COLOR = {
    "refractive": "#eb6834",            # refraction-aware (ours)
    "refractive_plus": "#6250d6",       # ours, using more of the model (whole-ball fit, pose refinement)
    "afrt": "#1baf7a",                  # pinhole + AFRT
    "pinhole_dist": "#2a78d6",          # pinhole + distortion
    "pinhole": "#eda100",               # pinhole, the window ignored
}
LABEL = {
    "refractive": "refraction-aware (ours)",
    "afrt": "pinhole + AFRT",
    "pinhole_dist": "pinhole + distortion",
    "pinhole": "pinhole",
}
# what a camera sees of a point, in diagrams
SEEN = {"none": "#3d3d3a", "one": "#c3c2b7", "two": "#e87ba4"}


def use():
    """Apply the style to matplotlib."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.titlepad": 8, "axes.labelsize": 9, "axes.labelcolor": INK2, "text.color": INK,
        "axes.edgecolor": BASE, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-", "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "xtick.major.size": 0, "ytick.major.size": 0,
        "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
        "patch.linewidth": 0, "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.12,
    })


def bar_labels(ax, bars, fmt, color=INK2, fontsize=8):
    """The value above each column, in text ink (never the bar's colour)."""
    for b in bars:
        x, y = b.get_x() + b.get_width() / 2, b.get_height()
        ax.annotate(fmt(y), (x, y), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                    fontsize=fontsize, color=color)


def hbar_labels(ax, bars, fmt, color=INK2, fontsize=8):
    """The value after each horizontal bar, in text ink."""
    for b in bars:
        x, y = b.get_width(), b.get_y() + b.get_height() / 2
        ax.annotate(fmt(x), (x, y), xytext=(4, 0), textcoords="offset points", ha="left", va="center",
                    fontsize=fontsize, color=color)

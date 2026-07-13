"""Shared figure palette (extracted from the legacy analysis.py so active scripts
don't import the archived cycles-project code)."""

from matplotlib.colors import LinearSegmentedColormap

INK, SECONDARY, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
BLUES = LinearSegmentedColormap.from_list("blues", ["#fcfcfb", "#cde2fb", "#3987e5", "#104281", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("div", ["#104281", "#3987e5", "#f0efec", "#e34948", "#8c2b2b"])

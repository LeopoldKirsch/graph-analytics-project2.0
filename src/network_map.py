"""Maps of the network for the report and the defence.

1. full network: world map, every route drawn as a great-circle
   arc, CDG's routes highlighted.
2. French airports only: metropolitan France with insets for the Antilles,
   French Guiana, Reunion / Mayotte and Tahiti; metro-overseas routes cross
   from the main map into the insets.

Node size ~ sqrt(passengers on the airport's routes); line width ~ passengers.

External data (stated in the README):
- data/external/airport_coords.csv: airport positions from OurAirports
  (public domain, https://ourairports.com/data/), plus closed airports placed
  by hand (column "source"), see fetch_data.py.
- data/external/ne_50m_countries.geojson: Natural Earth 1:50m countries
  (public domain, https://www.naturalearthdata.com).

Usage:
    python src/network_map.py         # 2024 -> results/map_*.png
    python src/network_map.py 2019
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch

from build_graph import MAIN_YEAR, build_graph

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "data" / "external"
RESULTS_DIR = ROOT / "results"
HUB = "LFPG"

# colours (validated default palette, slots 1-2) and neutrals
FRENCH = "#2a78d6"
FOREIGN = "#eb6834"
HUB_ROUTE = "#2a78d6"
OTHER_ROUTE = "#8a8984"
LAND = "#ecebe7"
BORDER = "#ffffff"
SEA = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"

SHORT_NAMES = {
    "LFPG": "Paris-CDG", "LFPO": "Paris-Orly", "LFOB": "Beauvais",
    "FMEE": "Réunion", "FMCZ": "Mayotte", "NTAA": "Tahiti",
    "NTTB": "Bora Bora", "NTTR": "Raiatea", "TFFR": "Guadeloupe",
    "TFFF": "Martinique", "TFFG": "St-Martin (Grand Case)",
    "TFFJ": "St-Barthélemy", "SOCA": "Cayenne", "SOOA": "Maripasoula",
    "LFSB": "Basel-Mulhouse", "LFKJ": "Ajaccio",
    "LFKB": "Bastia", "LFKF": "Figari", "LFMN": "Nice", "LFLL": "Lyon",
    "LFML": "Marseille", "LFBO": "Toulouse", "LFBD": "Bordeaux",
    "LFRS": "Nantes", "LFQQ": "Lille", "LFST": "Strasbourg",
    "LFRB": "Brest", "LFRN": "Rennes", "LFBZ": "Biarritz",
    "LFLS": "Grenoble", "LFLB": "Chambéry", "LFBE": "Bergerac",
    "LFBT": "Tarbes-Lourdes", "LFRK": "Caen", "LFOT": "Tours", "LFBH": "La Rochelle",
    "LFBL": "Limoges", "LFLC": "Clermont-Ferrand", "LFTW": "Nîmes", "LFMT": "Montpellier",
    "LFMU": "Béziers", "LFMK": "Carcassonne", "LFBP": "Pau", "LFMP": "Perpignan",
    "LFTH": "Toulon-Hyères", "LFKC": "Calvi", "KLAX": "Los Angeles",
}


# ---------------------------------------------------------------- geometry

def load_coords() -> pd.DataFrame:
    return pd.read_csv(EXTERNAL / "airport_coords.csv", keep_default_na=False).set_index("icao")


def load_countries() -> list[np.ndarray]:
    """Country outlines as a flat list of (lon, lat) rings."""
    with open(EXTERNAL / "ne_50m_countries.geojson") as f:
        gj = json.load(f)
    rings = []
    for feat in gj["features"]:
        geom = feat["geometry"]
        polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        for poly in polys:
            rings.append(np.asarray(poly[0]))
    return rings


def great_circle(lon1, lat1, lon2, lat2, n=64) -> list[np.ndarray]:
    """Points along the great circle, split where it crosses the date line."""
    p1, p2 = np.radians([lat1, lon1]), np.radians([lat2, lon2])
    v1 = np.array([np.cos(p1[0]) * np.cos(p1[1]), np.cos(p1[0]) * np.sin(p1[1]), np.sin(p1[0])])
    v2 = np.array([np.cos(p2[0]) * np.cos(p2[1]), np.cos(p2[0]) * np.sin(p2[1]), np.sin(p2[0])])
    omega = np.arccos(np.clip(v1 @ v2, -1, 1))
    if omega < 1e-9:
        return [np.array([[lon1, lat1], [lon2, lat2]])]
    t = np.linspace(0, 1, n)[:, None]
    pts = (np.sin((1 - t) * omega) * v1 + np.sin(t * omega) * v2) / np.sin(omega)
    lat = np.degrees(np.arcsin(pts[:, 2]))
    lon = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
    cuts = np.where(np.abs(np.diff(lon)) > 180)[0] + 1
    return [seg for seg in np.split(np.column_stack([lon, lat]), cuts) if len(seg) > 1]


# ---------------------------------------------------------------- drawing

def draw_land(ax, rings, extent):
    ax.add_collection(PolyCollection(rings, facecolor=LAND, edgecolor=BORDER,
                                     linewidth=0.4, zorder=0))
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_facecolor(SEA)
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#d6d5d0")


def width(passengers, pmax, lo=0.4, hi=4.5):
    return lo + (hi - lo) * np.sqrt(passengers / pmax)


def size(strength, smax, lo=8, hi=520):
    return lo + (hi - lo) * np.sqrt(strength / smax)


def draw_edges(ax, G, xy, pmax, great=True, alpha=0.55):
    """Hub routes in blue on top of the other routes in grey."""
    for hub_pass in (False, True):
        segs, widths = [], []
        for u, v, p in G.edges(data="passengers"):
            if (HUB in (u, v)) != hub_pass:
                continue
            (x1, y1), (x2, y2) = xy[u], xy[v]
            parts = great_circle(x1, y1, x2, y2) if great else [np.array([[x1, y1], [x2, y2]])]
            segs += parts
            widths += [width(p, pmax)] * len(parts)
        ax.add_collection(LineCollection(
            segs, linewidths=widths, colors=HUB_ROUTE if hub_pass else OTHER_ROUTE,
            alpha=alpha if hub_pass else alpha * 0.8, capstyle="round", zorder=1 + hub_pass))


def draw_nodes(ax, G, xy, smax, nodes=None):
    nodes = list(G) if nodes is None else nodes
    nodes = sorted(nodes, key=lambda n: -G.degree(n, weight="passengers"))  # big first
    ax.scatter([xy[n][0] for n in nodes], [xy[n][1] for n in nodes],
               s=[size(G.degree(n, weight="passengers"), smax) for n in nodes],
               c=[FRENCH if G.nodes[n]["french"] else FOREIGN for n in nodes],
               edgecolors="white", linewidths=1.2, zorder=4)


def label(ax, xy, code, text=None, dx=5, dy=5, ha="left", size=8, weight="normal"):
    ax.annotate(text or SHORT_NAMES.get(code, code), xy[code], xytext=(dx, dy),
                textcoords="offset points", ha=ha, va="center", fontsize=size,
                color=INK, weight=weight, zorder=5)


def legend(ax, G, smax, pmax, loc, hub_label="Routes via Paris-CDG", foreign=True):
    p_ref = 10 ** np.floor(np.log10(pmax))
    handles = [
        Line2D([], [], marker="o", ls="", ms=8, mfc=FRENCH, mec="white", label="French airport"),
    ] + ([Line2D([], [], marker="o", ls="", ms=8, mfc=FOREIGN, mec="white",
                 label="Foreign airport")] if foreign else []) + [
        Line2D([], [], color=HUB_ROUTE, lw=2.5, label=hub_label),
        Line2D([], [], color=OTHER_ROUTE, lw=2.5, label="Other routes"),
        Line2D([], [], color=INK_2, lw=width(p_ref, pmax), label=f"{p_ref / 1e6:g}M passengers / year"),
    ]
    ax.legend(handles=handles, loc=loc, fontsize=8, frameon=True, facecolor="white",
              edgecolor="#d6d5d0", labelcolor=INK, borderpad=0.8)


# ---------------------------------------------------------------- maps

def positions(G, coords):
    missing = [n for n in G if n not in coords.index]
    if missing:
        raise KeyError(f"no coordinates for {missing}")
    return {n: (coords.at[n, "lon"], coords.at[n, "lat"]) for n in G}


def map_full(G, coords, rings, path):
    xy = positions(G, coords)
    pmax = max(p for _, _, p in G.edges(data="passengers"))
    smax = max(s for _, s in G.degree(weight="passengers"))

    fig, ax = plt.subplots(figsize=(14, 7.6))
    draw_land(ax, rings, (-170, 180, -45, 72))
    draw_edges(ax, G, xy, pmax)
    draw_nodes(ax, G, xy, smax)
    label(ax, xy, "LFPG", dx=-8, dy=10, ha="right", size=10, weight="bold")
    for code, dx, dy, ha in (("NTAA", 6, -8, "left"), ("FMEE", -8, -4, "right"),
                             ("TFFR", 7, 4, "left"), ("KLAX", -7, -6, "right")):
        if code in G:
            label(ax, xy, code, text=SHORT_NAMES.get(code, G.nodes[code]["name"].title()),
                  dx=dx, dy=dy, ha=ha)

    # Europe zoom over the empty area south-east (Australia / Indian Ocean)
    ext = (-11, 32, 34, 61)
    iax = ax.inset_axes([0.70, 0.03, 0.27, 0.34])  # stays below Singapore (lat 1N)
    draw_land(iax, rings, ext)
    iax.set_aspect(1 / np.cos(np.radians(47)), adjustable="datalim")
    inside = [n for n in G if ext[0] <= xy[n][0] <= ext[1] and ext[2] <= xy[n][1] <= ext[3]]
    draw_edges(iax, G, xy, pmax)
    draw_nodes(iax, G, xy, smax, nodes=inside)
    label(iax, xy, "LFPG", dx=-6, dy=8, ha="right", size=8.5, weight="bold")
    iax.set_title(f"Europe: {len(inside)} of the {G.number_of_nodes()} airports",
                  fontsize=8.5, color=INK_2, loc="left")
    ax.indicate_inset_zoom(iax, edgecolor=INK_2, alpha=0.6)

    n_fr = sum(1 for _, f in G.nodes(data="french") if f)
    via = G.degree(HUB) / G.number_of_edges()
    ax.set_title(
        f"Full network, {G.graph['year']}: {G.number_of_nodes()} airports "
        f"({n_fr} French, {G.number_of_nodes() - n_fr} foreign), {G.number_of_edges()} routes, "
        f"{via:.0%} of them through Paris-CDG",
        loc="left", fontsize=12, color=INK)
    legend(ax, G, smax, pmax, loc="lower left")
    ax.text(1, -0.02, "Data: Eurostat (routes), OurAirports (positions), Natural Earth (map). "
            "Only routes touching France are in the data.",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


INSETS = {
    # name: (extent lon0, lon1, lat0, lat1), axes rect in figure fraction
    "Antilles": ((-63.4, -60.6, 14.2, 18.3), [0.02, 0.62, 0.17, 0.26]),
    "French Guiana": ((-55.0, -51.6, 2.0, 6.0), [0.02, 0.42, 0.17, 0.17]),
    "Réunion / Mayotte": ((44.6, 56.2, -21.8, -12.3), [0.02, 0.17, 0.17, 0.22]),
    "Tahiti": ((-150.0, -149.2, -17.95, -17.4), [0.02, 0.03, 0.17, 0.10]),
}
METRO_EXTENT = (-5.6, 10.0, 41.1, 51.4)


def map_french(G, coords, rings, path):
    F = G.subgraph([n for n, f in G.nodes(data="french") if f]).copy()
    xy = positions(F, coords)
    pmax = max(p for _, _, p in F.edges(data="passengers"))
    smax = max(s for _, s in F.degree(weight="passengers"))

    fig = plt.figure(figsize=(11, 10))
    ax = fig.add_axes([0.21, 0.04, 0.77, 0.86])
    draw_land(ax, rings, METRO_EXTENT)
    ax.set_aspect(1 / np.cos(np.radians(46.3)))

    def in_box(n, ext):
        x, y = xy[n]
        return ext[0] <= x <= ext[1] and ext[2] <= y <= ext[3]

    metro = [n for n in F if in_box(n, METRO_EXTENT)]
    insets = {}
    for name, (ext, rect) in INSETS.items():
        iax = fig.add_axes(rect)
        draw_land(iax, rings, ext)
        iax.set_aspect(1 / np.cos(np.radians((ext[2] + ext[3]) / 2)), adjustable="datalim")
        iax.set_title(name, fontsize=8.5, color=INK_2, loc="left")
        insets[name] = (iax, [n for n in F if in_box(n, ext)])

    def axis_of(n):
        if n in metro:
            return ax
        return next(iax for iax, ns in insets.values() if n in ns)

    # routes inside one panel: straight lines; routes between panels: connectors
    for target, nodes in [(ax, metro)] + [(iax, ns) for iax, ns in insets.values()]:
        draw_edges(target, F.subgraph(nodes), xy, pmax, great=False, alpha=0.7)
        draw_nodes(target, F, xy, smax, nodes=nodes)
    for u, v, p in F.edges(data="passengers"):
        au, av = axis_of(u), axis_of(v)
        if au is av:
            continue
        hub_route = HUB in (u, v)
        fig.add_artist(ConnectionPatch(
            xy[u], xy[v], coordsA="data", coordsB="data", axesA=au, axesB=av,
            color=HUB_ROUTE if hub_route else OTHER_ROUTE, lw=width(p, pmax),
            alpha=0.55, zorder=0.5, connectionstyle="arc3,rad=0.12"))

    # labels: metro airports with traffic on French routes, plus all overseas
    offsets = {"LFPG": (6, 9, "left"), "LFPO": (-7, -8, "right"), "LFOB": (-6, 6, "right"),
               "LFKB": (6, 3, "left"), "LFKJ": (-7, 0, "right"), "LFKF": (6, -3, "left"),
               "LFSB": (7, 4, "left"), "LFST": (7, 2, "left"), "LFLB": (7, -4, "left"),
               "LFLS": (7, 4, "left"), "LFBE": (7, -5, "left"), "LFBT": (7, -3, "left"),
               "LFBZ": (-7, 0, "right"), "LFRB": (6, 6, "left"), "LFRN": (6, 5, "left"),
               "LFMN": (7, 4, "left"), "LFML": (-4, -10, "right")}
    for n in F:
        dx, dy, ha = offsets.get(n, (7, 4, "left"))
        weight = "bold" if n in ("LFPG", "LFPO") else "normal"
        grey = F.degree(n) == 0
        a = axis_of(n)
        a.annotate(SHORT_NAMES.get(n, n), xy[n], xytext=(dx, dy), textcoords="offset points",
                   ha=ha, va="center", fontsize=8, weight=weight,
                   color=INK_2 if grey else INK, zorder=5)

    isolated = sum(1 for n in F if F.degree(n) == 0)
    fig.text(0.02, 0.965,
             f"French airports only, {G.graph['year']}: {F.number_of_nodes()} airports, "
             f"{F.number_of_edges()} domestic routes",
             fontsize=13, color=INK)
    fig.text(0.02, 0.93,
             f"Two Paris hubs: Orly carries the most domestic passengers, CDG has the most "
             f"domestic routes.\n"
             f"{isolated} airports have no domestic route in the data (grey labels).",
             fontsize=9, color=INK_2)
    legend(ax, F, smax, pmax, loc="upper right", foreign=False)
    fig.text(0.98, 0.012, "Data: Eurostat (routes), OurAirports (positions), Natural Earth (map).",
             ha="right", fontsize=7.5, color=INK_2)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    year = int(sys.argv[1]) if len(sys.argv) > 1 else MAIN_YEAR
    G = build_graph(year)
    coords, rings = load_coords(), load_countries()
    RESULTS_DIR.mkdir(exist_ok=True)
    map_full(G, coords, rings, RESULTS_DIR / f"map_full_{year}.png")
    map_french(G, coords, rings, RESULTS_DIR / f"map_french_{year}.png")
    print(f"Saved results/map_full_{year}.png and results/map_french_{year}.png")

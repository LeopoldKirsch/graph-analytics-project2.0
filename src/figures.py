"""Figures for the report, saved to results/ (PNG, 200 dpi).

- network_map_<year>.png      largest component, airports coloured by Louvain
                              community (weighted, seed 42, computed on the whole
                              graph), size = passengers, top hubs labelled.
                              Layout: Kamada-Kawai on hop distances, a topological
                              map (the geographic maps are in network_map.py).
- centrality_<year>.png       top 10 airports: share of passengers (weighted
                              degree) and betweenness, side by side.
- resilience_curves_<year>.png largest component and passengers kept as
                              airports are removed: targeted (static and
                              adaptive) vs random.
- resilience_scenarios.png    CDG / Orly / CDG + Orly vs random, 2019-2020-2024.
- comparison_years.png        key indicators 2019 / 2020 / 2024.

The analysis modules do not need to be run first: every figure recomputes from
build_graph and the analysis modules, so this script alone reproduces them.

Usage:
    python src/figures.py        # all figures, map / centrality / curves for 2024
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from build_graph import MAIN_YEAR, build_graph, build_graphs
from communities import louvain
from comparison import YEARS, year_table
from resilience import curves, scenarios

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# Fixed categorical order (never cycled); grey for "other".
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
           "#6250d6", "#e34948"]
GREY = "#b9b8b1"
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
YEAR_COLORS = {2019: PALETTE[0], 2020: PALETTE[1], 2024: PALETTE[2]}

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "figure.facecolor": "white", "savefig.dpi": 200, "savefig.bbox": "tight",
})


def _short(name: str) -> str:
    """'PARIS-CHARLES DE GAULLE' -> 'Paris-Charles De Gaulle', trimmed."""
    s = name.title()
    return s if len(s) <= 24 else s[:23] + "…"


def _save(fig, name: str) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / name
    fig.savefig(path)
    plt.close(fig)
    return path


def community_label(G: nx.Graph, c: set) -> str:
    """Name a community after its biggest airport."""
    hub = max(c, key=lambda n: G.degree(n, weight="passengers"))
    return f"{_short(G.nodes[hub]['name'])} group ({len(c)})"


def network_map(year: int = MAIN_YEAR, n_coloured: int = 5, n_labels: int = 8) -> Path:
    G = build_graph(year)
    comms = louvain(G, weighted=True)
    colour = {}
    for i, c in enumerate(comms):
        for n in c:
            colour[n] = PALETTE[i] if i < n_coloured else GREY

    # Drawn on the largest connected component: the 2-airport components would
    # otherwise be pushed to the corners and squash the layout.
    n_all = G.number_of_nodes()
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    pos = nx.kamada_kawai_layout(G)  # unweighted hop distances, deterministic
    strength = dict(G.degree(weight="passengers"))
    smax = max(strength.values())
    sizes = [15 + 900 * math.sqrt(strength[n] / smax) for n in G]

    fig, ax = plt.subplots(figsize=(13, 9))
    ax.grid(False)
    ax.axis("off")
    nx.draw_networkx_edges(G, pos, ax=ax, width=0.4, edge_color=GREY, alpha=0.6)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes,
                           node_color=[colour[n] for n in G],
                           edgecolors="white", linewidths=0.8)
    top = sorted(G, key=lambda n: -strength[n])[:n_labels]
    for n in top:
        x, y = pos[n]
        ax.annotate(_short(G.nodes[n]["name"]), (x, y), xytext=(6, 6),
                    textcoords="offset points", fontsize=8, color=INK,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    handles = [plt.Line2D([], [], marker="o", ls="", color=PALETTE[i], markersize=8,
                          label=community_label(G, c))
               for i, c in enumerate(comms[:n_coloured])]
    if len(comms) > n_coloured:
        rest = sum(len(c) for c in comms[n_coloured:])
        handles.append(plt.Line2D([], [], marker="o", ls="", color=GREY, markersize=8,
                                  label=f"Other small groups ({rest}, partly outside the map)"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 0.98), frameon=False,
              fontsize=9, title="Louvain community (weighted, seed 42)", title_fontsize=9,
              alignment="left")
    scope = ("" if G.number_of_nodes() == n_all
             else f"largest component, {G.number_of_nodes()} of ")
    ax.set_title(f"French air network {year}: {scope}{n_all} airports, "
                 f"{G.number_of_edges()} routes. Node size = passengers", loc="left")
    return _save(fig, f"network_map_{year}.png")


def centrality_chart(year: int = MAIN_YEAR, top: int = 10) -> Path:
    G = build_graph(year)
    strength = dict(G.degree(weight="passengers"))
    total = sum(strength.values()) / 2
    share = {n: s / total for n, s in strength.items()}
    bc = nx.betweenness_centrality(G)
    nodes = sorted(G, key=lambda n: -share[n])[:top][::-1]
    labels = [_short(G.nodes[n]["name"]) for n in nodes]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, values, title, fmt in (
        (axes[0], [share[n] for n in nodes], "Share of passengers", "{:.0%}"),
        (axes[1], [bc[n] for n in nodes], "Betweenness (unweighted, normalised)", "{:.2f}"),
    ):
        bars = ax.barh(labels, values, color=PALETTE[0], height=0.65)
        for b, v in zip(bars, values):
            ax.text(b.get_width(), b.get_y() + b.get_height() / 2, " " + fmt.format(v),
                    va="center", fontsize=8, color=INK_2)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(values) * 1.18)
    axes[0].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    fig.suptitle(f"Top {top} airports by passengers, {year}", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    return _save(fig, f"centrality_{year}.png")


STRATEGY_STYLE = {
    "degree": ("Targeted: most routes first", PALETTE[1]),
    "passengers": ("Targeted: most passengers first", PALETTE[4]),
    "betweenness": ("Targeted: highest betweenness first", PALETTE[6]),
    "betweenness_adaptive": ("Adaptive: betweenness recomputed", PALETTE[3]),
    "random": ("Random (mean, 5-95 % band)", PALETTE[0]),
}


def resilience_curves(year: int = MAIN_YEAR) -> Path:
    G = build_graph(year)
    c = curves(G)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for ax, col, title in ((axes[0], "lcc_share", "Airports in the largest component"),
                           (axes[1], "passengers_kept", "Passengers still flying")):
        for strat, (label, col_hex) in STRATEGY_STYLE.items():
            d = c[c["strategy"] == strat]
            ax.plot(d["removed"], d[col], color=col_hex, lw=2, marker="o", ms=3.5,
                    label=label)
            if strat == "random":
                ax.fill_between(d["removed"], d[f"{col}_p05"], d[f"{col}_p95"],
                                color=col_hex, alpha=0.15, lw=0)
        ax.set_title(title, loc="left")
        ax.set_ylim(0, 1.02)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_xlabel("Airports removed")
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
    d = c[c["strategy"] == "degree"]
    first = d.iloc[1]
    axes[0].annotate(f"CDG removed: {first['lcc_share']:.0%}", (1, first["lcc_share"]),
                     xytext=(12, -4), textcoords="offset points", fontsize=8, color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"Resilience {year}: targeted removal of hubs vs random failures",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, f"resilience_curves_{year}.png")


def resilience_scenarios(years=YEARS) -> Path:
    scen = pd.concat([scenarios(build_graph(y)) for y in years], ignore_index=True)
    scen["scenario"] = scen["scenario"].str.replace(r" \(mean of \d+\)", " (mean)", regex=True)
    order = ["none", "CDG", "Orly", "CDG + Orly", "random 2 (mean)"]
    names = ["Intact", "No CDG", "No Orly", "No CDG + Orly", "2 random airports"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    width = 0.26
    for ax, col, title in ((axes[0], "lcc_share", "Airports in the largest component"),
                           (axes[1], "passengers_kept", "Passengers still flying")):
        for j, y in enumerate(years):
            d = scen[scen["year"] == y].set_index("scenario").loc[order, col]
            xs = [i + (j - 1) * width for i in range(len(order))]
            ax.bar(xs, d.values, width=width - 0.03, color=YEAR_COLORS[y], label=str(y))
        ax.set_xticks(range(len(order)), names, rotation=15)
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, ncol=3, loc="upper right",
               bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("Removing Paris hubs vs removing airports at random",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, "resilience_scenarios.png")


def comparison_chart(years=YEARS) -> Path:
    t = year_table(build_graphs(years))
    panels = [
        ("passengers_m", "Passengers (millions)", "{:.0f}"),
        ("edges", "Routes (edges)", "{:.0f}"),
        ("cdg_route_share", "Share of routes touching CDG", "{:.0%}"),
        ("betweenness_CDG", "CDG betweenness", "{:.2f}"),
        ("effective_airports", "Effective number of airports (1 / HHI)", "{:.1f}"),
        ("lcc_share_without_cdg", "Largest component without CDG", "{:.0%}"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, (col, title, fmt) in zip(axes.flat, panels):
        vals = t[col]
        bars = ax.bar([str(y) for y in vals.index], vals.values,
                      color=[YEAR_COLORS[y] for y in vals.index], width=0.6)
        for b, v in zip(bars, vals.values):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v),
                    ha="center", va="bottom", fontsize=9, color=INK)
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(axis="x", visible=False)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_ylim(0, vals.max() * 1.18)
    fig.suptitle("2019 (pre-COVID), 2020 (COVID), 2024", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    return _save(fig, "comparison_years.png")


def main(year: int = MAIN_YEAR) -> None:
    for f in (lambda: network_map(year), lambda: centrality_chart(year),
              lambda: resilience_curves(year), resilience_scenarios, comparison_chart):
        print("wrote", f())


if __name__ == "__main__":
    main()

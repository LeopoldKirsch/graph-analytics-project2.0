"""Option 3 (explored, kept as context): the network every year, 2010-2024.

The same graph model is rebuilt for each year with build_graph, and a few
indicators are tracked:
- size: nodes (French / foreign), edges, density, passengers
- connectivity: components, size of the largest component, its diameter and
  radius (unweighted hops)
- dependence on CDG (LFPG): its degree share, betweenness, share of
  passengers on its routes, and how many components remain without it
- stability: edges kept / gained / lost versus the previous year

Caveat: only airports that report to Eurostat have all their routes, and the
number of reporting French airports changes a little from year to year
(39-42, but about 30 in 2020-2021 when traffic fell below the reporting
threshold): column reporting_airports. Year-to-year changes in small routes
partly reflect who reports.

Usage:
    python src/time_series.py   # writes results/time_series.csv and .png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from build_graph import YEARS, build_graph, load_routes

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
HUB = "LFPG"
PARIS = ("LFPG", "LFPO")


def year_metrics(G: nx.Graph) -> dict:
    n = G.number_of_nodes()
    total_pax = sum(p for _, _, p in G.edges(data="passengers"))
    lcc = G.subgraph(max(nx.connected_components(G), key=len))
    paris_pax = sum(
        p for u, v, p in G.edges(data="passengers") if u in PARIS or v in PARIS
    )

    without_hub = G.copy()
    without_hub.remove_node(HUB)

    return {
        "year": G.graph["year"],
        "nodes": n,
        "french_nodes": sum(1 for _, f in G.nodes(data="french") if f),
        "foreign_nodes": sum(1 for _, f in G.nodes(data="french") if not f),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "mean_degree": 2 * G.number_of_edges() / n,
        "passengers_m": total_pax / 1e6,
        "components": nx.number_connected_components(G),
        "lcc_nodes": lcc.number_of_nodes(),
        "lcc_diameter": nx.diameter(lcc),
        "lcc_radius": nx.radius(lcc),
        "cdg_degree": G.degree(HUB),
        "cdg_degree_share": G.degree(HUB) / (n - 1),
        "cdg_betweenness": nx.betweenness_centrality(G)[HUB],
        "cdg_passenger_share": G.degree(HUB, weight="passengers") / total_pax,
        "paris_passenger_share": paris_pax / total_pax,
        "components_without_cdg": nx.number_connected_components(without_hub),
        "isolated_without_cdg": nx.number_of_isolates(without_hub),
    }


def edge_turnover(prev: nx.Graph, cur: nx.Graph) -> dict:
    """Routes kept, gained and lost versus the previous year."""
    a = {frozenset(e) for e in prev.edges}
    b = {frozenset(e) for e in cur.edges}
    return {
        "edges_kept": len(a & b),
        "edges_new": len(b - a),
        "edges_lost": len(a - b),
        "edge_jaccard": len(a & b) / len(a | b),
    }


def run(years=YEARS) -> pd.DataFrame:
    rows, prev = [], None
    for y in years:
        G = build_graph(y)
        row = year_metrics(G)
        r = load_routes(y)
        row["reporting_airports"] = r.loc[r["reporting_country"] == "FR", "reporting"].nunique() + int(
            (r["reporting_country"] == "CH").any())
        if prev is not None:
            row.update(edge_turnover(prev, G))
        rows.append(row)
        prev = G
    return pd.DataFrame(rows).set_index("year")


# Small multiples: one panel per indicator, each with its own y-axis
# (never two scales on one axis). Single series per panel, so one hue and no
# legend: the panel title names it.
PANELS = [
    ("edges", "Routes (edges)", "{:.0f}"),
    ("passengers_m", "Passengers (millions)", "{:.0f}"),
    ("cdg_betweenness", "CDG betweenness centrality", "{:.2f}"),
    # two series, same unit: one axis, legend + direct end labels
    ([("paris_passenger_share", "Paris (CDG + Orly)"),
      ("cdg_passenger_share", "CDG")],
     "Share of passengers on Paris routes", "{:.0%}"),
    ("components_without_cdg", "Components after removing CDG", "{:.0f}"),
    ("edge_jaccard", "Routes kept vs previous year (Jaccard)", "{:.2f}"),
]
INK = "#0b0b0b"
INK_2 = "#52514e"
SERIES = "#2a78d6"
SERIES_2 = "#eb6834"
GRID = "#e4e3df"
EVENTS = {2020: "COVID"}


def _marker(ax, x, y, color):
    ax.plot(x, y, "o", color=color, markersize=5,
            markeredgecolor="white", markeredgewidth=1.5)


def _plot_single(ax, s: pd.Series, fmt: str) -> None:
    ax.plot(s.index, s.values, color=SERIES, linewidth=2)
    # direct labels on the years the report compares: 2019 (left of its
    # point), 2020 and the last year (right of theirs)
    for year, dx, ha in ((2019, -6, "right"), (2020, 6, "left"),
                         (s.index.max(), 0, "center")):
        if year in s.index:
            _marker(ax, year, s[year], SERIES)
            ax.annotate(fmt.format(s[year]), (year, s[year]),
                        textcoords="offset points", xytext=(dx, 7),
                        ha=ha, fontsize=8, color=INK)


def _plot_multi(ax, df: pd.DataFrame, cols, fmt: str) -> None:
    # several series sharing one unit: legend, plus the last value labelled
    # with the series name so identity never rests on colour alone
    for (col, name), color in zip(cols, (SERIES, SERIES_2)):
        s = df[col].dropna()
        ax.plot(s.index, s.values, color=color, linewidth=2, label=name)
        last = s.index.max()
        _marker(ax, last, s[last], color)
        ax.annotate(f"{name.split(' ')[0]} {fmt.format(s[last])}",
                    (last, s[last]), textcoords="offset points",
                    xytext=(0, 7), ha="right", fontsize=8, color=INK)
    ax.legend(loc="center left", bbox_to_anchor=(0, 0.68), fontsize=8,
              frameon=False, labelcolor=INK)


def plot(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(10, 9), sharex=True)
    for ax, (col, title, fmt) in zip(axes.flat, PANELS):
        if isinstance(col, list):
            _plot_multi(ax, df, col, fmt)
        else:
            _plot_single(ax, df[col].dropna(), fmt)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=9, length=0)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _, fmt=fmt: fmt.format(v))
        )
        for year in EVENTS:
            ax.axvline(year, color=INK_2, linewidth=0.8, linestyle=":")
    for year, label in EVENTS.items():
        axes[-1, 0].annotate(label, (year, 0), xycoords=("data", "axes fraction"),
                             xytext=(3, 4), textcoords="offset points",
                             fontsize=7.5, color=INK_2)
    fig.suptitle("French air network, 2010-2024 (Eurostat routes)",
                 x=0.06, ha="left", fontsize=12.5, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)
    df = run()
    df.round(4).to_csv(RESULTS_DIR / "time_series.csv")
    plot(df, RESULTS_DIR / "time_series.png")
    pd.set_option("display.width", 250)
    print(df.round(3).to_string())
    print(f"\nSaved {RESULTS_DIR / 'time_series.csv'} and time_series.png")

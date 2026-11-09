"""Null model: is the network's structure more than its degree sequence?

We compare the real graph with random graphs that keep every airport's
number of routes (degree-preserving double-edge swaps: (a-b, c-d) ->
(a-d, c-b)). Swaps that would create a self-loop, a duplicate route or a
foreign-foreign route are rejected, because the route data can never contain
foreign-foreign routes: the random graphs obey the same data constraint.

For each metric we report the real value, the mean and standard deviation
over the random graphs, the z-score and the empirical two-sided p-value.

Because degrees are fixed, degree and degree centralization are identical
by construction. The question is what the *wiring* adds: e.g. is CDG's
betweenness higher than a hub with 99 routes would get anyway? Are
communities stronger than chance? Do hubs avoid each other (assortativity)?

Modularity is computed unweighted (random graphs have no meaningful
passenger weights), with Louvain, seed fixed per graph.

Usage:
    python src/null_model.py            # 2024, 500 random graphs per view
    python src/null_model.py 2024 100   # fewer samples, faster
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.community import louvain_communities, modularity

from build_graph import MAIN_YEAR, build_graph

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
SEED = 42


def constrained_swaps(G: nx.Graph, rng: random.Random, swaps_per_edge: int = 10) -> nx.Graph:
    """Degree-preserving randomisation that never creates foreign-foreign edges."""
    french = {n: G.nodes[n]["french"] for n in G}
    edges = [tuple(e) for e in G.edges()]
    present = {frozenset(e) for e in edges}
    target, done, tries = swaps_per_edge * len(edges), 0, 0
    while done < target and tries < 100 * target:
        tries += 1
        i, j = rng.randrange(len(edges)), rng.randrange(len(edges))
        if i == j:
            continue
        a, b = edges[i]
        c, d = edges[j]
        if rng.random() < 0.5:
            c, d = d, c
        new1, new2 = frozenset((a, d)), frozenset((c, b))
        if (a == d or c == b or new1 == new2 or new1 in present or new2 in present
                or not (french[a] or french[d]) or not (french[c] or french[b])):
            continue
        present -= {frozenset((a, b)), frozenset((c, d))}
        present |= {new1, new2}
        edges[i], edges[j] = (a, d), (c, b)
        done += 1
    R = nx.Graph()
    R.add_nodes_from(G.nodes(data=True))
    R.add_edges_from(edges)
    return R


def canonical(G: nx.Graph) -> nx.Graph:
    """Copy of G with nodes and edges inserted in sorted order (reproducible iteration)."""
    H = nx.Graph()
    H.add_nodes_from(sorted(G.nodes(data=True)))
    H.add_edges_from(sorted((min(u, v), max(u, v), d) for u, v, d in G.edges(data=True)))
    return H


def hub_density(G: nx.Graph, k: int = 10) -> float:
    """Density among the k airports with the most routes (the hubs).

    Degrees are preserved by the swaps, so the same k airports are compared
    in the real and random graphs (ties broken by ICAO code)."""
    hubs = sorted(G, key=lambda n: (-G.degree(n), n))[:k]
    return nx.density(G.subgraph(hubs))


def metrics(G: nx.Graph, seed: int = SEED) -> dict:
    lcc = G.subgraph(max(nx.connected_components(G), key=len))
    btw = nx.betweenness_centrality(G)
    bmax = max(btw.values())
    comms = louvain_communities(G, seed=seed)
    return {
        "cdg_betweenness": btw.get("LFPG", np.nan),
        "orly_betweenness": btw.get("LFPO", np.nan),
        "betweenness_centralization": sum(bmax - b for b in btw.values()) / (len(G) - 1),
        "degree_assortativity": nx.degree_assortativity_coefficient(G),
        "hub_density_top10": hub_density(G),
        "modularity_louvain": modularity(G, comms),
        "communities": len(comms),
        "components": nx.number_connected_components(G),
        "avg_shortest_path_lcc": nx.average_shortest_path_length(lcc),
        "diameter_lcc": nx.diameter(lcc),
    }


def compare(G: nx.Graph, n: int, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(SEED)
    real = metrics(G)
    null = pd.DataFrame([metrics(constrained_swaps(G, rng), seed=SEED + i) for i in range(n)])
    rows = []
    for m, v in real.items():
        s = null[m].dropna()
        sd = s.std()
        rows.append({
            "view": label, "metric": m, "real": v,
            "null_mean": s.mean(), "null_sd": sd,
            "z": (v - s.mean()) / sd if sd > 0 else np.nan,
            "p_two_sided": min(1.0, 2 * min((s >= v).mean(), (s <= v).mean())),
        })
    null.insert(0, "view", label)
    return pd.DataFrame(rows), null


def plot(table: pd.DataFrame, null: pd.DataFrame, path: Path) -> None:
    """Small multiples: null distribution (histogram) and real value (line)."""
    shown = ["cdg_betweenness", "betweenness_centralization", "degree_assortativity",
             "modularity_louvain"]
    titles = {"cdg_betweenness": "CDG betweenness",
              "betweenness_centralization": "Betweenness centralization",
              "degree_assortativity": "Degree assortativity",
              "modularity_louvain": "Modularity (Louvain)"}
    views = list(table["view"].unique())
    fig, axes = plt.subplots(len(views), len(shown), figsize=(13, 3.2 * len(views)))
    axes = np.atleast_2d(axes)
    for r, view in enumerate(views):
        for c, m in enumerate(shown):
            ax = axes[r, c]
            vals = null.loc[null["view"] == view, m].dropna()
            row = table[(table["view"] == view) & (table["metric"] == m)].iloc[0]
            ax.hist(vals, bins=25, color="#b7d3f6", edgecolor="white", linewidth=0.8)
            ax.axvline(row["real"], color="#eb6834", linewidth=2)
            ax.annotate(f"real {row['real']:.2f}\nz = {row['z']:.1f}", (row["real"], 1),
                        xycoords=("data", "axes fraction"), xytext=(4, -4),
                        textcoords="offset points", va="top", fontsize=8, color="#0b0b0b")
            ax.set_title(f"{titles[m]} ({view})" if c == 0 else titles[m],
                         loc="left", fontsize=9.5, color="#0b0b0b")
            ax.set_yticks([])
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
            ax.spines["bottom"].set_color("#d6d5d0")
            ax.tick_params(colors="#52514e", labelsize=8, length=0)
    fig.suptitle("Real network (orange line) vs degree-preserving random graphs (blue histogram)",
                 x=0.01, ha="left", fontsize=11.5, color="#0b0b0b")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def run(year: int = MAIN_YEAR, n: int = 500):
    G = build_graph(year)
    F = G.subgraph([v for v, f in G.nodes(data="french") if f]).copy()
    F.remove_nodes_from(list(nx.isolates(F)))  # isolated airports cannot be rewired
    # subgraph() iterates over a set of node codes, whose order changes with
    # Python's string hashing from one run to the next; rebuild F in sorted
    # order so the swaps and Louvain see the same graph every time.
    F = canonical(F)
    tables, nulls = zip(*(compare(H, n, label) for H, label in
                          ((G, "full network"), (F, "domestic"))))
    return pd.concat(tables, ignore_index=True), pd.concat(nulls, ignore_index=True)


if __name__ == "__main__":
    year = int(sys.argv[1]) if len(sys.argv) > 1 else MAIN_YEAR
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    table, null = run(year, n)
    RESULTS_DIR.mkdir(exist_ok=True)
    pd.set_option("display.width", 200, "display.float_format", "{:.3f}".format)
    print(f"=== {year}, {n} random graphs per view ===\n{table.to_string(index=False)}")
    table.to_csv(RESULTS_DIR / f"null_model_{year}.csv", index=False)
    plot(table, null, RESULTS_DIR / f"null_model_{year}.png")
    print(f"\nSaved results/null_model_{year}.csv and .png")

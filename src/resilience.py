"""Resilience of the French air network to the loss of its hubs (NetworkX).

Question: how fragile does the concentration on Paris-CDG make the network?

Method: airports (nodes) are removed and the remaining graph is measured.

1. Fixed scenarios: remove CDG (LFPG), Orly (LFPO), then CDG + Orly together.
   They are compared with removing the same number of airports at random
   (``N_RANDOM`` draws, seeds 0..N_RANDOM-1) so that the effect of the hub can be
   separated from the effect of simply losing one or two nodes.
2. Removal curves: airports are removed one by one, up to ``MAX_REMOVED``, in
   order of
     - degree (number of routes),
     - weighted degree (annual passengers),
     - betweenness (unweighted, computed once on the intact graph),
     - random order (mean and 5-95 % band over ``N_RANDOM`` seeds).
   The targeted orders are computed once on the intact graph (static attack),
   which is the standard simple version and easy to reproduce.
   For comparison, an adaptive attack recomputes betweenness after every
   removal and always removes the currently most central airport
   ("betweenness_adaptive"); it is the more damaging, worst-case version.

Measures after each removal (all on the remaining graph):
- size of the largest connected component (LCC), in nodes and as a share of
  the original number of airports;
- number of connected components and number of isolated airports;
- share of pairs of the French airports still in the graph that can reach
  each other (any path, through any airport): the "can I still fly between two
  French airports" view of connectivity. The removed airports themselves are
  left out of the count, so the measure is about disconnection only;
- share of the year's passengers carried by routes that still
  exist (routes touching a removed airport are lost);
- diameter of the LCC in unweighted hops (not computed for the random
  removal curves, to keep the run short).

The graph is undirected, so connectivity means ordinary connectivity.

Usage:
    python src/resilience.py            # 2019, 2020, 2024
    python src/resilience.py 2024       # one year
Outputs: results/resilience_scenarios.csv and results/resilience_curves_<year>.csv
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import networkx as nx
import pandas as pd

from build_graph import build_graph

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
CDG, ORLY = "LFPG", "LFPO"
SCENARIOS = {"none": [], "CDG": [CDG], "Orly": [ORLY], "CDG + Orly": [CDG, ORLY]}
N_RANDOM = 100
MAX_REMOVED = 20


def french_pairs_connected(H: nx.Graph, french: set) -> float:
    """Share of pairs of the French airports still in H that can reach each other."""
    french = french & set(H)
    total = len(french) * (len(french) - 1) / 2
    if total == 0:
        return float("nan")
    connected = 0
    for comp in nx.connected_components(H):
        k = len(comp & french)
        connected += k * (k - 1) / 2
    return connected / total


def measure(H: nx.Graph, G: nx.Graph, french: set, total_pax: float,
            diameter: bool = True) -> dict:
    """Connectivity of H (G minus some airports), relative to the intact G."""
    if H.number_of_nodes() == 0:
        return {"lcc_nodes": 0, "lcc_share": 0.0, "components": 0, "isolated": 0,
                "french_pairs_connected": 0.0, "passengers_kept": 0.0,
                "lcc_diameter": 0}
    comps = sorted(nx.connected_components(H), key=len, reverse=True)
    lcc = comps[0]
    return {
        "lcc_nodes": len(lcc),
        "lcc_share": len(lcc) / G.number_of_nodes(),
        "components": len(comps),
        "isolated": sum(1 for c in comps if len(c) == 1),
        "french_pairs_connected": french_pairs_connected(H, french),
        "passengers_kept": sum(p for _, _, p in H.edges(data="passengers")) / total_pax,
        "lcc_diameter": (nx.diameter(H.subgraph(lcc)) if len(lcc) > 1 else 0)
        if diameter else float("nan"),
    }


def _context(G: nx.Graph):
    french = {n for n, f in G.nodes(data="french") if f}
    total_pax = sum(p for _, _, p in G.edges(data="passengers"))
    return french, total_pax


def remove(G: nx.Graph, nodes) -> nx.Graph:
    H = G.copy()
    H.remove_nodes_from(nodes)
    return H


def evaluate(G: nx.Graph, nodes) -> dict:
    french, total_pax = _context(G)
    H = remove(G, nodes)
    return measure(H, G, french, total_pax)


def scenarios(G: nx.Graph, n_random: int = N_RANDOM) -> pd.DataFrame:
    """Fixed scenarios (CDG, Orly, CDG + Orly) and their random baselines."""
    rows = []
    nodes = sorted(G)
    for name, removed in SCENARIOS.items():
        rows.append({"year": G.graph["year"], "scenario": name,
                     "removed": len(removed), **evaluate(G, removed)})
    for k in (1, 2):
        draws = pd.DataFrame(
            [evaluate(G, random.Random(s).sample(nodes, k)) for s in range(n_random)]
        )
        mean = draws.mean().to_dict()
        rows.append({"year": G.graph["year"], "scenario": f"random {k} (mean of {n_random})",
                     "removed": k, **mean})
    return pd.DataFrame(rows)


def attack_order(G: nx.Graph, strategy: str, seed: int | None = None) -> list:
    """Order in which airports are removed (static: computed on the intact graph)."""
    if strategy == "degree":
        score = dict(G.degree())
    elif strategy == "passengers":
        score = dict(G.degree(weight="passengers"))
    elif strategy == "betweenness":
        score = nx.betweenness_centrality(G)  # unweighted hops
    elif strategy == "random":
        order = sorted(G)
        random.Random(seed).shuffle(order)
        return order
    else:
        raise ValueError(strategy)
    # ties broken by code so the order is reproducible
    return sorted(G, key=lambda n: (-score[n], n))


def removal_curve(G: nx.Graph, order: list, max_removed: int = MAX_REMOVED,
                  diameter: bool = True) -> pd.DataFrame:
    french, total_pax = _context(G)
    rows = []
    H = G.copy()
    for k in range(max_removed + 1):
        if k > 0:
            H.remove_node(order[k - 1])
        m = measure(H, G, french, total_pax, diameter)
        rows.append({"removed": k, "last_removed": order[k - 1] if k else "", **m})
    return pd.DataFrame(rows)


def adaptive_order(G: nx.Graph, max_removed: int = MAX_REMOVED) -> list:
    """Adaptive betweenness attack: recompute betweenness after each removal.

    Ties are broken by degree in the current graph, then by code, so the order
    is reproducible."""
    H = G.copy()
    order = []
    for _ in range(min(max_removed, H.number_of_nodes())):
        bc = nx.betweenness_centrality(H)  # unweighted hops
        target = min(H, key=lambda n: (-bc[n], -H.degree(n), n))
        order.append(target)
        H.remove_node(target)
    return order


def curves(G: nx.Graph, n_random: int = N_RANDOM, max_removed: int = MAX_REMOVED) -> pd.DataFrame:
    """Removal curves for the targeted strategies and the random baseline."""
    parts = []
    for strategy in ("degree", "passengers", "betweenness"):
        c = removal_curve(G, attack_order(G, strategy), max_removed)
        c.insert(0, "strategy", strategy)
        parts.append(c)
    c = removal_curve(G, adaptive_order(G, max_removed), max_removed)
    c.insert(0, "strategy", "betweenness_adaptive")
    parts.append(c)

    rand = pd.concat(
        [removal_curve(G, attack_order(G, "random", s), max_removed, diameter=False).assign(seed=s)
         for s in range(n_random)]
    )
    cols = ["lcc_share", "french_pairs_connected", "passengers_kept", "components"]
    agg = rand.groupby("removed")[cols]
    mean = agg.mean()
    low = agg.quantile(0.05).add_suffix("_p05")
    high = agg.quantile(0.95).add_suffix("_p95")
    r = pd.concat([mean, low, high], axis=1).reset_index()
    r.insert(0, "strategy", "random")
    parts.append(r)

    out = pd.concat(parts, ignore_index=True)
    out.insert(0, "year", G.graph["year"])
    return out


def main(years) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    scen = []
    for y in years:
        G = build_graph(y)
        scen.append(scenarios(G))
        c = curves(G)
        c.to_csv(RESULTS_DIR / f"resilience_curves_{y}.csv", index=False)
        first = c[c["strategy"] == "degree"].head(6)[["removed", "last_removed", "lcc_share",
                                                      "french_pairs_connected"]]
        print(f"\n=== {y}: first removals by degree ===")
        print(first.round(3).to_string(index=False))
    scen = pd.concat(scen, ignore_index=True)
    scen.to_csv(RESULTS_DIR / "resilience_scenarios.csv", index=False)
    print()
    print(scen.round(3).to_string(index=False))


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [2019, 2020, 2024])

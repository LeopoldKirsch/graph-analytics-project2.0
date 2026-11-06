"""Community detection on the French air network (Louvain, NetworkX).

Question: beyond "Paris vs the rest", do airports form natural groups?

Method:
- Louvain (``networkx.community.louvain_communities``), resolution 1.0,
  ``seed=42`` so the partition is reproducible.
- Two versions are compared:
    * weighted by annual passengers (``weight="passengers"``): an edge counts
      as much as the traffic it carries, so communities follow the volume of
      flows;
    * unweighted (``weight=None``): every route counts the same, so
      communities follow the route structure only.
- Quality is measured by modularity Q (same weight as the partition).
- Stability: Louvain is randomised, so the partition is recomputed for 20
  seeds and compared with the seed-42 one using the adjusted Rand index (ARI)
  and the spread of Q. A high mean ARI means the groups are not an artefact of
  one seed.
- Communities are run on the whole graph, small disconnected components
  included: each of those components necessarily becomes its own community
  (Louvain never merges nodes that share no edge). They are flagged as such in
  the composition table.

Communities are named from their composition: share of French metro / overseas /
foreign airports, dominant ICAO world regions and biggest airports by
passengers (weighted degree).

Usage:
    python src/communities.py            # 2019, 2020, 2024
    python src/communities.py 2024       # one year
Outputs: results/communities_<year>.csv (one row per community) and
results/communities_summary.csv (one row per year and weighting).
"""

from __future__ import annotations

import sys
from collections import Counter
from itertools import combinations
from math import comb
from pathlib import Path

import networkx as nx
import pandas as pd

from build_graph import build_graph

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
SEED = 42
STABILITY_SEEDS = range(20)


def louvain(G: nx.Graph, weighted: bool = True, seed: int = SEED,
            resolution: float = 1.0) -> list[set]:
    """Louvain communities, largest first."""
    weight = "passengers" if weighted else None
    comms = nx.community.louvain_communities(
        G, weight=weight, resolution=resolution, seed=seed
    )
    return sorted(comms, key=lambda c: (-len(c), sorted(c)[0]))


def modularity(G: nx.Graph, comms: list[set], weighted: bool = True) -> float:
    return nx.community.modularity(G, comms, weight="passengers" if weighted else None)


def membership(comms: list[set]) -> dict:
    """Node -> community index."""
    return {n: i for i, c in enumerate(comms) for n in c}


def adjusted_rand_index(a: dict, b: dict) -> float:
    """ARI between two partitions given as node -> label dicts (same nodes).

    Written out here to avoid adding scikit-learn as a dependency.
    """
    nodes = list(a)
    n = len(nodes)
    table = Counter((a[x], b[x]) for x in nodes)
    rows = Counter(a[x] for x in nodes)
    cols = Counter(b[x] for x in nodes)
    sum_ij = sum(comb(v, 2) for v in table.values())
    sum_a = sum(comb(v, 2) for v in rows.values())
    sum_b = sum(comb(v, 2) for v in cols.values())
    expected = sum_a * sum_b / comb(n, 2)
    max_index = (sum_a + sum_b) / 2
    if max_index == expected:
        return 1.0
    return (sum_ij - expected) / (max_index - expected)


def stability(G: nx.Graph, weighted: bool = True, seeds=STABILITY_SEEDS) -> dict:
    """How much the partition depends on the random seed."""
    ref = membership(louvain(G, weighted, SEED))
    parts = [louvain(G, weighted, s) for s in seeds]
    aris = [adjusted_rand_index(ref, membership(p)) for p in parts]
    qs = [modularity(G, p, weighted) for p in parts]
    return {
        "ari_mean": sum(aris) / len(aris),
        "ari_min": min(aris),
        "q_min": min(qs),
        "q_max": max(qs),
        "n_comms_min": min(len(p) for p in parts),
        "n_comms_max": max(len(p) for p in parts),
    }


def composition(G: nx.Graph, comms: list[set], top: int = 5) -> pd.DataFrame:
    """One row per community: size, zones, regions, main airports, traffic."""
    total_strength = sum(d for _, d in G.degree(weight="passengers"))
    components = {frozenset(c) for c in nx.connected_components(G)}
    rows = []
    for i, c in enumerate(comms):
        zones = Counter(G.nodes[n]["zone"] for n in c)
        regions = Counter(G.nodes[n]["icao_region"] for n in c)
        strength = {n: G.degree(n, weight="passengers") for n in c}
        # ties broken by name / code so the labels do not depend on set order
        main = sorted(c, key=lambda n: (-strength[n], n))[:top]
        top_regions = sorted(regions.items(), key=lambda rk: (-rk[1], rk[0]))[:3]
        sub = G.subgraph(c)
        internal_pax = sum(p for _, _, p in sub.edges(data="passengers"))
        rows.append({
            "community": i,
            "size": len(c),
            "metro": zones.get("metro", 0),
            "overseas": zones.get("overseas", 0),
            "foreign": zones.get("foreign", 0),
            "top_regions": ", ".join(f"{r} ({k})" for r, k in top_regions),
            "main_airports": ", ".join(f"{G.nodes[n]['name']} [{n}]" for n in main),
            "internal_edges": sub.number_of_edges(),
            "internal_passengers": internal_pax,
            # share of the network's weighted degree held by this community
            "strength_share": sum(strength.values()) / total_strength,
            "is_separate_component": frozenset(c) in components,
        })
    return pd.DataFrame(rows)


def compare_weightings(G: nx.Graph) -> float:
    """ARI between weighted and unweighted partitions (1 = identical)."""
    return adjusted_rand_index(
        membership(louvain(G, weighted=True)), membership(louvain(G, weighted=False))
    )


def paris_split(G: nx.Graph, comms: list[set]) -> dict:
    """Where CDG and Orly land: same community or not, and community sizes."""
    m = membership(comms)
    return {
        "cdg_comm": m.get("LFPG"),
        "orly_comm": m.get("LFPO"),
        "cdg_orly_same": m.get("LFPG") == m.get("LFPO"),
    }


def analyse(year: int) -> tuple[pd.DataFrame, list[dict]]:
    G = build_graph(year)
    summary = []
    tables = []
    for weighted in (True, False):
        comms = louvain(G, weighted)
        tab = composition(G, comms)
        tab.insert(0, "weighted", weighted)
        tab.insert(0, "year", year)
        tables.append(tab)
        summary.append({
            "year": year,
            "weighted": weighted,
            "n_communities": len(comms),
            "n_separate_components": int(tab["is_separate_component"].sum()),
            "modularity": modularity(G, comms, weighted),
            "largest_share": len(comms[0]) / G.number_of_nodes(),
            **paris_split(G, comms),
            **stability(G, weighted),
            "ari_weighted_vs_unweighted": compare_weightings(G),
        })
    return pd.concat(tables, ignore_index=True), summary


def main(years) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    rows = []
    for y in years:
        tab, summ = analyse(y)
        tab.to_csv(RESULTS_DIR / f"communities_{y}.csv", index=False)
        rows += summ
        print(f"\n=== {y} (weighted by passengers) ===")
        print(tab[tab["weighted"]][["community", "size", "metro", "overseas",
                                    "foreign", "main_airports"]].to_string(index=False))
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS_DIR / "communities_summary.csv", index=False)
    print()
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [2019, 2020, 2024])

"""Required graph metrics: size and type, connectivity, diameter and radius,
plus articulation points and bridges (direct measures of fragility).

Conventions (to state in the report):
- Graph: undirected, simple, weighted (see build_graph.py).
- Distances are counted in hops (number of flights), unweighted: the
  question is how many connections separate two airports, not the km flown.
- The graph is not connected, so diameter, radius, center, periphery and
  average shortest path length are computed on the largest connected
  component (LCC). Distances between components are undefined.
- Directed connectivity (weak / strong) does not apply: Eurostat route
  figures combine both directions of a route, so the graph is undirected.

Usage:
    python src/core_metrics.py          # 2024, writes results/core_*.csv
    python src/core_metrics.py 2019
"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import pandas as pd

from build_graph import MAIN_YEAR, build_graph

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def largest_component(G: nx.Graph) -> nx.Graph:
    return G.subgraph(max(nx.connected_components(G), key=len)).copy()


def size_and_type(G: nx.Graph) -> dict:
    deg = [d for _, d in G.degree()]
    zones = pd.Series(dict(G.nodes(data="zone"))).value_counts()
    n = G.number_of_nodes()
    return {
        "nodes": n,
        "nodes_french_metro": int(zones.get("metro", 0)),
        "nodes_french_overseas": int(zones.get("overseas", 0)),
        "nodes_foreign": int(zones.get("foreign", 0)),
        "edges": G.number_of_edges(),
        "directed": G.is_directed(),
        "multigraph": G.is_multigraph(),
        "self_loops": nx.number_of_selfloops(G),
        "density": nx.density(G),
        "degree_mean": sum(deg) / n,
        "degree_median": float(pd.Series(deg).median()),
        "degree_max": max(deg),
        "leaves": sum(1 for d in deg if d == 1),
        "leaves_share": sum(1 for d in deg if d == 1) / n,
        "foreign_leaves": sum(
            1 for v, d in G.degree() if d == 1 and not G.nodes[v]["french"]
        ),
    }


def components_table(G: nx.Graph) -> pd.DataFrame:
    """One row per connected component, largest first."""
    rows = []
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    for i, c in enumerate(comps):
        sub = G.subgraph(c)
        rows.append({
            "component": i,
            "nodes": len(c),
            "edges": sub.number_of_edges(),
            "passengers": sum(p for _, _, p in sub.edges(data="passengers")),
            "airports": ", ".join(sorted(f"{n} ({G.nodes[n]['name']})" for n in c))
            if len(c) <= 5 else f"{len(c)} airports incl. LFPG",
        })
    return pd.DataFrame(rows)


def distances(G: nx.Graph) -> dict:
    """Diameter, radius, center, periphery, mean distance on the LCC (hops)."""
    lcc = largest_component(G)
    ecc = nx.eccentricity(lcc)
    return {
        "lcc_nodes": lcc.number_of_nodes(),
        "lcc_share": lcc.number_of_nodes() / G.number_of_nodes(),
        "diameter": max(ecc.values()),
        "radius": min(ecc.values()),
        "center": sorted(nx.center(lcc, e=ecc)),
        "periphery_size": len(nx.periphery(lcc, e=ecc)),
        "avg_shortest_path": nx.average_shortest_path_length(lcc),
        "eccentricity": ecc,
    }


def fragility(G: nx.Graph) -> tuple[pd.DataFrame, dict]:
    """Articulation points and bridges.

    An articulation point is an airport whose removal disconnects the graph;
    `cut_off` counts how many airports would lose all their connection to the
    largest remaining component. A bridge is a route whose removal
    disconnects the graph. Most bridges are the single route of a leaf
    airport; the others ("core bridges") hold larger groups together.
    """
    rows = []
    n_comp = nx.number_connected_components(G)
    for a in nx.articulation_points(G):
        H = G.copy()
        H.remove_node(a)
        main = max(nx.connected_components(H), key=len)
        rows.append({
            "icao": a,
            "name": G.nodes[a]["name"],
            "zone": G.nodes[a]["zone"],
            "degree": G.degree(a),
            "components_after": nx.number_connected_components(H),
            "new_components": nx.number_connected_components(H) - n_comp + 1,
            "cut_off": len(largest_component(G)) - 1 - len(main),
        })
    aps = pd.DataFrame(rows).sort_values(["cut_off", "icao"], ascending=[False, True],
                                         ignore_index=True)

    bridges = list(nx.bridges(G))
    core = [(u, v) for u, v in bridges if G.degree(u) > 1 and G.degree(v) > 1]
    return aps, {
        "articulation_points": len(aps),
        "bridges": len(bridges),
        "leaf_bridges": len(bridges) - len(core),
        "core_bridges": [
            f"{u}-{v} ({G.nodes[u]['name']} - {G.nodes[v]['name']})" for u, v in core
        ],
    }


def run(year: int = MAIN_YEAR) -> dict:
    G = build_graph(year)
    dist = distances(G)
    aps, frag = fragility(G)
    summary = {
        "year": year,
        **size_and_type(G),
        "connected": nx.is_connected(G),
        "components": nx.number_connected_components(G),
        **{k: v for k, v in dist.items() if k != "eccentricity"},
        **{k: v for k, v in frag.items()},
    }
    return {
        "summary": summary,
        "components": components_table(G),
        "articulation_points": aps,
        "eccentricity": pd.Series(dist["eccentricity"], name="eccentricity")
        .value_counts().sort_index().rename_axis("hops"),
    }


if __name__ == "__main__":
    year = int(sys.argv[1]) if len(sys.argv) > 1 else MAIN_YEAR
    out = run(year)
    RESULTS_DIR.mkdir(exist_ok=True)
    pd.set_option("display.width", 200, "display.max_colwidth", 90)

    print(f"=== {year} ===")
    for k, v in out["summary"].items():
        print(f"{k:>22}: {round(v, 3) if isinstance(v, float) else v}")
    print("\nComponents:\n", out["components"].to_string(index=False))
    print("\nArticulation points:\n", out["articulation_points"].to_string(index=False))
    print("\nEccentricity distribution (LCC):\n", out["eccentricity"].to_string())

    pd.Series(out["summary"]).to_csv(RESULTS_DIR / f"core_summary_{year}.csv", header=["value"])
    out["components"].to_csv(RESULTS_DIR / f"core_components_{year}.csv", index=False)
    out["articulation_points"].to_csv(RESULTS_DIR / f"core_articulation_points_{year}.csv", index=False)
    print(f"\nSaved results/core_*_{year}.csv")

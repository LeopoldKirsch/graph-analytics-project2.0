"""Three views of the same graph model.

1. Full network (the model described in the report): all routes.
2. Domestic network: subgraph induced by French airports (metro + overseas).
   Who connects France internally? Caveat: routes inside French Polynesia
   and New Caledonia are not in the Eurostat data.
3. International layer: French-foreign routes only (bipartite). There are no
   foreign-foreign routes in the data, so foreign airports cannot be central;
   the international layer is measured as reach (destinations, world
   regions, passengers by distance band), not with centrality.

Usage:
    python src/views.py          # 2024, writes results/views_*.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import pandas as pd

from build_graph import MAIN_YEAR, build_graph
from centrality import centrality_table, freeman_centralization
from core_metrics import fragility, largest_component

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
HUBS = {"LFPG": "CDG", "LFPO": "Orly"}
DISTANCE_BANDS = [(0, 1500, "short (<1,500 km)"),
                  (1500, 4000, "medium (1,500-4,000 km)"),
                  (4000, 1e9, "long (>4,000 km)")]


def domestic_graph(G: nx.Graph) -> nx.Graph:
    # built in sorted order: subgraph() iterates over a set of codes, whose order
    # (and so the last float digits of betweenness, and ties) changes between runs
    F = nx.Graph(**G.graph, view="domestic")
    F.add_nodes_from(sorted((n, d) for n, d in G.nodes(data=True) if d["french"]))
    F.add_edges_from(sorted((min(u, v), max(u, v), d) for u, v, d in G.edges(data=True)
                            if G.nodes[u]["french"] and G.nodes[v]["french"]))
    return F


def international_graph(G: nx.Graph) -> nx.Graph:
    I = nx.Graph(**G.graph, view="international")
    I.add_edges_from((u, v, d) for u, v, d in G.edges(data=True)
                     if G.nodes[u]["french"] != G.nodes[v]["french"])
    for n in I:
        I.nodes[n].update(G.nodes[n])
    return I


# ------------------------------------------------------------ full vs domestic

def view_summary(H: nx.Graph, label: str) -> dict:
    """The required metrics for one view, as one column of the comparison table."""
    lcc = largest_component(H)
    ecc = nx.eccentricity(lcc)
    cent = centrality_table(H)
    aps, frag = fragility(H)
    top = cent.head(3)
    row = {
        "view": label,
        "nodes": H.number_of_nodes(),
        "edges": H.number_of_edges(),
        "density": H.number_of_edges() / (H.number_of_nodes() * (H.number_of_nodes() - 1) / 2),
        "components": nx.number_connected_components(H),
        "isolated_airports": nx.number_of_isolates(H),
        "lcc_nodes": lcc.number_of_nodes(),
        "diameter_lcc": max(ecc.values()),
        "radius_lcc": min(ecc.values()),
        "center_lcc": ", ".join(sorted(nx.center(lcc, e=ecc))),
        "avg_shortest_path_lcc": nx.average_shortest_path_length(lcc),
        **freeman_centralization(H),
        "top3_betweenness": ", ".join(f"{i} {b:.2f}" for i, b in top["betweenness"].items()),
        "articulation_points": frag["articulation_points"],
        "worst_articulation_point": (f"{aps.iloc[0]['icao']} (cuts off {aps.iloc[0]['cut_off']})"
                                     if len(aps) else "none"),
    }
    for code, short in HUBS.items():
        row[f"{short}_degree"] = H.degree(code)
        row[f"{short}_betweenness"] = cent.at[code, "betweenness"]
        row[f"{short}_harmonic"] = cent.at[code, "harmonic"]
        row[f"{short}_betweenness_rank"] = cent.at[code, "rank_betweenness"]
    return row


def compare_views(G: nx.Graph) -> pd.DataFrame:
    """Full vs domestic, side by side (rows = metrics)."""
    rows = [view_summary(G, "full network"), view_summary(domestic_graph(G), "domestic")]
    return pd.DataFrame(rows).set_index("view").T


def domestic_resilience(F: nx.Graph) -> pd.DataFrame:
    """Remove CDG, Orly, both: what is left of the domestic network?"""
    rows = []
    for removed in ([], ["LFPG"], ["LFPO"], ["LFPG", "LFPO"]):
        H = F.copy()
        H.remove_nodes_from(removed)
        lcc = largest_component(H)
        rows.append({
            "removed": " + ".join(HUBS[r] for r in removed) or "none",
            "components": nx.number_connected_components(H),
            "isolated": nx.number_of_isolates(H),
            "lcc_nodes": lcc.number_of_nodes(),
            "lcc_share_of_airports": lcc.number_of_nodes() / F.number_of_nodes(),
            "passengers_kept_in_lcc": sum(p for _, _, p in lcc.edges(data="passengers"))
            / sum(p for _, _, p in F.edges(data="passengers")),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------ international

def _intl_routes(G: nx.Graph) -> pd.DataFrame:
    rows = []
    for u, v, d in international_graph(G).edges(data=True):
        fr, fo = (u, v) if G.nodes[u]["french"] else (v, u)
        band = next(lbl for lo, hi, lbl in DISTANCE_BANDS if lo <= d["distance_km"] < hi)
        rows.append({"french": fr, "foreign": fo, "region": G.nodes[fo]["icao_region"],
                     "band": band, "passengers": d["passengers"], "km": d["distance_km"]})
    return pd.DataFrame(rows)


def _share_by(df: pd.DataFrame, key: str) -> pd.DataFrame:
    cdg = df[df["french"] == "LFPG"]
    only_cdg = df.groupby("foreign").filter(lambda g: set(g["french"]) == {"LFPG"})
    out = df.groupby(key).agg(routes=("foreign", "size"), destinations=("foreign", "nunique"),
                              passengers=("passengers", "sum"))
    out["cdg_routes"] = cdg.groupby(key).size()
    out["cdg_passenger_share"] = cdg.groupby(key)["passengers"].sum() / out["passengers"]
    out["destinations_only_via_cdg"] = only_cdg.groupby(key)["foreign"].nunique()
    return out.fillna(0).sort_values("passengers", ascending=False)


def international_reach(G: nx.Graph) -> dict:
    df = _intl_routes(G)
    per_airport = (df.groupby("french")
                   .agg(destinations=("foreign", "nunique"), regions=("region", "nunique"),
                        passengers=("passengers", "sum"),
                        long_haul_destinations=("band", lambda b: (b == DISTANCE_BANDS[2][2]).sum()))
                   .sort_values("passengers", ascending=False))
    per_airport["passenger_share"] = per_airport["passengers"] / df["passengers"].sum()
    per_airport.insert(0, "name", [G.nodes[n]["name"] for n in per_airport.index])
    return {
        "by_region": _share_by(df, "region"),
        "by_distance": _share_by(df, "band").reindex([b[2] for b in DISTANCE_BANDS]),
        "by_french_airport": per_airport,
        "summary": {
            "international_routes": len(df),
            "foreign_destinations": df["foreign"].nunique(),
            "cdg_passenger_share": df.loc[df["french"] == "LFPG", "passengers"].sum()
            / df["passengers"].sum(),
            "destinations_only_via_cdg": df.groupby("foreign")
            .filter(lambda g: set(g["french"]) == {"LFPG"})["foreign"].nunique(),
        },
    }


def run(year: int = MAIN_YEAR) -> dict:
    G = build_graph(year)
    F = domestic_graph(G)
    dom = centrality_table(F)
    return {
        "comparison": compare_views(G),
        "domestic_centrality": dom,
        "domestic_resilience": domestic_resilience(F),
        **{f"intl_{k}": v for k, v in international_reach(G).items()},
    }


if __name__ == "__main__":
    year = int(sys.argv[1]) if len(sys.argv) > 1 else MAIN_YEAR
    out = run(year)
    RESULTS_DIR.mkdir(exist_ok=True)
    pd.set_option("display.width", 220, "display.max_colwidth", 60,
                  "display.float_format", "{:.3f}".format)
    print(f"=== {year}: full vs domestic ===\n{out['comparison'].to_string()}")
    print("\nDomestic centrality (top 10):\n", out["domestic_centrality"].head(10)[
        ["name", "degree", "strength", "betweenness", "harmonic", "pagerank"]].to_string())
    print("\nDomestic resilience:\n", out["domestic_resilience"].to_string(index=False))
    print("\nInternational summary:", {k: round(v, 3) if isinstance(v, float) else v
                                      for k, v in out["intl_summary"].items()})
    print("\nInternational by distance:\n", out["intl_by_distance"].to_string())
    print("\nInternational by region:\n", out["intl_by_region"].to_string())
    print("\nInternational reach per French airport:\n", out["intl_by_french_airport"].to_string())

    out["comparison"].to_csv(RESULTS_DIR / f"views_comparison_{year}.csv")
    out["domestic_centrality"].to_csv(RESULTS_DIR / f"views_domestic_centrality_{year}.csv")
    out["domestic_resilience"].to_csv(RESULTS_DIR / f"views_domestic_resilience_{year}.csv", index=False)
    for k in ("by_region", "by_distance", "by_french_airport"):
        out[f"intl_{k}"].to_csv(RESULTS_DIR / f"views_international_{k}_{year}.csv")
    pd.Series(out["intl_summary"]).to_csv(RESULTS_DIR / f"views_international_summary_{year}.csv",
                                          header=["value"])
    print(f"\nSaved results/views_*_{year}.csv")

"""Centrality: which airports hold the network together, and how far behind
CDG are Orly, Nice and Lyon?

Measures (NetworkX):
- degree: number of routes (+ normalised degree centrality)
- strength: weighted degree with weight="passengers" (annual passengers on
  the airport's routes)
- coverage: strength / DGAC APT passengers of the airport. Share of the
  airport's real traffic captured by the Eurostat routes (close to 1 for the
  airports that report to Eurostat).
- betweenness (main, unweighted, normalised): share of shortest paths
  passing through the airport. "Holds the network together".
- betweenness_weighted (check): weight="cost" (= 1/passengers), so busy
  routes count as short. NetworkX reads weights as distances.
- harmonic closeness (normalised by n-1): mean of 1/distance to every other
  airport, with 1/inf = 0. Chosen over standard closeness because it is
  defined on a disconnected graph.
- pagerank (weight="passengers"): importance from being linked to important
  airports.

Network-level answers:
- Freeman centralization for degree and betweenness: 0 = all airports equal,
  1 = perfect star. One number for "how centralised is the network?".
- Spearman rank correlations between measures.
- Gateways: French airports that rank much higher on betweenness than on degree.
- Robustness: CDG betweenness and centralization recomputed (a) without
  foreign airports that have a single route, (b) on the French-only network.

Usage:
    python src/centrality.py          # 2024, writes results/centrality_*.csv
    python src/centrality.py 2019
"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import pandas as pd

from build_graph import MAIN_YEAR, build_graph

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
FOCUS = ["LFPG", "LFPO", "LFMN", "LFLL"]  # CDG, Orly, Nice, Lyon
MEASURES = ["degree", "strength", "betweenness", "betweenness_weighted",
            "harmonic", "pagerank"]


def centrality_table(G: nx.Graph) -> pd.DataFrame:
    n = G.number_of_nodes()
    df = pd.DataFrame({
        "name": dict(G.nodes(data="name")),
        "zone": dict(G.nodes(data="zone")),
        "degree": dict(G.degree()),
        "degree_centrality": nx.degree_centrality(G),
        "strength": dict(G.degree(weight="passengers")),
        "apt_passengers": dict(G.nodes(data="apt_passengers")),
        "betweenness": nx.betweenness_centrality(G),
        "betweenness_weighted": nx.betweenness_centrality(G, weight="cost"),
        "harmonic": {k: v / (n - 1) for k, v in nx.harmonic_centrality(G).items()},
        "pagerank": nx.pagerank(G, weight="passengers"),
    })
    df["coverage"] = df["strength"] / df["apt_passengers"]
    # Equal scores can differ in the last float digits depending on the library
    # version (e.g. harmonic closeness of airports at the same distances), which
    # would break ties differently from one machine to the next. Rounding to 12
    # decimals makes ranks, Spearman correlations and the row order reproducible.
    df[MEASURES] = df[MEASURES].round(12)
    for m in MEASURES:
        df[f"rank_{m}"] = df[m].rank(ascending=False, method="min").astype(int)
    df.index.name = "icao"
    return df.sort_index().sort_values("betweenness", ascending=False, kind="stable")


def focus_table(df: pd.DataFrame, extra_top: int = 10) -> pd.DataFrame:
    """CDG, Orly, Nice, Lyon and the top airports, with each measure as a ratio to CDG."""
    keep = list(dict.fromkeys(FOCUS + list(df.index[:extra_top])))
    out = df.loc[keep, ["name", "degree", "strength", "betweenness",
                        "betweenness_weighted", "harmonic", "pagerank", "coverage"]].copy()
    for m in ("degree", "strength", "betweenness", "harmonic", "pagerank"):
        out[f"{m}_vs_cdg"] = out[m] / df.loc["LFPG", m]
    return out


def freeman_centralization(G: nx.Graph) -> dict:
    """Freeman centralization, normalised so a star graph scores 1."""
    n = G.number_of_nodes()
    deg = [d for _, d in G.degree()]
    btw = list(nx.betweenness_centrality(G).values())  # already in [0, 1]
    return {
        "degree_centralization": sum(max(deg) - d for d in deg) / ((n - 1) * (n - 2)),
        "betweenness_centralization": sum(max(btw) - b for b in btw) / (n - 1),
    }


def gateways(df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """French airports whose betweenness rank is far above their degree rank:
    few routes, but they are the passage to a part of the network.
    Ranks are taken over the whole graph."""
    g = df[(df["betweenness"] > 0) & (df.index != "LFPG") & (df["zone"] != "foreign")].copy()
    g["rank_gap"] = g["rank_degree"] - g["rank_betweenness"]
    return g.sort_values(["rank_gap", "betweenness"], ascending=False).head(k)[
        ["name", "zone", "degree", "rank_degree", "betweenness", "rank_betweenness", "rank_gap"]
    ]


def robustness(G: nx.Graph) -> pd.DataFrame:
    """Is CDG's dominance an artefact of single-route foreign airports?"""
    no_leaves = G.copy()
    no_leaves.remove_nodes_from(
        [v for v, d in G.degree() if d == 1 and not G.nodes[v]["french"]]
    )
    # French-only graph built in sorted order (reproducible float digits)
    french = nx.Graph()
    french.add_nodes_from(sorted((n, d) for n, d in G.nodes(data=True) if d["french"]))
    french.add_edges_from(sorted((min(u, v), max(u, v), d) for u, v, d in G.edges(data=True)
                                 if G.nodes[u]["french"] and G.nodes[v]["french"]))
    rows = []
    for label, H in (("full graph", G),
                     ("without single-route foreign airports", no_leaves),
                     ("French airports only", french)):
        b = nx.betweenness_centrality(H)
        top = [a for a in sorted(b, key=b.get, reverse=True) if a != "LFPG"][:3]
        rows.append({
            "variant": label,
            "nodes": H.number_of_nodes(),
            "edges": H.number_of_edges(),
            "cdg_betweenness": b.get("LFPG", float("nan")),
            "top_others": ", ".join(f"{a} {b[a]:.2f}" for a in top),
            **freeman_centralization(H),
        })
    return pd.DataFrame(rows)


def run(year: int = MAIN_YEAR) -> dict:
    G = build_graph(year)
    df = centrality_table(G)
    return {
        "table": df,
        "focus": focus_table(df),
        "centralization": freeman_centralization(G),
        "spearman": df[MEASURES].corr(method="spearman"),
        "gateways": gateways(df),
        "robustness": robustness(G),
    }


if __name__ == "__main__":
    year = int(sys.argv[1]) if len(sys.argv) > 1 else MAIN_YEAR
    out = run(year)
    RESULTS_DIR.mkdir(exist_ok=True)
    pd.set_option("display.width", 250, "display.max_colwidth", 30,
                  "display.float_format", "{:.3f}".format)

    print(f"=== {year} ===\nCentralization:", {k: round(v, 3) for k, v in out["centralization"].items()})
    print("\nFocus airports (ratios to CDG):\n", out["focus"].to_string())
    print("\nSpearman rank correlation:\n", out["spearman"].to_string())
    print("\nGateways (betweenness rank >> degree rank):\n", out["gateways"].to_string())
    print("\nRobustness:\n", out["robustness"].to_string(index=False))

    out["table"].to_csv(RESULTS_DIR / f"centrality_{year}.csv")
    out["focus"].to_csv(RESULTS_DIR / f"centrality_focus_{year}.csv")
    out["spearman"].to_csv(RESULTS_DIR / f"centrality_spearman_{year}.csv")
    out["gateways"].to_csv(RESULTS_DIR / f"centrality_gateways_{year}.csv")
    out["robustness"].to_csv(RESULTS_DIR / f"centrality_robustness_{year}.csv", index=False)
    pd.Series(out["centralization"]).to_csv(RESULTS_DIR / f"centralization_{year}.csv", header=["value"])
    print(f"\nSaved results/centrality_*_{year}.csv")

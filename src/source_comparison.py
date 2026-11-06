"""Why Eurostat: the same analyses on Eurostat routes and on DGAC published routes.

The DGAC airport-pair series (ASP_PAP) only publishes a route-month served by
at least 3 airlines, none above 85 % of the traffic. Eurostat (avia_par_fr)
publishes every route of the reporting airports. Both are turned into the
same graph model (build_graph.build_graph / build_dgac_graph) and the
project's own functions are applied to both, for 2019, 2020 and 2024.

Three questions:
1. Coverage: which share of the French airports' real traffic (DGAC airport
   series, independent of both route sources) do the routes capture?
2. Quality: how many routes does each source have, how do passengers compare
   on common routes, and are the two Eurostat reports of a domestic route
   consistent?
3. Conclusions: do the headline results change with the source?

Usage:
    python src/source_comparison.py      # writes results/source_comparison_*.csv
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd

import centrality
import communities
import comparison
import null_model
import resilience
import views
from build_graph import build_dgac_graph, build_graph, load_routes

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
YEARS = (2019, 2020, 2024)
SOURCES = {"Eurostat": build_graph, "DGAC published": build_dgac_graph}
FOCUS = {"LFPG": "CDG", "LFPO": "Orly", "LFMN": "Nice", "LFLL": "Lyon", "LFBO": "Toulouse",
         "LFBD": "Bordeaux", "LFML": "Marseille", "LFRS": "Nantes", "LFOB": "Beauvais"}
N_NULL = 200


def coverage(G: nx.Graph) -> dict:
    """Passengers on the graph's routes / passengers counted at the airport (DGAC APT)."""
    s = dict(G.degree(weight="passengers"))
    french = [n for n, f in G.nodes(data="french") if f and G.nodes[n].get("apt_passengers")]
    out = {"all French airports": sum(s[n] for n in french)
           / sum(G.nodes[n]["apt_passengers"] for n in french)}
    for code, name in FOCUS.items():
        out[name] = s[code] / G.nodes[code]["apt_passengers"] if code in G else 0.0
    return out


def indicators(G: nx.Graph, null: bool = False) -> dict:
    """Headline results of the project on one graph."""
    cent = centrality.centrality_table(G)
    cz = centrality.freeman_centralization(G)
    conc = comparison.concentration(G)
    F = views.domestic_graph(G)
    F.remove_nodes_from(list(nx.isolates(F)))
    fbc = nx.betweenness_centrality(F)
    fmax = max(fbc.values())
    dres = views.domestic_resilience(views.domestic_graph(G)).set_index("removed")
    scen = resilience.scenarios(G, n_random=50).set_index("scenario")
    adaptive = resilience.removal_curve(G, resilience.adaptive_order(G, 10), 10, diameter=False)
    comms = communities.louvain(G, weighted=True)
    row = {
        "airports": G.number_of_nodes(),
        "routes": G.number_of_edges(),
        "domestic_routes": F.number_of_edges(),
        "passengers_m": sum(p for *_, p in G.edges(data="passengers")) / 1e6,
        "routes_under_1000_passengers": sum(1 for *_, p in G.edges(data="passengers") if p < 1000),
        "cdg_routes": G.degree("LFPG"),
        "cdg_passenger_share": conc["cdg_passenger_share"],
        "effective_airports": conc["effective_airports"],
        "cdg_betweenness": cent.at["LFPG", "betweenness"],
        "second_betweenness": f"{cent.index[1]} {cent['betweenness'].iloc[1]:.2f}",
        "betweenness_centralization": cz["betweenness_centralization"],
        "domestic_betweenness_centralization": sum(fmax - b for b in fbc.values()) / (len(F) - 1),
        "domestic_top_betweenness": max(sorted(fbc), key=fbc.get),
        "domestic_kept_without_cdg": dres.at["CDG", "passengers_kept_in_lcc"]
        / dres.at["none", "passengers_kept_in_lcc"],
        "domestic_kept_without_orly": dres.at["Orly", "passengers_kept_in_lcc"]
        / dres.at["none", "passengers_kept_in_lcc"],
        "lcc_share_without_cdg": scen.at["CDG", "lcc_share"],
        "passengers_kept_without_cdg": scen.at["CDG", "passengers_kept"],
        "passengers_kept_without_cdg_orly": scen.at["CDG + Orly", "passengers_kept"],
        "lcc_after_10_adaptive": adaptive.set_index("removed").at[10, "lcc_share"],
        "communities": len(comms),
        "modularity": communities.modularity(G, comms, weighted=True),
    }
    if null:
        t, _ = null_model.compare(G, N_NULL, "full")
        t = t.set_index("metric")
        row["null_cdg_betweenness_mean"] = t.at["cdg_betweenness", "null_mean"]
        row["null_cdg_betweenness_p"] = t.at["cdg_betweenness", "p_two_sided"]
    return row


def route_overlap(E: nx.Graph, D: nx.Graph) -> dict:
    e, d = {frozenset(x) for x in E.edges()}, {frozenset(x) for x in D.edges()}
    common = [tuple(x) for x in e & d]
    ratio = pd.Series([D.edges[x]["passengers"] / E.edges[x]["passengers"] for x in common])
    hidden = sorted(e - d, key=lambda x: -E.edges[tuple(x)]["passengers"])
    return {
        "routes_in_both": len(common),
        "routes_only_eurostat": len(e - d),
        "routes_only_dgac": len(d - e),
        "passengers_only_eurostat_m": sum(E.edges[tuple(x)]["passengers"] for x in e - d) / 1e6,
        "dgac_over_eurostat_median": ratio.median(),
        "dgac_over_eurostat_p10": ratio.quantile(0.1),
        "biggest_routes_only_eurostat": ", ".join(
            f"{'-'.join(sorted(x))} {E.edges[tuple(x)]['passengers'] / 1e3:.0f}k" for x in hidden[:6]),
    }


def double_reports(year: int) -> dict:
    """Routes reported by both airports: relative gap between the two reports."""
    r = load_routes(year)
    r = r.assign(u=r[["reporting", "partner"]].min(axis=1), v=r[["reporting", "partner"]].max(axis=1))
    g = r.groupby(["u", "v"])["passengers"].agg(["min", "max", "count"])
    g = g[g["count"] == 2]
    gap = (g["max"] - g["min"]) / g["max"]
    return {"routes_reported_twice": len(g), "report_gap_median": gap.median(),
            "report_gap_p90": gap.quantile(0.9)}


def run(years=YEARS) -> dict[str, pd.DataFrame]:
    rows, cov, routes = [], [], []
    for y in years:
        graphs = {name: build(y) for name, build in SOURCES.items()}
        for name, G in graphs.items():
            rows.append({"year": y, "source": name, **indicators(G, null=(y == max(years)))})
            cov.append({"year": y, "source": name, **coverage(G)})
        routes.append({"year": y, **route_overlap(graphs["Eurostat"], graphs["DGAC published"]),
                       **double_reports(y)})
    return {"indicators": pd.DataFrame(rows), "coverage": pd.DataFrame(cov),
            "routes": pd.DataFrame(routes)}


if __name__ == "__main__":
    out = run()
    RESULTS_DIR.mkdir(exist_ok=True)
    for k, df in out.items():
        df.to_csv(RESULTS_DIR / f"source_comparison_{k}.csv", index=False)
    pd.set_option("display.width", 220, "display.max_colwidth", 90)
    print(out["coverage"].round(2).to_string(index=False))
    print()
    print(out["indicators"].set_index(["year", "source"]).T.to_string())
    print()
    print(out["routes"].set_index("year").T.to_string())
    print("\nSaved results/source_comparison_*.csv")

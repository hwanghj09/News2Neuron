from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import linear_sum_assignment

from .provenance import write_json, manifest


class Circuit:
    def __init__(self, neurons, edges, metadata):
        self.neurons = neurons.copy().reset_index(drop=True)
        self.edges = edges.copy().reset_index(drop=True)
        self.metadata = metadata
        self.neurons["root_id"] = self.neurons.root_id.astype(str)
        for c in ["pre", "post"]:
            self.edges[c] = self.edges[c].astype(str)
        self.ids = self.neurons.root_id.tolist()
        self.index = {v: i for i, v in enumerate(self.ids)}
        if self.neurons.root_id.duplicated().any():
            raise ValueError("Duplicate root IDs")
        if not set(self.edges.pre).union(self.edges.post).issubset(self.index):
            raise ValueError("Edge endpoints are missing from neurons")
        if not set(self.neurons.version).issubset({metadata["connectome_version"]}):
            raise ValueError("Mixed connectome versions are forbidden")
        if not np.isfinite(self.edges[["synapses", "signed_synapses"]].to_numpy()).all() or (self.edges.synapses <= 0).any():
            raise ValueError("Invalid synapse weights")

    def cells(self, role=None, cell_type=None):
        mask = np.ones(len(self.ids), dtype=bool)
        if role is not None:
            mask &= self.neurons.role.eq(role)
        if cell_type is not None:
            mask &= self.neurons.cell_type.eq(cell_type)
        return self.neurons.loc[mask, "root_id"].tolist()

    def graph(self):
        g = nx.DiGraph()
        g.add_nodes_from(self.ids)
        g.add_weighted_edges_from(self.edges[["pre", "post", "synapses"]].itertuples(index=False, name=None))
        return g

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.neurons.to_csv(path / "neurons.csv", index=False)
        self.edges.to_parquet(path / "edges.parquet", index=False)
        write_json(path / "metadata.json", self.metadata)
        rows = []
        g = self.graph()
        for source in self.cells(role="input"):
            distances = nx.single_source_shortest_path_length(g, source, cutoff=3)
            for target in self.cells(role="output"):
                if target in distances:
                    p = nx.shortest_path(g, source, target)
                    rows.append({"input_id": source, "output_id": target, "hops": distances[target],
                                 "path": " -> ".join(p), "synapses_per_edge": ",".join(str(g[a][b]["weight"]) for a,b in zip(p,p[1:]))})
        pd.DataFrame(rows, columns=["input_id", "output_id", "hops", "path", "synapses_per_edge"]).to_csv(path / "paths.csv", index=False)

    @classmethod
    def load(cls, path):
        path = Path(path)
        return cls(pd.read_csv(path / "neurons.csv", dtype={"root_id":str}),
                   pd.read_parquet(path / "edges.parquet"), json.loads((path / "metadata.json").read_text()))


def _distances(edges, starts, hops, reverse=False):
    a, b = ("post", "pre") if reverse else ("pre", "post")
    d = {v: 0 for v in starts}
    frontier = set(starts)
    for depth in range(1, hops + 1):
        frontier = set(edges.loc[edges[a].isin(frontier), b]) - d.keys()
        d.update({v: depth for v in frontier})
    return d


def extract_v783(config):
    """Induced circuit on directed input-output paths of at most `hops` edges."""
    if config["version"] != "FAFB_v783":
        raise ValueError("This importer is specifically for FAFB_v783")
    comp = pd.read_csv(config["completeness"], index_col=0).index.astype("int64")
    annotations = pd.read_csv(config["annotations"], sep="\t", dtype=str).fillna("")
    if "root_id" not in annotations:
        raise ValueError("Expected documented v783 annotations with root_id")
    annotations = annotations[annotations.root_id.isin(comp.astype(str))].copy()
    # GF is explicitly backed by the annotation 'Giant Fiber', not a guessed root ID.
    gf = annotations.hemibrain_type.eq("Giant Fiber")
    annotations.loc[gf, "cell_type"] = "GF"
    sources = set(annotations.loc[annotations.cell_type.isin(config["input_types"]), "root_id"].astype("int64"))
    targets = set(annotations.loc[annotations.cell_type.isin(config["output_types"]), "root_id"].astype("int64"))
    missing_types = [t for t in config["input_types"] + config["output_types"] if not annotations.cell_type.eq(t).any()]
    if missing_types or not sources or not targets:
        raise ValueError(f"Unconfirmed target cell types: {missing_types}")
    columns = ["Presynaptic_ID", "Postsynaptic_ID", "Connectivity", "Excitatory x Connectivity"]
    chunks = []
    for batch in pq.ParquetFile(config["connectivity"]).iter_batches(batch_size=250_000, columns=columns):
        frame = batch.to_pandas(ignore_metadata=True)
        chunks.append(frame.loc[frame.Connectivity >= config["min_synapses"]])
    edges = pd.concat(chunks, ignore_index=True)
    edges.columns = ["pre", "post", "synapses", "signed_synapses"]
    if not set(edges.pre).union(edges.post).issubset(set(comp)):
        raise ValueError("Connectivity IDs do not match the v783 completeness snapshot")
    forward = _distances(edges, sources, config["hops"])
    backward = _distances(edges, targets, config["hops"], reverse=True)
    selected = {v for v in forward.keys() & backward.keys() if forward[v] + backward[v] <= config["hops"]} | sources | targets
    if len(selected) > config["max_neurons"]:
        raise ValueError(f"{len(selected)} neurons exceed configured cap; explicitly revise extraction settings")
    indegree = edges.groupby("post").size()
    outdegree = edges.groupby("pre").size()
    neurons = pd.DataFrame({"root_id": sorted(selected)})
    neurons["in_degree_full"] = neurons.root_id.map(indegree).fillna(0).astype(int)
    neurons["out_degree_full"] = neurons.root_id.map(outdegree).fillna(0).astype(int)
    neurons.root_id = neurons.root_id.astype(str)
    neurons = neurons.merge(annotations[["root_id", "cell_type", "hemibrain_type", "top_nt", "side"]], how="left", on="root_id", validate="one_to_one").fillna("")
    neurons["role"] = np.where(neurons.root_id.isin({str(v) for v in sources}), "input",
                               np.where(neurons.root_id.isin({str(v) for v in targets}), "output", "intermediate"))
    neurons["version"] = config["version"]
    selected_edges = edges[edges.pre.isin(selected) & edges.post.isin(selected)].copy()
    metadata = manifest([config[k] for k in ["completeness", "connectivity", "annotations"]],
        connectome_version=config["version"], data_source="measured_connectivity_model_activity",
        extraction="induced circuit on input-output paths; isolated specified targets retained",
        hops=config["hops"], min_synapses=config["min_synapses"], upstream_commit=config["upstream_commit"],
        annotations_commit=config.get("annotations_commit"),
        gf_alias="cell_type DNp01 / hemibrain_type Giant Fiber -> GF",
        missing_types=missing_types, n_neurons=len(neurons), n_edges=len(selected_edges),
        limitations=["Subcircuit omits external inputs and feedback", "Transmitter signs inherited from Shiu model, receptor effects unknown"])
    return Circuit(neurons, selected_edges, metadata)


def demo_circuit():
    """Fictional topology for offline software verification; never called FlyWire."""
    types = ["LPLC2"]*4 + ["LC4"]*4 + ["sensory"]*12 + ["relay"]*16 + ["GF", "DNp06"] + ["inhibitor"]*2
    ids = [f"demo_{i:03}" for i in range(len(types))]
    neurons = pd.DataFrame({"root_id": ids, "cell_type": types, "role": ["input"]*8 + ["intermediate"]*28 + ["output"]*2 + ["intermediate"]*2, "version":"SYNTHETIC_DEMO"})
    edges = []
    for i in range(8):
        for j in range(20, 28):
            edges.append((ids[i], ids[j], 20, 20))
    for i in range(20, 36):
        for j in [36, 37]:
            edges.append((ids[i], ids[j], 18, 18))
    for i in range(8,20):
        edges.append((ids[i], ids[28+i%8], 10, 10))
    edges.extend([(ids[38], ids[36], 8, -8), (ids[39], ids[37], 8, -8)])
    return Circuit(neurons, pd.DataFrame(edges, columns=["pre", "post", "synapses", "signed_synapses"]),
                   {"connectome_version":"SYNTHETIC_DEMO", "data_source":"synthetic_fixture", "n_neurons":len(ids), "n_edges":len(edges)})


def degree_matched_control(circuit, targets, excluded, seed):
    """Equal size, no replacement, closest log in/out degrees with random tie breaks."""
    g = circuit.graph()
    candidates = sorted(set(circuit.ids) - set(excluded) - set(targets))
    if len(candidates) < len(targets):
        raise ValueError("Insufficient neurons for equal-size disjoint degree-matched control")
    def degree(ids):
        return np.log1p(np.array([[g.in_degree(n), g.out_degree(n)] for n in ids], dtype=float))
    t, c = degree(targets), degree(candidates)
    cost = np.linalg.norm(t[:, None, :] - c[None, :, :], axis=2)
    cost += np.random.default_rng(seed).uniform(0, 1e-6, cost.shape)
    row, col = linear_sum_assignment(cost)
    selected = [candidates[i] for i in col]
    return selected, float(cost[row, col].mean())

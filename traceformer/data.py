import os
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData


def load_adata_files(data_dir: str) -> List[AnnData]:
    """Load all `.h5ad` files from a directory into a list."""

    files = [
        os.path.join(data_dir, fname)
        for fname in os.listdir(data_dir)
        if fname.endswith(".h5ad")
    ]
    if not files:
        raise FileNotFoundError(f"No .h5ad files found in {data_dir}")
    return [sc.read_h5ad(path) for path in sorted(files)]


def align_adata_to_genes(adata: AnnData, gene_list: Sequence[str]) -> AnnData:
    """Ensure an AnnData object has the provided genes in the requested order."""

    if "log1p" not in adata.layers:
        raise ValueError("`adata.layers['log1p']` missing. Run preprocessing first.")

    genes = list(gene_list)
    if not genes:
        raise ValueError("`gene_list` cannot be empty when aligning AnnData objects.")

    matrix = adata.layers["log1p"]
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()

    df = pd.DataFrame(matrix, index=adata.obs_names, columns=adata.var_names)
    for gene in genes:
        if gene not in df.columns:
            df[gene] = 0.0

    df = df[genes]
    aligned = sc.AnnData(df.to_numpy(), obs=adata.obs.copy(), var=pd.DataFrame(index=genes))
    aligned.layers["log1p"] = aligned.X.copy()
    return aligned


def preprocess_and_integrate(
    adatas: Iterable[AnnData],
    gene_list: Sequence[str],
    pca_components: int = 50,
    compute_topology: bool = False,
) -> AnnData:
    """Concatenate AnnData objects and run minimal preprocessing.

    - Normalise to 1e4 counts per cell and log1p-transform.
    - Align features to the provided ``gene_list`` with zero-filled missing genes.
    - Optionally compute PCA/neighbours/Leiden/PAGA for exploratory analysis.
    """

    adata = sc.concat(list(adatas), join="outer", label="dataset", index_unique=None)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers["log1p"] = adata.X.copy()

    adata = align_adata_to_genes(adata, gene_list)
    if adata.n_vars > 1:
        sc.tl.pca(adata, n_comps=min(pca_components, adata.n_vars - 1))
    else:
        adata.obsm["X_pca"] = np.zeros((adata.n_obs, 1))

    if compute_topology:
        sc.pp.neighbors(adata, use_rep="X_pca")
        sc.tl.leiden(adata, key_added="leiden")
        sc.tl.paga(adata, groups="leiden")
    return adata


def score_stemness(adata: AnnData, stemness_genes: Sequence[str]) -> None:
    """Compute stemness scores and store them in ``adata.obs``."""

    sc.tl.score_genes(adata, gene_list=list(stemness_genes), score_name="stemness_score")
    scores = adata.obs["stemness_score"].to_numpy()
    if scores.ptp() == 0:
        adata.obs["stemness_norm"] = np.zeros_like(scores)
    else:
        adata.obs["stemness_norm"] = (scores - scores.min()) / (scores.max() - scores.min())


def paga_edge_list(adata: AnnData, min_connectivity: float = 0.05) -> List[Tuple[str, str]]:
    """Return a list of Leiden-cluster edges from PAGA above a connectivity threshold."""

    paga_connectivities = adata.uns.get("paga", {}).get("connectivities")
    if paga_connectivities is None:
        raise ValueError("PAGA connectivities not found; set `compute_topology=True` in preprocessing.")

    edges = []
    clusters = adata.obs["leiden"].cat.categories
    matrix = paga_connectivities.A if hasattr(paga_connectivities, "A") else paga_connectivities
    for i, src in enumerate(clusters):
        for j, dst in enumerate(clusters):
            if j <= i:
                continue
            weight = matrix[i, j]
            if weight >= min_connectivity:
                edges.append((src, dst))
    return edges


def generate_trajectory_sequences(
    adata: AnnData,
    n_walks_per_cell: int = 2,
    max_len: int | None = 10,
    min_connectivity: float = 0.05,
) -> List[List[int]]:
    """Generate directed random-walk trajectories constrained by stemness gradients.

    Walks now continue until no neighbour has lower stemness (or ``max_len`` is reached),
    allowing longer sentences that better exploit the transformer's sequence modelling.
    """

    if "stemness_norm" not in adata.obs:
        raise ValueError("Stemness scores missing; run `score_stemness` first.")
    if "connectivities" not in adata.obsp or "leiden" not in adata.obs:
        raise ValueError("Topology missing; set `compute_topology=True` during preprocessing.")

    sequences: List[List[int]] = []
    edges = paga_edge_list(adata, min_connectivity=min_connectivity)
    if not edges:
        return sequences

    cluster_means = adata.obs.groupby("leiden")["stemness_norm"].mean().to_dict()
    conn_matrix = adata.obsp["connectivities"]
    neighbor_indices = [conn_matrix[idx].nonzero()[1] for idx in range(adata.n_obs)]
    stemness = adata.obs["stemness_norm"].to_numpy()

    for src, dst in edges:
        src_mean = cluster_means[src]
        dst_mean = cluster_means[dst]
        if src_mean == dst_mean:
            continue
        start_cluster, end_cluster = (src, dst) if src_mean > dst_mean else (dst, src)
        start_indices = np.where(adata.obs["leiden"].to_numpy() == start_cluster)[0]

        for start in start_indices:
            for _ in range(n_walks_per_cell):
                walk = [int(start)]
                current = start
                while max_len is None or len(walk) < max_len:
                    neighbors = [
                        n for n in neighbor_indices[current] if stemness[n] < stemness[current]
                    ]
                    if not neighbors:
                        break
                    current = int(np.random.choice(neighbors))
                    walk.append(current)
                if len(walk) > 1:
                    sequences.append(walk)
    return sequences

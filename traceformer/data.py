import os
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scanpy as sc
from anndata import AnnData


def load_adata_files(data_dir: str) -> List[AnnData]:
    """Load all `.h5ad` files from a directory into a list.

    Parameters
    ----------
    data_dir:
        Directory containing `.h5ad` files.

    Returns
    -------
    List[AnnData]
        List of loaded AnnData objects.
    """
    files = [
        os.path.join(data_dir, fname)
        for fname in os.listdir(data_dir)
        if fname.endswith(".h5ad")
    ]
    if not files:
        raise FileNotFoundError(f"No .h5ad files found in {data_dir}")
    return [sc.read_h5ad(path) for path in sorted(files)]


def preprocess_and_integrate(
    adatas: Iterable[AnnData],
    n_hvgs: int = 2000,
    pca_components: int = 50,
) -> AnnData:
    """Concatenate AnnData objects and run preprocessing.

    The returned AnnData contains:
    - `adata.layers['log1p']`: Log1p-normalised (unnscaled) expression used as model input.
    - `adata.obsm['X_pca_harmony']`: Harmony-corrected PCA space when `sample` column exists,
      otherwise the standard PCA representation.
    - Computed neighbours/leiden/paga on the harmony-aligned space.

    Parameters
    ----------
    adatas:
        Iterable of AnnData objects to concatenate.
    n_hvgs:
        Number of highly variable genes to keep.
    pca_components:
        Number of principal components for PCA/Harmony.
    """
    adata = sc.concat(list(adatas), join="outer", label="dataset", index_unique=None)
    # Normalise + log1p without scaling for model inputs.
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers["log1p"] = adata.X.copy()

    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvgs, flavor="seurat_v3")
    hvgs = adata.var[adata.var["highly_variable"]].index
    adata = adata[:, hvgs].copy()

    sc.tl.pca(adata, n_comps=pca_components)
    if "sample" in adata.obs.columns:
        sc.external.pp.harmony_integrate(adata, key="sample")
        adata.obsm["X_pca_harmony"] = adata.obsm["X_pca_harmony"][:, :pca_components]
    else:
        adata.obsm["X_pca_harmony"] = adata.obsm["X_pca"]

    sc.pp.neighbors(adata, use_rep="X_pca_harmony")
    sc.tl.leiden(adata, key_added="leiden")
    sc.tl.paga(adata, groups="leiden")
    return adata


def score_stemness(adata: AnnData, stemness_genes: Sequence[str]) -> None:
    """Compute stemness scores and store them in ``adata.obs``.

    Adds two columns:
    - ``stemness_score``: raw score from ``sc.tl.score_genes``.
    - ``stemness_norm``: min-max normalised scores in [0, 1].
    """
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
        return []
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
    max_len: int = 10,
    min_connectivity: float = 0.05,
) -> List[List[int]]:
    """Generate directed random-walk trajectories constrained by stemness gradients.

    Walks follow PAGA-connected cluster pairs, moving from higher mean stemness clusters
    toward lower mean stemness clusters. Within a walk, steps only follow neighbours whose
    stemness is lower than the current cell.
    """
    if "stemness_norm" not in adata.obs:
        raise ValueError("Stemness scores missing; run `score_stemness` first.")

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
                for _ in range(max_len - 1):
                    neighbors = [
                        n for n in neighbor_indices[current] if stemness[n] < stemness[current]
                    ]
                    if not neighbors:
                        break
                    current = int(np.random.choice(neighbors))
                    walk.append(current)
                    if adata.obs["leiden"].iat[current] == end_cluster:
                        break
                if len(walk) > 1:
                    sequences.append(walk)
    return sequences

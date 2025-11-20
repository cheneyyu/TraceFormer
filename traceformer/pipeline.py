from typing import Sequence, Tuple

import torch

from .data import generate_trajectory_sequences, load_adata_files, preprocess_and_integrate, score_stemness
from .model import Traceformer
from .training import create_dataloader, train_traceformer


def prepare_data(
    data_dir: str,
    stemness_genes: Sequence[str],
    n_walks_per_cell: int = 2,
    max_len: int = 10,
):
    """Load data, run preprocessing, score stemness, and generate trajectories."""
    adatas = load_adata_files(data_dir)
    adata = preprocess_and_integrate(adatas)
    score_stemness(adata, stemness_genes)
    sequences = generate_trajectory_sequences(
        adata, n_walks_per_cell=n_walks_per_cell, max_len=max_len
    )
    return adata, sequences


def run_training(
    data_dir: str,
    stemness_genes: Sequence[str],
    device: torch.device,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 4,
    batch_size: int = 8,
    epochs: int = 10,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
):
    """End-to-end training pipeline for Traceformer."""
    adata, sequences = prepare_data(data_dir, stemness_genes)
    dataloader = create_dataloader(adata, sequences, batch_size=batch_size)

    input_dim = adata.layers["log1p"].shape[1]
    model = Traceformer(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
    )
    train_traceformer(
        model,
        dataloader,
        device=device,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
    )
    return model

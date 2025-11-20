from typing import Sequence

import torch

from .data import load_adata_files, preprocess_and_integrate
from .model import GeneVAE
from .training import create_gene_dataloader, train_vae


def prepare_adata(data_dir: str, gene_list: Sequence[str]):
    """Load `.h5ad` files, normalise/log1p, and align to a shared gene list."""

    adatas = load_adata_files(data_dir)
    return preprocess_and_integrate(adatas, gene_list)


def train_new_vae(
    data_dir: str,
    gene_list: Sequence[str],
    device: torch.device,
    latent_dim: int = 256,
    hidden_dims: Sequence[int] | None = None,
    batch_size: int = 256,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    kl_weight: float = 1e-3,
) -> GeneVAE:
    """Train a VAE from scratch on log1p-normalised expression."""

    adata = prepare_adata(data_dir, gene_list)
    dataloader = create_gene_dataloader(adata, batch_size=batch_size)

    model = GeneVAE(
        input_dim=adata.n_vars,
        latent_dim=latent_dim,
        hidden_dims=list(hidden_dims) if hidden_dims is not None else None,
    )
    train_vae(
        model,
        dataloader,
        device=device,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        kl_weight=kl_weight,
    )
    return model


def continue_training(
    checkpoint_path: str,
    data_dir: str,
    gene_list: Sequence[str],
    device: torch.device,
    latent_dim: int = 256,
    hidden_dims: Sequence[int] | None = None,
    batch_size: int = 256,
    epochs: int = 10,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    kl_weight: float = 1e-3,
) -> GeneVAE:
    """Continue training a pretrained VAE on new AnnData collections."""

    adata = prepare_adata(data_dir, gene_list)
    dataloader = create_gene_dataloader(adata, batch_size=batch_size)

    model = GeneVAE(
        input_dim=adata.n_vars,
        latent_dim=latent_dim,
        hidden_dims=list(hidden_dims) if hidden_dims is not None else None,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    saved_genes = checkpoint.get("gene_list")
    if saved_genes is not None and list(saved_genes) != list(gene_list):
        raise ValueError("Checkpoint gene list does not match requested `gene_list`.")

    train_vae(
        model,
        dataloader,
        device=device,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        kl_weight=kl_weight,
    )
    return model


def save_checkpoint(model: GeneVAE, path: str, gene_list: Sequence[str]) -> None:
    """Save model weights together with the gene list used during training."""

    torch.save({"state_dict": model.state_dict(), "gene_list": list(gene_list)}, path)

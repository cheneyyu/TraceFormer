# TraceFormer

Log1p VAE training for scRNA expression with explicit gene-list alignment.

## Features
- Load and concatenate multiple `.h5ad` files and normalize every matrix to 1e4 counts followed by `log1p` for model input.
- Keep only the user-provided gene list; unseen genes are zero-filled to keep all AnnData objects aligned to the same size.
- 256-dimensional latent VAE (with configurable hidden layers) tailored for raw log1p expression values.
- Two workflows: train from scratch or continue training on new AnnData collections using pretrained weights (online/continual learning).

## Quickstart: train from scratch
```python
import torch
from traceformer.pipeline import save_checkpoint, train_new_vae

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gene_list = ["SOX2", "POU5F1", "NANOG"]

model = train_new_vae(
    data_dir="/path/to/h5ad_dir",
    gene_list=gene_list,
    device=device,
    epochs=50,
    batch_size=256,
)

# Persist weights along with the gene list for later online/continual training.
save_checkpoint(model, "vae_gene_list.ckpt", gene_list)
```

## Online/continual: continue training with pretrained weights on new AnnData
```python
import torch
from traceformer.pipeline import continue_training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gene_list = ["SOX2", "POU5F1", "NANOG"]

model = continue_training(
    checkpoint_path="vae_gene_list.ckpt",
    data_dir="/path/to/new_h5ad_dir",
    gene_list=gene_list,
    device=device,
    epochs=10,
    lr=1e-4,
)
```

## Notes
- Inputs use `log1p` of matrices normalized to 1e4 counts; no HVG filtering or batch correction is applied.
- Only genes in the provided list are retained; missing genes are zero-filled with consistent column ordering, which supports online/continual scenarios.
- The VAE defaults to `latent_dim=256`; adjust `kl_weight`, hidden widths, and dropout through `train_new_vae`/`continue_training` as needed.

# TraceFormer

Next token prediction for scRNA trajectory with a Harmony-aware, batch-invariant Transformer.

## Features
- Loads and concatenates multiple `.h5ad` count matrices.
- Harmony-corrected topology construction (PCA/Neighbors/Leiden/PAGA) while keeping unscaled log1p HVGs for model input.
- Stemness-aware, downhill random-walk trajectory corpus generation.
- Continuous-space autoregressive Transformer (Traceformer) with stemness time encoding and direction-aware loss.

## End-to-end usage
```python
import torch
from traceformer.pipeline import run_training

stemness_genes = ["SOX2", "POU5F1", "NANOG"]
model = run_training(
    data_dir="/path/to/h5ad_dir",
    stemness_genes=stemness_genes,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    epochs=5,
    batch_size=8,
)
```

## Modular API
```python
from traceformer import data, model, training
import torch

# 1) Preprocess + topology
adatas = data.load_adata_files("/path/to/h5ad_dir")
adata = data.preprocess_and_integrate(adatas)
data.score_stemness(adata, ["SOX2", "POU5F1", "NANOG"])
sequences = data.generate_trajectory_sequences(adata, n_walks_per_cell=2, max_len=12)

# 2) Dataloader
loader = training.create_dataloader(adata, sequences, batch_size=8)

# 3) Model + training
traceformer = model.Traceformer(input_dim=adata.layers["log1p"].shape[1])
training.train_traceformer(traceformer, loader, device=torch.device("cpu"), epochs=5)
```

## Notes
- Model inputs are the 2000 HVGs after log1p normalisation (no scaling, no batch correction).
- Graph construction (neighbors/Leiden/PAGA) always uses Harmony-corrected PCA when a `sample` column is present; otherwise standard PCA.
- Random walks move only to neighbours with lower stemness scores and follow PAGA edges from higher to lower mean-stemness clusters.

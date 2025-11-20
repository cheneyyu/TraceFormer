from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .model import DirectionAwareLoss, Traceformer


class TrajectoryDataset(Dataset):
    """Dataset that returns expression/stemness sequences from cell-index trajectories."""

    def __init__(self, adata, sequences: Sequence[Sequence[int]]):
        if "log1p" not in adata.layers:
            raise ValueError("`adata.layers['log1p']` missing. Run preprocessing first.")
        if "stemness_norm" not in adata.obs:
            raise ValueError("Stemness scores missing; run `score_stemness` first.")
        self.data = np.asarray(adata.layers["log1p"])
        self.stemness = adata.obs["stemness_norm"].to_numpy()
        self.sequences = [seq for seq in sequences if len(seq) > 1]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        seq = self.sequences[idx]
        expr = self.data[seq]
        stem = self.stemness[seq]
        return expr.astype(np.float32), stem.astype(np.float32)


def _pad_stack(arr_list: List[np.ndarray], max_len: int) -> np.ndarray:
    padded = []
    for arr in arr_list:
        pad_width = ((0, max_len - arr.shape[0]), (0, 0))
        padded.append(np.pad(arr, pad_width, constant_values=0.0))
    return np.stack(padded)


def collate_padded(batch: List[Tuple[np.ndarray, np.ndarray]]):
    """Collate variable-length sequences with right-padding."""
    exprs, stems = zip(*batch)
    lengths = [len(x) - 1 for x in exprs]  # inputs exclude final token
    max_len = max(lengths)

    inputs = [x[:-1] for x in exprs]
    targets = [x[1:] for x in exprs]
    stem_inputs = [s[:-1].reshape(-1, 1) for s in stems]

    padding_mask = np.zeros((len(exprs), max_len), dtype=bool)
    for i, length in enumerate(lengths):
        if length < max_len:
            padding_mask[i, length:] = True

    return (
        torch.from_numpy(_pad_stack(inputs, max_len)),
        torch.from_numpy(_pad_stack(stem_inputs, max_len)),
        torch.from_numpy(_pad_stack(targets, max_len)),
        torch.from_numpy(padding_mask),
    )


def train_traceformer(
    model: Traceformer,
    dataloader: DataLoader,
    device: torch.device,
    epochs: int = 10,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
) -> None:
    criterion = DirectionAwareLoss()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for inputs, stems, targets, padding_mask in dataloader:
            inputs = inputs.to(device)
            stems = stems.to(device)
            targets = targets.to(device)
            padding_mask = padding_mask.to(device)

            optimizer.zero_grad()
            preds = model(inputs, stems, padding_mask=padding_mask)
            loss = criterion(preds, targets, current=inputs)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)
        avg_loss = epoch_loss / len(dataloader.dataset)
        print(f"Epoch {epoch}: loss={avg_loss:.4f}")


def create_dataloader(adata, sequences: Sequence[Sequence[int]], batch_size: int = 8, shuffle: bool = True):
    dataset = TrajectoryDataset(adata, sequences)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_padded)

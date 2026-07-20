from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_ROOT = ROOT / "data" / "features"
DEFAULT_TEXT_NPZ = DEFAULT_FEATURE_ROOT / "L" / "biobert_cls_english.npz"
DEFAULT_DATASET_INDEX = ROOT / "data" / "dataset_index.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "mqp_group5cv"

REAL_MODES = ["A", "V", "L", "AV", "AL", "VL", "AVL"]
RECON_MODES = ["AR", "ALR"]
ALL_MODES = REAL_MODES + RECON_MODES
MODE_NAMES = {
    "A": "MRI only",
    "V": "MPM only",
    "L": "English report only",
    "AV": "MRI + MPM",
    "AL": "MRI + English report",
    "VL": "MPM + English report",
    "AVL": "MRI + MPM + English report",
    "AR": "MRI + reconstructed MPM",
    "ALR": "MRI + English report + reconstructed MPM",
}
MODE_MODALITIES = {
    "A": ("A",),
    "V": ("V",),
    "L": ("L",),
    "AV": ("A", "V"),
    "AL": ("A", "L"),
    "VL": ("V", "L"),
    "AVL": ("A", "V", "L"),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(
    feature_root: Path,
    text_npz: Path,
    dataset_index: Path,
    mri_h5: str,
    mpm_h5: str,
) -> dict:
    manifest = json.loads((feature_root / "manifest.json").read_text(encoding="utf-8"))
    names = sorted(manifest["files"].keys())
    with h5py.File(feature_root / mri_h5, "r") as mri_handle:
        missing = [name for name in names if name not in mri_handle]
        if missing:
            raise RuntimeError(f"MRI features missing: {missing}")
        mri = np.stack(
            [np.asarray(mri_handle[name], dtype=np.float32).reshape(-1) for name in names]
        )
    with h5py.File(feature_root / mpm_h5, "r") as mpm_handle:
        missing = [name for name in names if name not in mpm_handle]
        if missing:
            raise RuntimeError(f"MPM features missing: {missing}")
        mpm = np.stack([np.asarray(mpm_handle[name], dtype=np.float32) for name in names])
    if mpm.ndim == 2:
        mpm = mpm[:, np.newaxis, :]

    text_data = np.load(text_npz)
    text_lookup = {
        str(name): feature.astype(np.float32)
        for name, feature in zip(text_data["case_ids"], text_data["features"])
    }
    missing = [name for name in names if name not in text_lookup]
    if missing:
        raise RuntimeError(f"English BioBERT features missing: {missing}")
    text = np.stack([text_lookup[name] for name in names])

    index = pd.read_csv(dataset_index).set_index("case_id")
    absent = [name for name in names if name not in index.index]
    if absent:
        raise RuntimeError(f"Cases missing from dataset index: {absent}")
    labels = np.asarray(
        [1 if str(index.loc[name, "label"]).lower() == "pik3ca" else 0 for name in names],
        dtype=np.int64,
    )
    groups = np.asarray([str(index.loc[name, "mr_sha256"]) for name in names], dtype=object)
    return {
        "names": names,
        "A": mri,
        "V": mpm,
        "L": text,
        "y": labels,
        "groups": groups,
    }


def subset(data: dict, indices: np.ndarray) -> dict:
    return {
        "names": [data["names"][int(index)] for index in indices],
        "A": data["A"][indices],
        "V": data["V"][indices],
        "L": data["L"][indices],
        "y": data["y"][indices],
        "groups": data["groups"][indices],
    }


class FoldPreprocessor:
    def __init__(self) -> None:
        self.stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def fit(self, train: dict) -> "FoldPreprocessor":
        for modality in ("A", "L"):
            mean = train[modality].mean(axis=0, keepdims=True)
            std = train[modality].std(axis=0, keepdims=True)
            std[std < 1e-6] = 1.0
            self.stats[modality] = (mean.astype(np.float32), std.astype(np.float32))
        flat = train["V"].reshape(-1, train["V"].shape[-1])
        mean = flat.mean(axis=0, keepdims=True)
        std = flat.std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        self.stats["V"] = (mean.astype(np.float32), std.astype(np.float32))
        return self

    def transform_one(self, split: dict) -> dict:
        mean_a, std_a = self.stats["A"]
        mean_l, std_l = self.stats["L"]
        mean_v, std_v = self.stats["V"]
        return {
            "names": split["names"],
            "A": ((split["A"] - mean_a) / std_a).astype(np.float32),
            "V": (
                (split["V"] - mean_v.reshape(1, 1, -1))
                / std_v.reshape(1, 1, -1)
            ).astype(np.float32),
            "L": ((split["L"] - mean_l) / std_l).astype(np.float32),
            "y": split["y"],
        }

    def transform(self, splits: dict[str, dict]) -> dict[str, dict]:
        return {name: self.transform_one(split) for name, split in splits.items()}


def to_tensors(split: dict, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "A": torch.as_tensor(split["A"], dtype=torch.float32, device=device),
        "V": torch.as_tensor(split["V"], dtype=torch.float32, device=device),
        "L": torch.as_tensor(split["L"], dtype=torch.float32, device=device),
        "y": torch.as_tensor(split["y"], dtype=torch.long, device=device),
    }


class CompactBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


class MultiScaleMPMBackbone(nn.Module):
    def __init__(self, input_dim: int, scales: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.scales = scales
        self.token_encoder = CompactBackbone(input_dim, hidden_dim, dropout)
        self.scale_embedding = nn.Parameter(torch.randn(1, scales, hidden_dim) * 0.02)
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            1,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, scales, feature_dim = value.shape
        tokens = self.token_encoder(value.reshape(batch * scales, feature_dim))
        tokens = tokens.reshape(batch, scales, -1) + self.scale_embedding[:, :scales]
        query = self.query.expand(batch, -1, -1)
        pooled, _ = self.attention(query, tokens, tokens, need_weights=False)
        return self.norm(pooled.squeeze(1) + tokens.mean(dim=1))


class GaussianDisentangler(nn.Module):
    def __init__(self, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.shared_mu = nn.Linear(hidden_dim, latent_dim)
        self.shared_logvar = nn.Linear(hidden_dim, latent_dim)
        self.private_mu = nn.Linear(hidden_dim, latent_dim)
        self.private_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "shared_mu": self.shared_mu(hidden),
            "shared_logvar": self.shared_logvar(hidden).clamp(-6.0, 2.0),
            "private_mu": self.private_mu(hidden),
            "private_logvar": self.private_logvar(hidden).clamp(-6.0, 2.0),
            "hidden": hidden,
        }


@dataclass
class MQPConfig:
    hidden_dim: int = 32
    latent_dim: int = 16
    dropout: float = 0.12
    lr: float = 0.002
    student_lr: float = 0.003
    weight_decay: float = 0.03
    mi_weight: float = 0.04
    disentangle_weight: float = 0.04
    kl_weight: float = 0.0005
    prototype_weight: float = 0.08
    uncertainty_weight: float = 0.05
    reconstruction_weight: float = 1.0
    kd_weight: float = 0.6
    student_ce_weight: float = 0.7
    temperature: float = 2.0


class MQP(nn.Module):
    """MIDAS-inspired medical multimodal model.

    M: mutual-information-guided shared/private disentanglement
    Q: quality and posterior-uncertainty-aware fusion
    P: predicted-class prototype-conditioned missing-MPM reconstruction
    """

    def __init__(
        self,
        dim_mri: int,
        dim_mpm: int,
        dim_text: int,
        mpm_scales: int,
        config: MQPConfig,
    ) -> None:
        super().__init__()
        hidden = config.hidden_dim
        latent = config.latent_dim
        self.config = config
        self.modalities = ("A", "V", "L")
        self.backbones = nn.ModuleDict(
            {
                "A": CompactBackbone(dim_mri, hidden, config.dropout),
                "V": MultiScaleMPMBackbone(
                    dim_mpm,
                    mpm_scales,
                    hidden,
                    config.dropout,
                ),
                "L": CompactBackbone(dim_text, hidden, config.dropout),
            }
        )
        self.disentanglers = nn.ModuleDict(
            {modality: GaussianDisentangler(hidden, latent) for modality in self.modalities}
        )
        self.private_adapters = nn.ModuleDict(
            {modality: nn.Linear(latent, latent) for modality in self.modalities}
        )
        self.modality_embedding = nn.ParameterDict(
            {
                modality: nn.Parameter(torch.randn(1, latent) * 0.02)
                for modality in self.modalities
            }
        )
        self.quality_heads = nn.ModuleDict(
            {
                modality: nn.Sequential(
                    nn.Linear(latent * 2, latent),
                    nn.GELU(),
                    nn.Linear(latent, 1),
                )
                for modality in self.modalities
            }
        )
        self.fusion_norm = nn.LayerNorm(latent)
        self.fusion_ffn = nn.Sequential(
            nn.Linear(latent, latent * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(latent * 2, latent),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(latent),
            nn.Dropout(config.dropout),
            nn.Linear(latent, 2),
        )
        self.class_prototypes = nn.Parameter(torch.randn(2, latent) * 0.02)
        self.reconstructor = nn.Sequential(
            nn.LayerNorm(latent * 2),
            nn.Linear(latent * 2, hidden * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden * 2, latent * 4),
        )

    @staticmethod
    def sample(mu: torch.Tensor, logvar: torch.Tensor, stochastic: bool) -> torch.Tensor:
        if not stochastic:
            return mu
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def encode(self, batch: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
        hidden = {
            "A": self.backbones["A"](batch["A"]),
            "V": self.backbones["V"](batch["V"]),
            "L": self.backbones["L"](batch["L"]),
        }
        return {
            modality: self.disentanglers[modality](hidden[modality])
            for modality in self.modalities
        }

    def token_and_score(
        self,
        modality: str,
        factor: dict[str, torch.Tensor],
        stochastic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shared = self.sample(
            factor["shared_mu"],
            factor["shared_logvar"],
            stochastic,
        )
        private = self.sample(
            factor["private_mu"],
            factor["private_logvar"],
            stochastic,
        )
        token = shared + self.private_adapters[modality](private)
        token = token + self.modality_embedding[modality]
        posterior_variance = 0.5 * (
            factor["shared_logvar"].exp().mean(dim=1, keepdim=True)
            + factor["private_logvar"].exp().mean(dim=1, keepdim=True)
        )
        learned_quality = self.quality_heads[modality](
            torch.cat([factor["shared_mu"], factor["private_mu"]], dim=1)
        )
        score = learned_quality - posterior_variance
        confidence = torch.sigmoid(score)
        return token, score, confidence

    def fuse(
        self,
        factors: dict[str, dict[str, torch.Tensor]],
        present: tuple[str, ...],
        stochastic: bool | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if stochastic is None:
            stochastic = self.training
        tokens = []
        scores = []
        confidences = []
        for modality in present:
            token, score, confidence = self.token_and_score(
                modality,
                factors[modality],
                stochastic,
            )
            tokens.append(token)
            scores.append(score)
            confidences.append(confidence)
        score_tensor = torch.cat(scores, dim=1)
        weights = torch.softmax(score_tensor, dim=1)
        token_tensor = torch.stack(tokens, dim=1)
        fused = (weights.unsqueeze(-1) * token_tensor).sum(dim=1)
        fused = self.fusion_norm(fused)
        fused = fused + self.fusion_ffn(fused)
        logits = self.classifier(fused)
        return logits, {
            "fused": fused,
            "weights": weights,
            "confidence": torch.cat(confidences, dim=1),
            "present": present,
        }

    def reconstruct_mpm(
        self,
        factors: dict[str, dict[str, torch.Tensor]],
        source_modalities: tuple[str, ...],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        source_logits, source_detail = self.fuse(
            factors,
            source_modalities,
            stochastic=self.training,
        )
        probabilities = torch.softmax(source_logits, dim=1)
        soft_prototype = probabilities @ self.class_prototypes
        parameters = self.reconstructor(
            torch.cat([source_detail["fused"], soft_prototype], dim=1)
        )
        shared_mu, shared_logvar, private_mu, private_logvar = parameters.chunk(4, dim=1)
        virtual = {
            "shared_mu": shared_mu,
            "shared_logvar": shared_logvar.clamp(-6.0, 2.0),
            "private_mu": private_mu,
            "private_logvar": private_logvar.clamp(-6.0, 2.0),
            "hidden": source_detail["fused"],
        }
        return virtual, {
            "source_logits": source_logits,
            "source_probabilities": probabilities,
        }

    def forward_mode(
        self,
        factors: dict[str, dict[str, torch.Tensor]],
        mode: str,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if mode in MODE_MODALITIES:
            return self.fuse(factors, MODE_MODALITIES[mode])
        if mode == "AR":
            virtual, recon_detail = self.reconstruct_mpm(factors, ("A",))
            logits, detail = self.fuse({**factors, "V": virtual}, ("A", "V"))
        elif mode == "ALR":
            virtual, recon_detail = self.reconstruct_mpm(factors, ("A", "L"))
            logits, detail = self.fuse({**factors, "V": virtual}, ("A", "V", "L"))
        else:
            raise ValueError(mode)
        detail.update(recon_detail)
        detail["virtual_v"] = virtual
        return logits, detail


def info_nce(first: torch.Tensor, second: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    first = F.normalize(first, dim=1)
    second = F.normalize(second, dim=1)
    logits = first @ second.T / temperature
    target = torch.arange(first.shape[0], device=first.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))


def cross_covariance_loss(shared: torch.Tensor, private: torch.Tensor) -> torch.Tensor:
    shared = shared - shared.mean(dim=0, keepdim=True)
    private = private - private.mean(dim=0, keepdim=True)
    denominator = max(shared.shape[0] - 1, 1)
    covariance = shared.T @ private / denominator
    return covariance.square().mean()


def gaussian_kl(factor: dict[str, torch.Tensor]) -> torch.Tensor:
    losses = []
    for prefix in ("shared", "private"):
        mu = factor[f"{prefix}_mu"]
        logvar = factor[f"{prefix}_logvar"]
        losses.append(-0.5 * (1.0 + logvar - mu.square() - logvar.exp()).mean())
    return sum(losses)


def relational_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    predicted = F.normalize(predicted, dim=1)
    target = F.normalize(target, dim=1)
    return F.mse_loss(predicted @ predicted.T, target @ target.T)


def class_weights(labels: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=2).float()
    weights = counts.sum() / counts.clamp_min(1.0)
    return weights / weights.mean()


def teacher_objective(
    model: MQP,
    batch: dict[str, torch.Tensor],
    config: MQPConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    factors = model.encode(batch)
    weights = class_weights(batch["y"])
    mode_coefficients = {
        "A": 0.25,
        "V": 0.25,
        "L": 0.15,
        "AV": 0.65,
        "AL": 0.40,
        "VL": 0.40,
        "AVL": 1.00,
    }
    classification = torch.zeros((), device=batch["y"].device)
    calibration = torch.zeros_like(classification)
    coefficient_sum = sum(mode_coefficients.values())
    for mode, coefficient in mode_coefficients.items():
        logits, detail = model.forward_mode(factors, mode)
        per_sample = F.cross_entropy(logits, batch["y"], weight=weights, reduction="none")
        classification = classification + coefficient * per_sample.mean()
        if mode in ("A", "V", "L"):
            target_confidence = torch.exp(-per_sample.detach()).clamp(0.05, 0.95)
            calibration = calibration + F.mse_loss(
                detail["confidence"].squeeze(1),
                target_confidence,
            )
    classification = classification / coefficient_sum

    shared = {modality: factors[modality]["shared_mu"] for modality in ("A", "V", "L")}
    mi_alignment = (
        info_nce(shared["A"], shared["V"])
        + info_nce(shared["A"], shared["L"])
        + info_nce(shared["V"], shared["L"])
    ) / 3.0
    disentangle = sum(
        cross_covariance_loss(
            factors[modality]["shared_mu"],
            factors[modality]["private_mu"],
        )
        for modality in ("A", "V", "L")
    ) / 3.0
    kl = sum(gaussian_kl(factors[modality]) for modality in ("A", "V", "L")) / 3.0
    prototype = F.smooth_l1_loss(
        factors["V"]["shared_mu"],
        model.class_prototypes[batch["y"]],
    )
    loss = (
        classification
        + config.mi_weight * mi_alignment
        + config.disentangle_weight * disentangle
        + config.kl_weight * kl
        + config.prototype_weight * prototype
        + config.uncertainty_weight * calibration
    )
    return loss, {
        "classification": float(classification.detach()),
        "mi_alignment": float(mi_alignment.detach()),
        "disentangle": float(disentangle.detach()),
        "kl": float(kl.detach()),
        "prototype": float(prototype.detach()),
        "calibration": float(calibration.detach()),
    }


def student_objective(
    model: MQP,
    batch: dict[str, torch.Tensor],
    config: MQPConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    with torch.no_grad():
        factors = model.encode(batch)
        teacher_logits, _ = model.forward_mode(factors, "AVL")
        target_v = {
            key: value.detach()
            for key, value in factors["V"].items()
        }
    virtual_al, _ = model.reconstruct_mpm(factors, ("A", "L"))
    virtual_a, _ = model.reconstruct_mpm(factors, ("A",))
    logits_alr, _ = model.fuse({**factors, "V": virtual_al}, ("A", "V", "L"))
    logits_ar, _ = model.fuse({**factors, "V": virtual_a}, ("A", "V"))

    latent = (
        F.smooth_l1_loss(virtual_al["shared_mu"], target_v["shared_mu"])
        + F.smooth_l1_loss(virtual_al["private_mu"], target_v["private_mu"])
        + 0.5 * F.smooth_l1_loss(virtual_a["shared_mu"], target_v["shared_mu"])
        + 0.5 * F.smooth_l1_loss(virtual_a["private_mu"], target_v["private_mu"])
    )
    cosine = (
        1.0
        - F.cosine_similarity(
            virtual_al["shared_mu"],
            target_v["shared_mu"],
            dim=1,
        ).mean()
    )
    relation = relational_loss(virtual_al["shared_mu"], target_v["shared_mu"])
    uncertainty = (
        F.smooth_l1_loss(virtual_al["shared_logvar"], target_v["shared_logvar"])
        + F.smooth_l1_loss(virtual_al["private_logvar"], target_v["private_logvar"])
    )
    temperature = config.temperature
    distillation = F.kl_div(
        F.log_softmax(logits_alr / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature**2)
    weights = class_weights(batch["y"])
    classification = 0.65 * F.cross_entropy(logits_alr, batch["y"], weight=weights)
    classification = classification + 0.35 * F.cross_entropy(
        logits_ar,
        batch["y"],
        weight=weights,
    )
    prototype = F.smooth_l1_loss(
        target_v["shared_mu"],
        model.class_prototypes[batch["y"]],
    )
    loss = (
        config.reconstruction_weight * (latent + 0.5 * cosine + 0.1 * relation)
        + 0.1 * uncertainty
        + config.kd_weight * distillation
        + config.student_ce_weight * classification
        + config.prototype_weight * prototype
    )
    return loss, {
        "latent": float(latent.detach()),
        "cosine": float(cosine.detach()),
        "relation": float(relation.detach()),
        "uncertainty": float(uncertainty.detach()),
        "distillation": float(distillation.detach()),
        "classification": float(classification.detach()),
        "prototype": float(prototype.detach()),
    }


def metrics(target: np.ndarray, probability: np.ndarray) -> dict:
    prediction = (probability >= 0.5).astype(np.int64)
    try:
        auc = float(roc_auc_score(target, probability))
    except ValueError:
        auc = None
    return {
        "n": int(target.size),
        "acc": float(accuracy_score(target, prediction)),
        "bal": float(balanced_accuracy_score(target, prediction)),
        "f1": float(f1_score(target, prediction, average="macro", zero_division=0)),
        "auc": auc,
        "cm": confusion_matrix(target, prediction, labels=[0, 1]).astype(int).tolist(),
        "prediction": prediction.tolist(),
        "probability": probability.tolist(),
    }


@torch.inference_mode()
def evaluate(model: MQP, split_np: dict, split_t: dict, mode: str) -> dict:
    model.eval()
    factors = model.encode(split_t)
    logits, detail = model.forward_mode(factors, mode)
    probability = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    result = metrics(split_np["y"], probability)
    result["names"] = split_np["names"]
    result["target"] = split_np["y"].tolist()
    result["fusion_weights"] = detail["weights"].cpu().numpy().tolist()
    result["confidence"] = detail["confidence"].cpu().numpy().tolist()
    return result


def result_key(result: dict, modes: tuple[str, ...]) -> tuple[float, ...]:
    f1 = np.mean([result[mode]["f1"] for mode in modes])
    aucs = [result[mode]["auc"] for mode in modes if result[mode]["auc"] is not None]
    auc = float(np.mean(aucs)) if aucs else -1.0
    return float(f1), auc


def clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def train_teacher(
    model: MQP,
    train_t: dict,
    val_np: dict,
    val_t: dict,
    config: MQPConfig,
    epochs: int,
    patience: int,
) -> dict:
    parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("reconstructor")
    ]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    best = None
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        loss, terms = teacher_objective(model, train_t, config)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 2.0)
        optimizer.step()

        model.eval()
        validation = {
            mode: evaluate(model, val_np, val_t, mode)
            for mode in ("A", "V", "L", "AV", "AL", "VL", "AVL")
        }
        key = result_key(validation, ("AV", "VL", "AVL"))
        if best is None or key > best["key"]:
            best = {
                "key": key,
                "epoch": epoch,
                "state_dict": clone_state(model),
                "terms": terms,
            }
            stale = 0
        else:
            stale += 1
        if epoch >= 40 and stale >= patience:
            break
    model.load_state_dict(best["state_dict"])
    return best


def train_student(
    model: MQP,
    train_t: dict,
    val_np: dict,
    val_t: dict,
    config: MQPConfig,
    epochs: int,
    patience: int,
) -> dict:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.reconstructor.parameters():
        parameter.requires_grad = True
    model.class_prototypes.requires_grad = True
    parameters = list(model.reconstructor.parameters()) + [model.class_prototypes]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.student_lr,
        weight_decay=config.weight_decay,
    )
    best = None
    stale = 0
    for epoch in range(1, epochs + 1):
        model.eval()
        model.reconstructor.train()
        loss, terms = student_objective(model, train_t, config)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 2.0)
        optimizer.step()

        validation = {
            mode: evaluate(model, val_np, val_t, mode)
            for mode in ("AR", "ALR")
        }
        key = result_key(validation, ("AR", "ALR"))
        if best is None or key > best["key"]:
            best = {
                "key": key,
                "epoch": epoch,
                "state_dict": clone_state(model),
                "terms": terms,
            }
            stale = 0
        else:
            stale += 1
        if epoch >= 50 and stale >= patience:
            break
    model.load_state_dict(best["state_dict"])
    for parameter in model.parameters():
        parameter.requires_grad = True
    return best


def split_outer_fold(
    data: dict,
    train_val_indices: np.ndarray,
    test_indices: np.ndarray,
    seed: int,
) -> dict:
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    labels = data["y"][train_val_indices]
    groups = data["groups"][train_val_indices]
    train_relative, val_relative = next(
        splitter.split(np.zeros(train_val_indices.size), labels, groups)
    )
    return {
        "train": subset(data, train_val_indices[train_relative]),
        "val": subset(data, train_val_indices[val_relative]),
        "test": subset(data, test_indices),
    }


def train_run(
    processed: dict,
    seed: int,
    config: MQPConfig,
    teacher_epochs: int,
    student_epochs: int,
    patience: int,
    device: torch.device,
) -> tuple[MQP, dict]:
    set_seed(seed)
    tensors = {
        split_name: to_tensors(split, device)
        for split_name, split in processed.items()
    }
    model = MQP(
        dim_mri=processed["train"]["A"].shape[1],
        dim_mpm=processed["train"]["V"].shape[-1],
        dim_text=processed["train"]["L"].shape[1],
        mpm_scales=processed["train"]["V"].shape[1],
        config=config,
    ).to(device)
    teacher = train_teacher(
        model,
        tensors["train"],
        processed["val"],
        tensors["val"],
        config,
        teacher_epochs,
        patience,
    )
    student = train_student(
        model,
        tensors["train"],
        processed["val"],
        tensors["val"],
        config,
        student_epochs,
        patience,
    )
    validation = {
        mode: evaluate(model, processed["val"], tensors["val"], mode)
        for mode in ALL_MODES
    }
    test = {
        mode: evaluate(model, processed["test"], tensors["test"], mode)
        for mode in ALL_MODES
    }
    return model, {
        "seed": seed,
        "config": asdict(config),
        "best_teacher_epoch": teacher["epoch"],
        "best_student_epoch": student["epoch"],
        "teacher_terms": teacher["terms"],
        "student_terms": student["terms"],
        "val": validation,
        "test": test,
    }


def summarize(folds: list[dict]) -> dict:
    summary = {}
    for mode in ALL_MODES:
        results = [fold["selected"]["test"][mode] for fold in folds]
        summary[mode] = {"name": MODE_NAMES[mode]}
        for metric_name in ("acc", "bal", "f1", "auc"):
            values = [
                result[metric_name]
                for result in results
                if result[metric_name] is not None
            ]
            summary[mode][f"{metric_name}_mean"] = float(np.mean(values))
            summary[mode][f"{metric_name}_std"] = float(np.std(values))
        confidences = [
            confidence
            for result in results
            for row in result["confidence"]
            for confidence in row
        ]
        summary[mode]["confidence_mean"] = float(np.mean(confidences))
    return summary


def write_outputs(
    output_dir: Path,
    data: dict,
    fold_results: list[dict],
    summary: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    payload = {
        "model": "MQP",
        "model_expansion": (
            "Mutual-information disentanglement, Quality-aware fusion, "
            "Prototype-conditioned missing-MPM reconstruction"
        ),
        "inspiration": {
            "paper": (
                "MIDAS: Mutual Information Disentanglement with "
                "Uncertainty-Aware Fusion for Incomplete Multimodal Sentiment Analysis"
            ),
            "doi": "10.1109/TPAMI.2026.3713694",
            "adaptation": (
                "The medical adaptation uses fold-trained Gaussian shared/private factors, "
                "InfoNCE alignment, shared-private cross-covariance minimization, posterior-"
                "variance quality weighting, and predicted-class prototype reconstruction."
            ),
        },
        "encoders": {
            "MRI": "MedCLIP ResNet50 ROI feature, frozen precomputed 512-D",
            "MPM": "DINOv2-small multi-scale feature, frozen precomputed 3x384-D",
            "Text": "BioBERT base cased v1.1 CLS from all-English reports, frozen precomputed 768-D",
        },
        "n": len(data["y"]),
        "class_counts": np.bincount(data["y"], minlength=2).astype(int).tolist(),
        "unique_groups": len(set(data["groups"])),
        "folds": args.folds,
        "outer_seed": args.outer_seed,
        "candidate_seeds": [
            int(seed)
            for seed in args.seeds.split(",")
            if seed.strip()
        ],
        "device": str(device),
        "feature_root": str(args.feature_root),
        "text_npz": str(args.text_npz),
        "dataset_index": str(args.dataset_index),
        "summary": summary,
        "fold_results": fold_results,
    }
    (output_dir / "cv_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "cv_summary.tsv").open("w", encoding="utf-8") as handle:
        handle.write(
            "mode\tname\tacc_mean\tacc_std\tbal_mean\tbal_std\t"
            "f1_mean\tf1_std\tauc_mean\tauc_std\tconfidence_mean\n"
        )
        for mode in ALL_MODES:
            result = summary[mode]
            handle.write(
                f"{mode}\t{result['name']}\t"
                f"{result['acc_mean']:.4f}\t{result['acc_std']:.4f}\t"
                f"{result['bal_mean']:.4f}\t{result['bal_std']:.4f}\t"
                f"{result['f1_mean']:.4f}\t{result['f1_std']:.4f}\t"
                f"{result['auc_mean']:.4f}\t{result['auc_std']:.4f}\t"
                f"{result['confidence_mean']:.4f}\n"
            )
    with (output_dir / "oof_predictions.tsv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "fold",
                "mode",
                "case_id",
                "target",
                "prediction",
                "probability_pik3ca",
                "fusion_weights",
                "confidence",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for fold in fold_results:
            for mode in ALL_MODES:
                result = fold["selected"]["test"][mode]
                for index, case_id in enumerate(result["names"]):
                    writer.writerow(
                        {
                            "fold": fold["fold"],
                            "mode": mode,
                            "case_id": case_id,
                            "target": result["target"][index],
                            "prediction": result["prediction"][index],
                            "probability_pik3ca": result["probability"][index],
                            "fusion_weights": json.dumps(result["fusion_weights"][index]),
                            "confidence": json.dumps(result["confidence"][index]),
                        }
                    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--text-npz", type=Path, default=DEFAULT_TEXT_NPZ)
    parser.add_argument("--dataset-index", type=Path, default=DEFAULT_DATASET_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mri-h5", default="A/mri_medclip_roi.h5")
    parser.add_argument("--mpm-h5", default="V/mpm_dinov2_multiscale.h5")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--outer-seed", type=int, default=2026)
    parser.add_argument("--seeds", default="7,13")
    parser.add_argument("--teacher-epochs", type=int, default=180)
    parser.add_argument("--student-epochs", type=int, default=220)
    parser.add_argument("--patience", type=int, default=45)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    data = load_data(
        args.feature_root,
        args.text_npz,
        args.dataset_index,
        args.mri_h5,
        args.mpm_h5,
    )
    if args.quick:
        args.teacher_epochs = min(args.teacher_epochs, 3)
        args.student_epochs = min(args.student_epochs, 3)
        args.patience = 3
        args.seeds = args.seeds.split(",")[0]

    configs = [
        MQPConfig(hidden_dim=32, latent_dim=16, dropout=0.10),
        MQPConfig(
            hidden_dim=40,
            latent_dim=20,
            dropout=0.15,
            mi_weight=0.03,
            disentangle_weight=0.06,
            prototype_weight=0.10,
        ),
    ]
    if args.quick:
        configs = configs[:1]

    outer = StratifiedGroupKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.outer_seed,
    )
    seeds = [int(seed) for seed in args.seeds.split(",") if seed.strip()]
    fold_results = []
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for fold, (train_val_indices, test_indices) in enumerate(
        outer.split(np.zeros(len(data["y"])), data["y"], data["groups"]),
        1,
    ):
        raw = split_outer_fold(
            data,
            train_val_indices,
            test_indices,
            9100 + fold,
        )
        preprocessor = FoldPreprocessor().fit(raw["train"])
        processed = preprocessor.transform(raw)
        selected = None
        runs = []
        for seed in seeds:
            for config in configs:
                model, result = train_run(
                    processed,
                    seed,
                    config,
                    args.teacher_epochs,
                    args.student_epochs,
                    args.patience,
                    device,
                )
                runs.append(result)
                recon_key = result_key(result["val"], ("AR", "ALR"))
                full_key = result_key(result["val"], ("AV", "VL", "AVL"))
                key = recon_key + full_key
                if selected is None or key > selected["key"]:
                    selected = {
                        "key": key,
                        "result": result,
                        "state_dict": clone_state(model),
                    }
                print(
                    json.dumps(
                        {
                            "fold": fold,
                            "seed": seed,
                            "hidden_dim": config.hidden_dim,
                            "latent_dim": config.latent_dim,
                            "teacher_epoch": result["best_teacher_epoch"],
                            "student_epoch": result["best_student_epoch"],
                            "val_AVL_auc": result["val"]["AVL"]["auc"],
                            "val_ALR_auc": result["val"]["ALR"]["auc"],
                            "test_AVL_auc": result["test"]["AVL"]["auc"],
                            "test_ALR_auc": result["test"]["ALR"]["auc"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

        splits = {
            split_name: {
                "n": len(split["y"]),
                "counts": np.bincount(split["y"], minlength=2).astype(int).tolist(),
                "names": split["names"],
            }
            for split_name, split in raw.items()
        }
        fold_payload = {
            "fold": fold,
            "splits": splits,
            "selected": selected["result"],
            "all_runs": runs,
        }
        fold_results.append(fold_payload)
        (args.output_dir / f"fold_{fold}.json").write_text(
            json.dumps(fold_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        torch.save(
            {
                "model": "MQP",
                "state_dict": selected["state_dict"],
                "config": selected["result"]["config"],
                "seed": selected["result"]["seed"],
                "preprocessor_stats": preprocessor.stats,
                "splits": splits,
                "feature_files": {
                    "A": args.mri_h5,
                    "V": args.mpm_h5,
                    "L": str(args.text_npz),
                },
            },
            checkpoint_dir / f"fold_{fold}.pt",
        )

    summary = summarize(fold_results)
    write_outputs(
        args.output_dir,
        data,
        fold_results,
        summary,
        args,
        device,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved={args.output_dir}")


if __name__ == "__main__":
    main()

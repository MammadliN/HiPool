import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from MainClasses.Models import Baseline, CDur, TALNet
from MainClasses.MILPooling import MILPooling


@dataclass
class AudioSegment:
    audio_path: str
    start_sec: float
    end_sec: float
    label: np.ndarray
    segment_id: str


class AudioSegmentDataset(Dataset):
    def __init__(
        self,
        segments: List[AudioSegment],
        sample_rate: int,
        n_mels: int,
        n_fft: int,
        hop_length: int,
        segment_seconds: int,
    ):
        self.segments = segments
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.segment_seconds = segment_seconds
        self.seq_len = int(np.floor((segment_seconds * sample_rate - n_fft) / hop_length) + 1)

    def __len__(self) -> int:
        return len(self.segments)

    def _load_segment(self, audio_path: str, start_sec: float, end_sec: float) -> np.ndarray:
        audio, _ = librosa.load(audio_path, sr=self.sample_rate)
        start_sample = int(start_sec * self.sample_rate)
        end_sample = int(end_sec * self.sample_rate)
        segment = audio[start_sample:end_sample]
        target_len = int((end_sec - start_sec) * self.sample_rate)
        if len(segment) < target_len:
            padding = np.zeros(target_len - len(segment), dtype=segment.dtype)
            segment = np.concatenate([segment, padding])
        return segment

    def _extract_logmel(self, audio: np.ndarray) -> np.ndarray:
        melspec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            win_length=self.n_fft,
            center=False,
            window="hamming",
        )
        logmelspec = librosa.power_to_db(melspec)
        logmelspec = logmelspec.T
        if logmelspec.shape[0] < self.seq_len:
            pad = np.zeros((self.seq_len - logmelspec.shape[0], self.n_mels), dtype=logmelspec.dtype)
            logmelspec = np.concatenate([logmelspec, pad], axis=0)
        elif logmelspec.shape[0] > self.seq_len:
            logmelspec = logmelspec[: self.seq_len]
        return logmelspec

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        seg = self.segments[idx]
        audio = self._load_segment(seg.audio_path, seg.start_sec, seg.end_sec)
        feats = self._extract_logmel(audio)
        return (
            torch.from_numpy(feats).float(),
            torch.from_numpy(seg.label).float(),
            seg.segment_id,
        )


class CNNBiGRU(nn.Module):
    def __init__(self, n_classes: int, pool_style: str, seq_len: int, n_mels: int = 64):
        super().__init__()
        self.pool_style = pool_style
        self.pool = MILPooling(n_classes=n_classes, seq_len=seq_len).get_pool(pool_style)
        self.name = f"cnn_bigru_{pool_style}"
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )
        self.gru = nn.GRU(
            input_size=(n_mels // 4) * 64,
            hidden_size=128,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.out = nn.Linear(256, n_classes)

    def forward(self, inputs: torch.Tensor, upsample: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        x = inputs.unsqueeze(1)
        x = self.features(x)
        x = x.transpose(1, 2).flatten(2)
        x, _ = self.gru(x)
        y_frames = torch.sigmoid(self.out(x)).clamp(1e-7, 1.0)
        y_clip = self.pool(y_frames)
        return y_clip, y_frames


class CNNTransformer(nn.Module):
    def __init__(self, n_classes: int, pool_style: str, seq_len: int, n_mels: int = 64):
        super().__init__()
        self.pool_style = pool_style
        self.pool = MILPooling(n_classes=n_classes, seq_len=seq_len).get_pool(pool_style)
        self.name = f"cnn_transformer_{pool_style}"
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=(n_mels // 4) * 64,
            nhead=4,
            dim_feedforward=256,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.out = nn.Linear((n_mels // 4) * 64, n_classes)

    def forward(self, inputs: torch.Tensor, upsample: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        x = inputs.unsqueeze(1)
        x = self.cnn(x)
        x = x.transpose(1, 2).flatten(2)
        x = self.transformer(x)
        y_frames = torch.sigmoid(self.out(x)).clamp(1e-7, 1.0)
        y_clip = self.pool(y_frames)
        return y_clip, y_frames


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_class_columns(metadata: pd.DataFrame, target_species: List[str]) -> List[str]:
    start_idx = metadata.columns.get_loc("subset") + 1
    all_species = list(metadata.columns[start_idx:])
    if not target_species:
        return all_species
    return [col for col in all_species if col in target_species]


def build_anuraset_segments(
    root_path: str,
    metadata: pd.DataFrame,
    class_columns: List[str],
    segment_seconds: int,
    subset: str,
) -> List[AudioSegment]:
    subset_meta = metadata[metadata["subset"] == subset]
    segments: List[AudioSegment] = []
    grouped = subset_meta.groupby(["site", "fname"])
    for (site, fname), group in grouped:
        audio_path = os.path.join(root_path, site, f"{fname}.wav")
        duration = 60
        for start_sec in range(0, duration, segment_seconds):
            end_sec = start_sec + segment_seconds
            label_rows = group[(group["min_t"] >= start_sec) & (group["max_t"] <= end_sec)]
            if label_rows.empty:
                label = np.zeros(len(class_columns), dtype=np.float32)
            else:
                label = label_rows[class_columns].max().values.astype(np.float32)
            segment_id = f"{fname}_{start_sec:02d}_{end_sec:02d}"
            segments.append(
                AudioSegment(
                    audio_path=audio_path,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    label=label,
                    segment_id=segment_id,
                )
            )
    return segments


def build_fnjv_segments(
    root_path: str,
    class_columns: List[str],
    segment_seconds: int,
) -> Tuple[List[AudioSegment], List[str]]:
    metadata_path = os.path.join(root_path, "metadata_filtered_filled.csv")
    metadata = pd.read_csv(metadata_path)
    metadata = metadata[metadata["Code"].ne("IGNORE")]
    file_to_codes: Dict[str, set] = {}
    for _, row in metadata.iterrows():
        fname = row["Arquivo do registro"]
        code = row["Code"]
        file_to_codes.setdefault(fname, set()).add(code)

    segments: List[AudioSegment] = []
    codes = sorted({code for codes in file_to_codes.values() for code in codes})
    for fname, codes_for_file in file_to_codes.items():
        audio_path = os.path.join(root_path, fname)
        duration = librosa.get_duration(path=audio_path)
        total_segments = max(1, int(duration // segment_seconds))
        for seg_idx in range(total_segments):
            start_sec = seg_idx * segment_seconds
            end_sec = start_sec + segment_seconds
            label = np.array([1.0 if col in codes_for_file else 0.0 for col in class_columns], dtype=np.float32)
            segment_id = f"{os.path.splitext(fname)[0]}_{start_sec:02d}_{end_sec:02d}"
            segments.append(
                AudioSegment(
                    audio_path=audio_path,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    label=label,
                    segment_id=segment_id,
                )
            )
    return segments, codes


def split_segments(
    segments: List[AudioSegment],
    validation_split: float,
    test_split: float,
    seed: int,
) -> Tuple[List[AudioSegment], List[AudioSegment], List[AudioSegment]]:
    random.Random(seed).shuffle(segments)
    total = len(segments)
    val_count = int(total * validation_split)
    test_count = int(total * test_split)
    val_segments = segments[:val_count]
    test_segments = segments[val_count:val_count + test_count]
    train_segments = segments[val_count + test_count:]
    return train_segments, val_segments, test_segments


def compute_error_rate(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> Tuple[float, float]:
    y_hat = (y_pred >= threshold).astype(np.int32)
    tp = (y_hat * y_true).sum(axis=0)
    fp = (y_hat * (1 - y_true)).sum(axis=0)
    fn = ((1 - y_hat) * y_true).sum(axis=0)
    er_per_class = []
    for k in range(y_true.shape[1]):
        s = min(tp[k] + fn[k], tp[k] + fp[k]) - tp[k]
        d = max(0.0, fn[k] - fp[k])
        i = max(0.0, fp[k] - fn[k])
        er = (s + d + i) / (tp[k] + fn[k] + 1e-10)
        er_per_class.append(er)
    macro_er = float(np.mean(er_per_class))
    total_tp = tp.sum()
    total_fp = fp.sum()
    total_fn = fn.sum()
    s = min(total_tp + total_fn, total_tp + total_fp) - total_tp
    d = max(0.0, total_fn - total_fp)
    i = max(0.0, total_fp - total_fn)
    micro_er = float((s + d + i) / (total_tp + total_fn + 1e-10))
    return micro_er, macro_er


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> Dict[str, float]:
    y_hat = (y_pred >= threshold).astype(np.int32)
    tp = (y_hat * y_true).sum(axis=0)
    fp = (y_hat * (1 - y_true)).sum(axis=0)
    fn = ((1 - y_hat) * y_true).sum(axis=0)
    precision_per = tp / (tp + fp + 1e-10)
    recall_per = tp / (tp + fn + 1e-10)
    f1_per = 2 * precision_per * recall_per / (precision_per + recall_per + 1e-10)

    macro_precision = float(np.mean(precision_per))
    macro_recall = float(np.mean(recall_per))
    macro_f1 = float(np.mean(f1_per))

    total_tp = tp.sum()
    total_fp = fp.sum()
    total_fn = fn.sum()
    micro_precision = float(total_tp / (total_tp + total_fp + 1e-10))
    micro_recall = float(total_tp / (total_tp + total_fn + 1e-10))
    micro_f1 = float(2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-10))

    micro_er, macro_er = compute_error_rate(y_true, y_pred, threshold)

    return {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_er": micro_er,
        "macro_er": macro_er,
    }


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    threshold: float,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    losses = []
    preds = []
    targets = []
    with torch.no_grad():
        for x_data, y_data, _ in loader:
            x_data = x_data.to(device)
            y_data = y_data.to(device)
            y_pred, _ = model(x_data)
            loss = loss_fn(y_pred, y_data).mean()
            losses.append(loss.item())
            preds.append(y_pred.cpu().numpy())
            targets.append(y_data.cpu().numpy())
    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(targets, axis=0)
    metrics = compute_metrics(y_true, y_pred, threshold)
    return float(np.mean(losses)), metrics


def plot_training_history(history: Dict[str, List[float]], output_path: str) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    def plot_metric(ax, train_key, test_key, title, ylabel, ylim=None):
        ax.plot(epochs, history[train_key], label="train")
        ax.plot(epochs, history[test_key], label="test")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend()
        if ylim is not None:
            ax.set_ylim(ylim)

    def common_ylim(keys: List[str]) -> Tuple[float, float]:
        values = [val for key in keys for val in history[key]]
        if not values:
            return 0.0, 1.0
        return min(values), max(values)

    plot_metric(axes[0], "train_loss", "test_loss", "Loss", "Loss")

    f1_ylim = common_ylim(["train_micro_f1", "test_micro_f1", "train_macro_f1", "test_macro_f1"])
    plot_metric(axes[1], "train_micro_f1", "test_micro_f1", "Micro F1", "F1", f1_ylim)
    plot_metric(axes[2], "train_macro_f1", "test_macro_f1", "Macro F1", "F1", f1_ylim)

    precision_ylim = common_ylim([
        "train_micro_precision",
        "test_micro_precision",
        "train_macro_precision",
        "test_macro_precision",
    ])
    plot_metric(axes[3], "train_micro_precision", "test_micro_precision", "Micro Precision", "Precision", precision_ylim)
    plot_metric(axes[4], "train_macro_precision", "test_macro_precision", "Macro Precision", "Precision", precision_ylim)

    recall_ylim = common_ylim([
        "train_micro_recall",
        "test_micro_recall",
        "train_macro_recall",
        "test_macro_recall",
    ])
    plot_metric(axes[5], "train_micro_recall", "test_micro_recall", "Micro Recall", "Recall", recall_ylim)
    plot_metric(axes[6], "train_macro_recall", "test_macro_recall", "Macro Recall", "Recall", recall_ylim)

    er_ylim = common_ylim([
        "train_micro_er",
        "test_micro_er",
        "train_macro_er",
        "test_macro_er",
    ])
    plot_metric(axes[7], "train_micro_er", "test_micro_er", "Micro ER", "ER", er_ylim)
    plot_metric(axes[8], "train_macro_er", "test_macro_er", "Macro ER", "ER", er_ylim)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def build_model(model_name: str, n_classes: int, pool_style: str, seq_len: int, n_mels: int) -> nn.Module:
    if model_name == "Baseline":
        return Baseline(pool_style=pool_style, n_classes=n_classes, seq_len=seq_len)
    if model_name == "CNN-biGRU":
        return CNNBiGRU(n_classes=n_classes, pool_style=pool_style, seq_len=seq_len, n_mels=n_mels)
    if model_name == "CNN-Transformer":
        return CNNTransformer(n_classes=n_classes, pool_style=pool_style, seq_len=seq_len, n_mels=n_mels)
    if model_name == "CDur":
        return CDur(pool_style=pool_style, n_classes=n_classes, seq_len=seq_len)
    if model_name == "TALNet":
        return TALNet(pool_style=pool_style, n_classes=n_classes, seq_len=seq_len)
    raise ValueError(f"Unsupported model name: {model_name}")


if __name__ == "__main__":
    set_seed(42)

    ANURASET_ROOT = "/ds-iml/Bioacoustics/AnuraSet/raw_data"
    FNJV_ROOT = "/ds-iml/Bioacoustics/FNJV/458"
    # /ds-iml/Bioacoustics/FNJV/578

    DATASET_TRAIN = "AnuraSet"
    DATASET_VAL = "AnuraSet"
    DATASET_TEST = "AnuraSet"

    POOLING = "mean"
    BAG_SECONDS = 10

    MODEL_NAME = "CDur"
    EPOCHS = 100
    BATCH_SIZE = 8
    NUM_WORKERS = 4
    LEARNING_RATE = 1e-3

    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.1

    TARGET_SPECIES = [
        "DENMIN",
        "LEPLAT",
        "PHYCUV",
        "SPHSUR",
        "SCIPER",
        "BOABIS",
        "BOAFAB",
        "LEPPOD",
        "PHYALB",
    ]
    # TARGET_SPECIES = ["DENMIN", "BOARAN", "DENNAN", "LEPFUS", "SCIFUS"]

    pool_map = {
        "max": "max_pool",
        "mean": "avg_pool",
        "linear": "linear_pool",
        "exp": "exp_pool",
        "att": "attention_pool",
        "auto": "auto_pool",
        "power": "power_pool",
        "hi": "hi_pool",
        "hi_plus": "hi_pool_plus",
        "hi_fixed": "hi_pool_fixed",
    }
    pool_style = pool_map[POOLING]

    sample_rate = 16000
    n_mels = 64
    n_fft = 1024
    hop_length = 664

    metadata_path = os.path.join(ANURASET_ROOT, "metadata.csv")
    anuraset_metadata = pd.read_csv(metadata_path)
    class_columns = get_class_columns(anuraset_metadata, TARGET_SPECIES)

    fnjv_segments = []
    fnjv_codes: List[str] = []
    if DATASET_TRAIN == "FNJV" or DATASET_VAL == "FNJV" or DATASET_TEST == "FNJV":
        fnjv_segments, fnjv_codes = build_fnjv_segments(FNJV_ROOT, class_columns, BAG_SECONDS)
        if TARGET_SPECIES:
            class_columns = [col for col in class_columns if col in TARGET_SPECIES]
        if DATASET_TRAIN == "FNJV":
            class_columns = [col for col in class_columns if col in fnjv_codes]

    train_segments = []
    val_segments = []
    test_segments = []

    if DATASET_TRAIN == "AnuraSet":
        train_segments = build_anuraset_segments(
            ANURASET_ROOT,
            anuraset_metadata,
            class_columns,
            BAG_SECONDS,
            subset="train",
        )
        if DATASET_VAL == "AnuraSet":
            train_segments, val_segments, _ = split_segments(
                train_segments,
                VALIDATION_SPLIT,
                0.0,
                seed=42,
            )
        if DATASET_TEST == "AnuraSet":
            test_segments = build_anuraset_segments(
                ANURASET_ROOT,
                anuraset_metadata,
                class_columns,
                BAG_SECONDS,
                subset="test",
            )

    if DATASET_TRAIN == "FNJV":
        train_segments = fnjv_segments
        val_segments = build_anuraset_segments(
            ANURASET_ROOT,
            anuraset_metadata,
            class_columns,
            BAG_SECONDS,
            subset="train",
        )
        test_segments = build_anuraset_segments(
            ANURASET_ROOT,
            anuraset_metadata,
            class_columns,
            BAG_SECONDS,
            subset="test",
        )

    if DATASET_VAL == "FNJV":
        _, fnjv_val, _ = split_segments(fnjv_segments, VALIDATION_SPLIT, TEST_SPLIT, seed=42)
        val_segments = fnjv_val

    if DATASET_TEST == "FNJV":
        _, _, fnjv_test = split_segments(fnjv_segments, VALIDATION_SPLIT, TEST_SPLIT, seed=42)
        test_segments = fnjv_test

    train_dataset = AudioSegmentDataset(train_segments, sample_rate, n_mels, n_fft, hop_length, BAG_SECONDS)
    val_dataset = AudioSegmentDataset(val_segments, sample_rate, n_mels, n_fft, hop_length, BAG_SECONDS)
    test_dataset = AudioSegmentDataset(test_segments, sample_rate, n_mels, n_fft, hop_length, BAG_SECONDS)

    num_classes = len(class_columns)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(MODEL_NAME, num_classes, pool_style, train_dataset.seq_len, n_mels).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    loss_fn = nn.BCELoss()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    output_dir = os.path.join(
        "TALNet",
        "outputs",
        f"{DATASET_TRAIN}_{POOLING}_{BAG_SECONDS}sec_{EPOCHS}epoch_{BATCH_SIZE}batch",
    )
    os.makedirs(output_dir, exist_ok=True)
    metrics_plot_path = os.path.join(output_dir, "training_metrics.png")
    best_macro_path = os.path.join(output_dir, "best_model_macro.pt")
    best_micro_path = os.path.join(output_dir, "best_model_micro.pt")

    history = {
        "train_loss": [],
        "test_loss": [],
        "train_micro_f1": [],
        "test_micro_f1": [],
        "train_macro_f1": [],
        "test_macro_f1": [],
        "train_micro_precision": [],
        "test_micro_precision": [],
        "train_macro_precision": [],
        "test_macro_precision": [],
        "train_micro_recall": [],
        "test_micro_recall": [],
        "train_macro_recall": [],
        "test_macro_recall": [],
        "train_micro_er": [],
        "test_micro_er": [],
        "train_macro_er": [],
        "test_macro_er": [],
    }

    best_macro_f1 = -1.0
    best_micro_f1 = -1.0

    threshold = 0.5

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        train_preds = []
        train_targets = []
        for x_data, y_data, _ in train_loader:
            x_data = x_data.to(device)
            y_data = y_data.to(device)
            y_pred, _ = model(x_data)
            loss = loss_fn(y_pred, y_data).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_preds.append(y_pred.detach().cpu().numpy())
            train_targets.append(y_data.detach().cpu().numpy())

        scheduler.step()
        train_loss = float(np.mean(train_losses))
        train_metrics = compute_metrics(
            np.concatenate(train_targets, axis=0),
            np.concatenate(train_preds, axis=0),
            threshold,
        )

        test_loss, test_metrics = evaluate_model(model, val_loader, loss_fn, device, threshold)

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        for key in [
            "micro_f1",
            "macro_f1",
            "micro_precision",
            "macro_precision",
            "micro_recall",
            "macro_recall",
            "micro_er",
            "macro_er",
        ]:
            history[f"train_{key}"].append(train_metrics[key])
            history[f"test_{key}"].append(test_metrics[key])

        plot_training_history(history, metrics_plot_path)

        if test_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = test_metrics["macro_f1"]
            torch.save(model.state_dict(), best_macro_path)

        if test_metrics["micro_f1"] > best_micro_f1:
            best_micro_f1 = test_metrics["micro_f1"]
            torch.save(model.state_dict(), best_micro_path)

        print(
            "Epoch {epoch}: train loss={train_loss:.4f}, micro-P={train_micro_p:.4f}, "
            "micro-R={train_micro_r:.4f}, micro-F1={train_micro_f1:.4f}, "
            "macro-P={train_macro_p:.4f}, macro-R={train_macro_r:.4f}, "
            "macro-F1={train_macro_f1:.4f}, ER={train_er:.4f}".format(
                epoch=epoch,
                train_loss=train_loss,
                train_micro_p=train_metrics["micro_precision"],
                train_micro_r=train_metrics["micro_recall"],
                train_micro_f1=train_metrics["micro_f1"],
                train_macro_p=train_metrics["macro_precision"],
                train_macro_r=train_metrics["macro_recall"],
                train_macro_f1=train_metrics["macro_f1"],
                train_er=train_metrics["macro_er"],
            )
        )
        print(
            "           val loss={val_loss:.4f}, micro-P={val_micro_p:.4f}, "
            "micro-R={val_micro_r:.4f}, micro-F1={val_micro_f1:.4f}, "
            "macro-P={val_macro_p:.4f}, macro-R={val_macro_r:.4f}, "
            "macro-F1={val_macro_f1:.4f}, ER={val_er:.4f}".format(
                val_loss=test_loss,
                val_micro_p=test_metrics["micro_precision"],
                val_micro_r=test_metrics["micro_recall"],
                val_micro_f1=test_metrics["micro_f1"],
                val_macro_p=test_metrics["macro_precision"],
                val_macro_r=test_metrics["macro_recall"],
                val_macro_f1=test_metrics["macro_f1"],
                val_er=test_metrics["macro_er"],
            )
        )

    best_macro_model = build_model(MODEL_NAME, num_classes, pool_style, train_dataset.seq_len, n_mels).to(device)
    best_macro_model.load_state_dict(torch.load(best_macro_path, map_location=device))
    macro_test_loss, macro_test_metrics = evaluate_model(
        best_macro_model, test_loader, loss_fn, device, threshold
    )

    best_micro_model = build_model(MODEL_NAME, num_classes, pool_style, train_dataset.seq_len, n_mels).to(device)
    best_micro_model.load_state_dict(torch.load(best_micro_path, map_location=device))
    micro_test_loss, micro_test_metrics = evaluate_model(
        best_micro_model, test_loader, loss_fn, device, threshold
    )

    print("Best Macro Model Test Results")
    print(f"Loss: {macro_test_loss:.4f}")
    print(macro_test_metrics)

    print("Best Micro Model Test Results")
    print(f"Loss: {micro_test_loss:.4f}")
    print(micro_test_metrics)

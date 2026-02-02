import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import config
from MainClasses.Models import Baseline, CDur, TALNet
from MainClasses.MILPooling import MILPooling
from MainClasses.loc_vad import activity_detection


@dataclass
class AudioSegment:
    audio_path: str
    start_sec: float
    end_sec: float
    label: np.ndarray
    segment_id: str
    duration_sec: Optional[float] = None


class AudioSegmentDataset(Dataset):
    def __init__(
        self,
        segments: List[AudioSegment],
        sample_rate: int,
        n_mels: int,
        n_fft: int,
        hop_length: int,
        segment_seconds: int,
        fixed_frames: Optional[int] = None,
    ):
        self.segments = segments
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.segment_seconds = segment_seconds
        if fixed_frames is not None:
            self.seq_len = fixed_frames
        else:
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
        return logmelspec.T

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, int, str, float, float]:
        seg = self.segments[idx]
        audio = self._load_segment(seg.audio_path, seg.start_sec, seg.end_sec)
        feats = self._extract_logmel(audio)
        frames_len = feats.shape[0]
        return (
            torch.from_numpy(feats).float(),
            torch.from_numpy(seg.label).float(),
            seg.segment_id,
            frames_len,
            seg.audio_path,
            seg.start_sec,
            seg.end_sec,
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

    def forward(
        self, inputs: torch.Tensor, upsample: bool = False, mask: Optional[torch.Tensor] = None, **_kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = inputs.unsqueeze(1)
        x = self.features(x)
        x = x.transpose(1, 2).flatten(2)
        x, _ = self.gru(x)
        y_frames = torch.sigmoid(self.out(x)).clamp(1e-7, 1.0)
        y_clip = pool_with_mask(self.pool, y_frames, mask)
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

    def forward(
        self, inputs: torch.Tensor, upsample: bool = False, mask: Optional[torch.Tensor] = None, **_kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = inputs.unsqueeze(1)
        x = self.cnn(x)
        x = x.transpose(1, 2).flatten(2)
        x = self.transformer(x)
        y_frames = torch.sigmoid(self.out(x)).clamp(1e-7, 1.0)
        y_clip = pool_with_mask(self.pool, y_frames, mask)
        return y_clip, y_frames


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_frames(duration_sec: float, sample_rate: int, n_fft: int, hop_length: int) -> int:
    frames = int(np.floor((duration_sec * sample_rate - n_fft) / hop_length) + 1)
    return max(frames, 1)


def pool_with_mask(pool, y_frames, mask=None):
    if mask is None:
        return pool(y_frames)
    try:
        return pool(y_frames, mask)
    except TypeError:
        return pool(y_frames)


def pool_requires_fixed_seq_len(pool_style: str) -> bool:
    return pool_style in {"attention_pool", "hi_pool", "hi_pool_plus", "hi_pool_fixed"}


def collate_default(batch):
    feats, labels, ids, _lengths, paths, starts, ends = zip(*batch)
    return torch.stack(feats), torch.stack(labels), list(ids), None, list(paths), list(starts), list(ends)


def collate_full_bag_pad(batch, fixed_frames: int, pad_mode: str):
    feats, labels, ids, lengths, paths, starts, ends = zip(*batch)
    padded = []
    masks = []
    for feat, length in zip(feats, lengths):
        if length >= fixed_frames:
            feat_pad = feat[:fixed_frames]
            mask = torch.ones(fixed_frames, dtype=torch.float32)
        else:
            if pad_mode == "repeat":
                reps = fixed_frames // length
                remainder = fixed_frames % length
                tiles = [feat] * reps
                if remainder > 0:
                    tiles.append(feat[:remainder])
                feat_pad = torch.cat(tiles, dim=0)
                mask = None
            else:
                pad_len = fixed_frames - length
                pad = torch.zeros((pad_len, feat.shape[1]), dtype=feat.dtype)
                feat_pad = torch.cat([feat, pad], dim=0)
                mask = torch.cat([torch.ones(length), torch.zeros(pad_len)])
        padded.append(feat_pad)
        masks.append(mask)

    x = torch.stack(padded)
    y = torch.stack(labels)
    if pad_mode == "silence":
        mask = torch.stack(masks)
    else:
        mask = None
    return x, y, list(ids), mask, list(paths), list(starts), list(ends)


def get_max_duration(segments: List[AudioSegment]) -> float:
    if not segments:
        return 0.0
    return max(
        seg.duration_sec if seg.duration_sec is not None else (seg.end_sec - seg.start_sec)
        for seg in segments
    )


def load_strong_labels(csv_path: str, allowed_levels: List[str]) -> Dict[str, List[Dict[str, object]]]:
    df = pd.read_csv(csv_path)
    if allowed_levels:
        df = df[df["level"].isin(allowed_levels)]
    df["file_stem"] = df["file_name"].str.replace(".txt", "", regex=False)
    events: Dict[str, List[Dict[str, object]]] = {}
    for _, row in df.iterrows():
        events.setdefault(row["file_stem"], []).append(
            {
                "start": float(row["start_second"]),
                "end": float(row["end_second"]),
                "label": str(row["label"]),
            }
        )
    return events


def strong_label_vector(
    events: List[Dict[str, object]],
    class_columns: List[str],
    start_sec: float,
    end_sec: float,
) -> np.ndarray:
    label_vec = np.zeros(len(class_columns), dtype=np.float32)
    if not events:
        return label_vec
    for event in events:
        if event["end"] > start_sec and event["start"] < end_sec:
            if event["label"] in class_columns:
                idx = class_columns.index(event["label"])
                label_vec[idx] = 1.0
    return label_vec


def anuraset_file_stem(audio_path: str) -> str:
    return os.path.splitext(os.path.basename(audio_path))[0]
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
    overlap_bags: bool,
    hop_seconds: int,
    full_bag: bool,
    subset: str,
) -> List[AudioSegment]:
    subset_meta = metadata[metadata["subset"] == subset]
    segments: List[AudioSegment] = []
    grouped = subset_meta.groupby(["site", "fname"])
    for (site, fname), group in grouped:
        audio_path = os.path.join(root_path, site, f"{fname}.wav")
        duration = 60
        if full_bag:
            label = group[class_columns].max().values.astype(np.float32)
            segments.append(
                AudioSegment(
                    audio_path=audio_path,
                    start_sec=0,
                    end_sec=duration,
                    label=label,
                    segment_id=f"{fname}_full",
                    duration_sec=duration,
                )
            )
        else:
            step = hop_seconds if overlap_bags else segment_seconds
            start_sec = 0
            while start_sec + segment_seconds <= duration:
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
                        duration_sec=segment_seconds,
                    )
                )
                start_sec += step
    return segments


def build_fnjv_segments(
    root_path: str,
    class_columns: List[str],
    segment_seconds: int,
    overlap_bags: bool,
    hop_seconds: int,
    full_bag: bool,
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
        label = np.array([1.0 if col in codes_for_file else 0.0 for col in class_columns], dtype=np.float32)
        if full_bag:
            segments.append(
                AudioSegment(
                    audio_path=audio_path,
                    start_sec=0,
                    end_sec=duration,
                    label=label,
                    segment_id=f"{os.path.splitext(fname)[0]}_full",
                    duration_sec=duration,
                )
            )
        else:
            step = hop_seconds if overlap_bags else segment_seconds
            start_sec = 0
            if duration < segment_seconds:
                end_sec = segment_seconds
                segment_id = f"{os.path.splitext(fname)[0]}_{start_sec:02d}_{end_sec:02d}"
                segments.append(
                    AudioSegment(
                        audio_path=audio_path,
                        start_sec=start_sec,
                        end_sec=end_sec,
                        label=label,
                        segment_id=segment_id,
                        duration_sec=segment_seconds,
                    )
                )
            else:
                while start_sec + segment_seconds <= duration:
                    end_sec = start_sec + segment_seconds
                    segment_id = f"{os.path.splitext(fname)[0]}_{start_sec:02d}_{end_sec:02d}"
                    segments.append(
                        AudioSegment(
                            audio_path=audio_path,
                            start_sec=start_sec,
                            end_sec=end_sec,
                            label=label,
                            segment_id=segment_id,
                            duration_sec=segment_seconds,
                        )
                    )
                    start_sec += step
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


def compute_class_f1(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> np.ndarray:
    y_hat = (y_pred >= threshold).astype(np.int32)
    tp = (y_hat * y_true).sum(axis=0)
    fp = (y_hat * (1 - y_true)).sum(axis=0)
    fn = ((1 - y_hat) * y_true).sum(axis=0)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    return f1


def localization_frame_labels(
    events: List[Dict[str, object]],
    class_columns: List[str],
    start_sec: float,
    end_sec: float,
    frame_count: int,
) -> np.ndarray:
    labels = np.zeros((frame_count, len(class_columns)), dtype=np.float32)
    if not events or frame_count <= 0:
        return labels
    duration = max(end_sec - start_sec, 1e-6)
    frames_per_sec = frame_count / duration
    for event in events:
        if event["end"] <= start_sec or event["start"] >= end_sec:
            continue
        if event["label"] not in class_columns:
            continue
        class_idx = class_columns.index(event["label"])
        onset = max(event["start"], start_sec) - start_sec
        offset = min(event["end"], end_sec) - start_sec
        start_frame = int(np.floor(onset * frames_per_sec))
        end_frame = int(np.ceil(offset * frames_per_sec))
        start_frame = max(0, min(frame_count, start_frame))
        end_frame = max(0, min(frame_count, end_frame))
        if end_frame > start_frame:
            labels[start_frame:end_frame, class_idx] = 1.0
    return labels


def evaluate_localization(
    model: nn.Module,
    loader: DataLoader,
    strong_events: Dict[str, List[Dict[str, object]]],
    class_columns: List[str],
    threshold: float,
    device: torch.device,
) -> Tuple[Dict[str, float], np.ndarray]:
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for x_data, _y_data, _ids, mask, paths, starts, ends in loader:
            x_data = x_data.to(device)
            if mask is not None:
                mask = mask.to(device)
                _y_pred, y_frames = model(x_data, mask=mask)
            else:
                _y_pred, y_frames = model(x_data)
            y_frames = y_frames.cpu().numpy()
            mask_np = mask.cpu().numpy() if mask is not None else None
            for i in range(len(paths)):
                file_stem = anuraset_file_stem(paths[i])
                events = strong_events.get(file_stem, [])
                frames_len = y_frames[i].shape[0]
                if mask_np is not None:
                    frames_len = int(mask_np[i].sum())
                    frames_len = max(frames_len, 1)
                    frame_pred = y_frames[i][:frames_len]
                else:
                    frame_pred = y_frames[i]
                frame_true = localization_frame_labels(
                    events,
                    class_columns,
                    float(starts[i]),
                    float(ends[i]),
                    frames_len,
                )
                preds.append(frame_pred)
                targets.append(frame_true)
    if not preds:
        return {
            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "micro_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
        }, np.zeros(len(class_columns), dtype=np.float32)
    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(targets, axis=0)
    metrics = compute_metrics(y_true, y_pred, threshold)
    class_f1 = compute_class_f1(y_true, y_pred, threshold)
    return metrics, class_f1


def binarize_with_activity_detection(
    frame_pred: np.ndarray,
    class_columns: List[str],
    thresholds: Dict[str, float],
) -> np.ndarray:
    n_frames, n_classes = frame_pred.shape
    binary = np.zeros((n_frames, n_classes), dtype=np.float32)
    for k in range(n_classes):
        pairs = activity_detection(
            x=frame_pred[:, k],
            thres=thresholds["loc_threshold_high"],
            low_thres=thresholds["loc_threshold_low"],
            n_smooth=thresholds["smooth"],
            n_salt=thresholds["smooth"],
        )
        for onset, offset in pairs:
            onset = max(0, min(n_frames, int(onset)))
            offset = max(0, min(n_frames, int(offset)))
            if offset > onset:
                binary[onset:offset, k] = 1.0
    return binary


def visualize_predictions(
    model: nn.Module,
    loader: DataLoader,
    strong_events: Dict[str, List[Dict[str, object]]],
    class_columns: List[str],
    thresholds: Dict[str, float],
    output_root: str,
    prefix: str,
    max_per_class: int = 5,
) -> None:
    model.eval()
    selections = {name: {"correct": [], "wrong": []} for name in class_columns}
    with torch.no_grad():
        for x_data, y_data, _ids, mask, paths, starts, ends in loader:
            device = next(model.parameters()).device
            x_data = x_data.to(device)
            if mask is not None:
                mask = mask.to(x_data.device)
                clip_out, frame_out = model(x_data, mask=mask)
            else:
                clip_out, frame_out = model(x_data)
            clip_out = clip_out.cpu().numpy()
            frame_out = frame_out.cpu().numpy()
            mask_np = mask.cpu().numpy() if mask is not None else None
            y_true = y_data.cpu().numpy()
            for i in range(len(paths)):
                file_stem = anuraset_file_stem(paths[i])
                events = strong_events.get(file_stem, [])
                frame_pred = frame_out[i]
                if mask_np is not None:
                    valid_len = int(mask_np[i].sum())
                    valid_len = max(valid_len, 1)
                    frame_pred = frame_pred[:valid_len]
                frame_true = localization_frame_labels(
                    events,
                    class_columns,
                    float(starts[i]),
                    float(ends[i]),
                    frame_pred.shape[0],
                )
                for class_idx, class_name in enumerate(class_columns):
                    if len(selections[class_name]["correct"]) >= max_per_class and len(
                        selections[class_name]["wrong"]
                    ) >= max_per_class:
                        continue
                    pred = clip_out[i, class_idx] >= thresholds["tag_threshold"]
                    target = y_true[i, class_idx] >= 0.5
                    bucket = "correct" if pred == target else "wrong"
                    if len(selections[class_name][bucket]) >= max_per_class:
                        continue
                    selections[class_name][bucket].append(
                        {
                            "class_idx": class_idx,
                            "class_name": class_name,
                            "audio_path": paths[i],
                            "start": float(starts[i]),
                            "end": float(ends[i]),
                            "clip_out": clip_out[i],
                            "frame_pred": frame_pred,
                            "frame_true": frame_true,
                            "target": y_true[i],
                        }
                    )

    for class_name, buckets in selections.items():
        for bucket_name, samples in buckets.items():
            out_dir = os.path.join(output_root, class_name, bucket_name)
            os.makedirs(out_dir, exist_ok=True)
            for sample in samples:
                frame_pred = sample["frame_pred"]
                frame_true = sample["frame_true"]
                clip_out = sample["clip_out"]
                target = sample["target"]
                active_classes = np.where(target >= 0.5)[0].tolist()
                if not active_classes:
                    active_classes = [sample["class_idx"]]
                active_names = [class_columns[idx] for idx in active_classes]
                clip_len = max(sample["end"] - sample["start"], 1e-6)
                seq_len = frame_pred.shape[0]
                t_sec = np.arange(seq_len) * (clip_len / seq_len)

                fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
                gt_data = frame_true[:, active_classes].T
                axes[0].imshow(gt_data, aspect="auto", origin="lower", extent=[0, clip_len, 0, len(active_classes)])
                axes[0].set_yticks(np.arange(len(active_classes)) + 0.5)
                axes[0].set_yticklabels(active_names)
                axes[0].set_title("Ground Truth (strong labels)")

                for idx in active_classes:
                    axes[1].plot(clip_out[idx] * np.ones_like(t_sec), label=class_columns[idx])
                axes[1].axhline(thresholds["tag_threshold"], color="red", linestyle="--", linewidth=1)
                axes[1].set_title("Clip-wise predictions")
                axes[1].legend(loc="upper right")

                pred_data = frame_pred[:, active_classes].T
                axes[2].imshow(pred_data, aspect="auto", origin="lower", extent=[0, clip_len, 0, len(active_classes)])
                axes[2].set_yticks(np.arange(len(active_classes)) + 0.5)
                axes[2].set_yticklabels(active_names)
                axes[2].set_title("Frame-wise predictions")

                binary = binarize_with_activity_detection(frame_pred, class_columns, thresholds)
                binary_data = binary[:, active_classes].T
                axes[3].imshow(binary_data, aspect="auto", origin="lower", extent=[0, clip_len, 0, len(active_classes)])
                axes[3].set_yticks(np.arange(len(active_classes)) + 0.5)
                axes[3].set_yticklabels(active_names)
                axes[3].set_title("Frame-wise predictions (VAD)")
                axes[3].set_xlabel("Time (s)")

                fig.tight_layout()
                audio_name = os.path.basename(sample["audio_path"]).replace(".wav", "")
                fig_path = os.path.join(out_dir, f"{audio_name}_{prefix}_{class_name}.png")
                fig.savefig(fig_path)
                plt.close(fig)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    threshold: float,
    return_arrays: bool = False,
) -> Tuple[float, Dict[str, float], Optional[np.ndarray], Optional[np.ndarray]]:
    model.eval()
    losses = []
    preds = []
    targets = []
    with torch.no_grad():
        for x_data, y_data, _, mask, _paths, _starts, _ends in loader:
            x_data = x_data.to(device)
            y_data = y_data.to(device)
            if mask is not None:
                mask = mask.to(device)
                y_pred, _ = model(x_data, mask=mask)
            else:
                y_pred, _ = model(x_data)
            loss = loss_fn(y_pred, y_data).mean()
            losses.append(loss.item())
            preds.append(y_pred.cpu().numpy())
            targets.append(y_data.cpu().numpy())
    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(targets, axis=0)
    metrics = compute_metrics(y_true, y_pred, threshold)
    if return_arrays:
        return float(np.mean(losses)), metrics, y_true, y_pred
    return float(np.mean(losses)), metrics, None, None


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
    set_seed(config.SEED)

    ANURASET_ROOT = config.ANURASET_ROOT
    FNJV_ROOT = config.FNJV_ROOT

    DATASET_TRAIN = config.DATASET_TRAIN
    DATASET_VAL = config.DATASET_VAL
    DATASET_TEST = config.DATASET_TEST

    POOLING = config.POOLING
    BAG_SECONDS = config.BAG_SECONDS
    FULL_BAG_METHOD = config.FULL_BAG_METHOD
    PAD_MODE = config.PAD_MODE

    MODEL_NAME = config.MODEL_NAME
    EPOCHS = config.EPOCHS
    BATCH_SIZE = config.BATCH_SIZE
    NUM_WORKERS = config.NUM_WORKERS
    LEARNING_RATE = config.LEARNING_RATE

    VALIDATION_SPLIT = config.VALIDATION_SPLIT
    TEST_SPLIT = config.TEST_SPLIT
    APPLY_VALIDATION_SPLIT = config.APPLY_VALIDATION_SPLIT
    APPLY_TEST_SPLIT = config.APPLY_TEST_SPLIT

    TARGET_SPECIES = config.TARGET_SPECIES
    STRONG_LABELS_458 = config.STRONG_LABELS_458
    STRONG_LABELS_578 = config.STRONG_LABELS_578
    STRONG_LABEL_LEVELS = config.STRONG_LABEL_LEVELS
    ANURASET_EVAL = config.ANURASET_EVAL

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

    sample_rate = config.sample_rate
    n_mels = config.n_mels
    n_fft = config.n_fft
    hop_length = config.hop_length
    OVERLAP_BAGS = config.OVERLAP_BAGS
    HOP_SECONDS = config.HOP_SECONDS
    full_bag = BAG_SECONDS == "full"

    if full_bag and FULL_BAG_METHOD == "batch":
        BATCH_SIZE = 1

    if full_bag and FULL_BAG_METHOD == "batch" and pool_requires_fixed_seq_len(pool_style):
        print(">>> [config] Full-bag batch mode: switching pooling to avg_pool for variable length input.")
        pool_style = "avg_pool"

    metadata_path = os.path.join(ANURASET_ROOT, "metadata.csv")
    anuraset_metadata = pd.read_csv(metadata_path)
    class_columns = get_class_columns(anuraset_metadata, TARGET_SPECIES)

    fnjv_segments = []
    fnjv_codes: List[str] = []
    if DATASET_TRAIN == "FNJV" or DATASET_VAL == "FNJV" or DATASET_TEST == "FNJV":
        fnjv_segments, fnjv_codes = build_fnjv_segments(
            FNJV_ROOT,
            class_columns,
            BAG_SECONDS,
            OVERLAP_BAGS,
            HOP_SECONDS,
            full_bag,
        )
        if TARGET_SPECIES:
            class_columns = [col for col in class_columns if col in TARGET_SPECIES]
        if DATASET_TRAIN == "FNJV":
            class_columns = [col for col in class_columns if col in fnjv_codes]

    train_segments = []
    val_segments = []
    test_segments = []

    anura_train_segments = build_anuraset_segments(
        ANURASET_ROOT,
        anuraset_metadata,
        class_columns,
        BAG_SECONDS,
        OVERLAP_BAGS,
        HOP_SECONDS,
        full_bag,
        subset="train",
    )
    anura_test_segments = build_anuraset_segments(
        ANURASET_ROOT,
        anuraset_metadata,
        class_columns,
        BAG_SECONDS,
        OVERLAP_BAGS,
        HOP_SECONDS,
        full_bag,
        subset="test",
    )

    if APPLY_VALIDATION_SPLIT:
        anura_train_segments, anura_val_segments, _ = split_segments(
            anura_train_segments,
            VALIDATION_SPLIT,
            0.0,
            seed=config.SEED,
        )
    else:
        anura_val_segments = anura_train_segments

    if APPLY_TEST_SPLIT:
        _, _, anura_test_segments = split_segments(
            anura_test_segments,
            0.0,
            TEST_SPLIT,
            seed=config.SEED,
        )

    fnjv_id = "578" if "578" in FNJV_ROOT else "458"
    strong_labels_path = STRONG_LABELS_578 if fnjv_id == "578" else STRONG_LABELS_458
    strong_events = None
    if os.path.exists(strong_labels_path):
        strong_events = load_strong_labels(strong_labels_path, STRONG_LABEL_LEVELS)

    if APPLY_VALIDATION_SPLIT or APPLY_TEST_SPLIT:
        fnjv_train_segments, fnjv_val_segments, fnjv_test_segments = split_segments(
            fnjv_segments,
            VALIDATION_SPLIT if APPLY_VALIDATION_SPLIT else 0.0,
            TEST_SPLIT if APPLY_TEST_SPLIT else 0.0,
            seed=config.SEED,
        )
    else:
        fnjv_train_segments = fnjv_segments
        fnjv_val_segments = fnjv_segments
        fnjv_test_segments = fnjv_segments

    if DATASET_TRAIN == "AnuraSet":
        train_segments = anura_train_segments
    else:
        train_segments = fnjv_train_segments

    if DATASET_VAL == "AnuraSet":
        val_segments = anura_val_segments
    else:
        val_segments = fnjv_val_segments

    if DATASET_TEST == "AnuraSet":
        test_segments = anura_test_segments
    else:
        test_segments = fnjv_test_segments

    if strong_events is not None:
        if DATASET_VAL == "AnuraSet":
            updated_segments = []
            for seg in val_segments:
                stem = anuraset_file_stem(seg.audio_path)
                events = strong_events.get(stem, [])
                label = strong_label_vector(events, class_columns, seg.start_sec, seg.end_sec)
                updated_segments.append(
                    AudioSegment(
                        audio_path=seg.audio_path,
                        start_sec=seg.start_sec,
                        end_sec=seg.end_sec,
                        label=label,
                        segment_id=seg.segment_id,
                        duration_sec=seg.duration_sec,
                    )
                )
            val_segments = updated_segments
        if DATASET_TEST == "AnuraSet":
            updated_segments = []
            for seg in test_segments:
                stem = anuraset_file_stem(seg.audio_path)
                events = strong_events.get(stem, [])
                label = strong_label_vector(events, class_columns, seg.start_sec, seg.end_sec)
                updated_segments.append(
                    AudioSegment(
                        audio_path=seg.audio_path,
                        start_sec=seg.start_sec,
                        end_sec=seg.end_sec,
                        label=label,
                        segment_id=seg.segment_id,
                        duration_sec=seg.duration_sec,
                    )
                )
            test_segments = updated_segments

    if full_bag and FULL_BAG_METHOD == "pad":
        max_duration_sec = max(
            get_max_duration(train_segments + val_segments + test_segments),
            0.0,
        )
        fixed_frames = (
            compute_frames(max_duration_sec, sample_rate, n_fft, hop_length)
            if max_duration_sec > 0
            else None
        )
        if fixed_frames is None:
            raise ValueError("Full-bag pad mode requires at least one segment to compute max length.")
    else:
        fixed_frames = None

    segment_seconds = BAG_SECONDS if isinstance(BAG_SECONDS, int) else 1

    train_dataset = AudioSegmentDataset(
        train_segments,
        sample_rate,
        n_mels,
        n_fft,
        hop_length,
        segment_seconds,
        fixed_frames=fixed_frames,
    )
    val_dataset = AudioSegmentDataset(
        val_segments,
        sample_rate,
        n_mels,
        n_fft,
        hop_length,
        segment_seconds,
        fixed_frames=fixed_frames,
    )
    test_dataset = AudioSegmentDataset(
        test_segments,
        sample_rate,
        n_mels,
        n_fft,
        hop_length,
        segment_seconds,
        fixed_frames=fixed_frames,
    )

    num_classes = len(class_columns)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_seq_len = fixed_frames if fixed_frames is not None else train_dataset.seq_len
    model = build_model(MODEL_NAME, num_classes, pool_style, model_seq_len, n_mels).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    loss_fn = nn.BCELoss()

    if full_bag and FULL_BAG_METHOD == "pad":
        collate_fn = lambda batch: collate_full_bag_pad(batch, fixed_frames, PAD_MODE)
    else:
        collate_fn = collate_default

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
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

    threshold = config.threshold

    print(">>> [config] SEED=", config.SEED)
    print(">>> [config] ANURASET_ROOT=", ANURASET_ROOT)
    print(">>> [config] FNJV_ROOT=", FNJV_ROOT)
    print(">>> [config] DATASET_TRAIN=", DATASET_TRAIN)
    print(">>> [config] DATASET_VAL=", DATASET_VAL)
    print(">>> [config] DATASET_TEST=", DATASET_TEST)
    print(">>> [config] POOLING=", POOLING)
    print(">>> [config] BAG_SECONDS=", BAG_SECONDS)
    print(">>> [config] FULL_BAG_METHOD=", FULL_BAG_METHOD)
    print(">>> [config] PAD_MODE=", PAD_MODE)
    print(">>> [config] MODEL_NAME=", MODEL_NAME)
    print(">>> [config] EPOCHS=", EPOCHS)
    print(">>> [config] BATCH_SIZE=", BATCH_SIZE)
    print(">>> [config] NUM_WORKERS=", NUM_WORKERS)
    print(">>> [config] LEARNING_RATE=", LEARNING_RATE)
    print(">>> [config] VALIDATION_SPLIT=", VALIDATION_SPLIT)
    print(">>> [config] TEST_SPLIT=", TEST_SPLIT)
    print(">>> [config] APPLY_VALIDATION_SPLIT=", APPLY_VALIDATION_SPLIT)
    print(">>> [config] APPLY_TEST_SPLIT=", APPLY_TEST_SPLIT)
    print(">>> [config] TARGET_SPECIES=", TARGET_SPECIES)
    print(">>> [config] STRONG_LABEL_LEVELS=", STRONG_LABEL_LEVELS)
    print(">>> [config] ANURASET_EVAL=", ANURASET_EVAL)
    print(">>> [config] sample_rate=", sample_rate)
    print(">>> [config] n_mels=", n_mels)
    print(">>> [config] n_fft=", n_fft)
    print(">>> [config] hop_length=", hop_length)
    print(">>> [config] threshold=", threshold)
    print(">>> [config] OVERLAP_BAGS=", OVERLAP_BAGS)
    print(">>> [config] HOP_SECONDS=", HOP_SECONDS)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        train_preds = []
        train_targets = []
        for x_data, y_data, _, mask, _paths, _starts, _ends in train_loader:
            x_data = x_data.to(device)
            y_data = y_data.to(device)
            if mask is not None:
                mask = mask.to(device)
                y_pred, _ = model(x_data, mask=mask)
            else:
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

        test_loss, test_metrics, _, _ = evaluate_model(model, val_loader, loss_fn, device, threshold)

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

    best_macro_model = build_model(MODEL_NAME, num_classes, pool_style, model_seq_len, n_mels).to(device)
    best_macro_model.load_state_dict(torch.load(best_macro_path, map_location=device))
    macro_test_loss, macro_test_metrics, macro_y_true, macro_y_pred = evaluate_model(
        best_macro_model, test_loader, loss_fn, device, threshold, return_arrays=True
    )
    macro_class_f1 = compute_class_f1(macro_y_true, macro_y_pred, threshold)

    best_micro_model = build_model(MODEL_NAME, num_classes, pool_style, model_seq_len, n_mels).to(device)
    best_micro_model.load_state_dict(torch.load(best_micro_path, map_location=device))
    micro_test_loss, micro_test_metrics, micro_y_true, micro_y_pred = evaluate_model(
        best_micro_model, test_loader, loss_fn, device, threshold, return_arrays=True
    )
    micro_class_f1 = compute_class_f1(micro_y_true, micro_y_pred, threshold)

    print("Best Macro Model Test Results")
    print(f"Loss: {macro_test_loss:.4f}")
    print(macro_test_metrics)
    print("Macro Model Class F1:")
    for class_name, f1_value in zip(class_columns, macro_class_f1):
        print(f"  {class_name}: {f1_value:.4f}")

    print("Best Micro Model Test Results")
    print(f"Loss: {micro_test_loss:.4f}")
    print(micro_test_metrics)
    print("Micro Model Class F1:")
    for class_name, f1_value in zip(class_columns, micro_class_f1):
        print(f"  {class_name}: {f1_value:.4f}")

    if strong_events is not None and DATASET_TEST == "AnuraSet":
        loc_metrics_macro, loc_class_f1_macro = evaluate_localization(
            best_macro_model,
            test_loader,
            strong_events,
            class_columns,
            threshold,
            device,
        )
        print("Macro Model Localization Metrics")
        print(loc_metrics_macro)
        print("Macro Model Localization Class F1:")
        for class_name, f1_value in zip(class_columns, loc_class_f1_macro):
            print(f"  {class_name}: {f1_value:.4f}")

        loc_metrics_micro, loc_class_f1_micro = evaluate_localization(
            best_micro_model,
            test_loader,
            strong_events,
            class_columns,
            threshold,
            device,
        )
        print("Micro Model Localization Metrics")
        print(loc_metrics_micro)
        print("Micro Model Localization Class F1:")
        for class_name, f1_value in zip(class_columns, loc_class_f1_micro):
            print(f"  {class_name}: {f1_value:.4f}")

        visualize_predictions(
            best_macro_model,
            test_loader,
            strong_events,
            class_columns,
            ANURASET_EVAL,
            output_root="results/AnuraSet/viz",
            prefix="macro",
        )
        visualize_predictions(
            best_micro_model,
            test_loader,
            strong_events,
            class_columns,
            ANURASET_EVAL,
            output_root="results/AnuraSet/viz",
            prefix="micro",
        )

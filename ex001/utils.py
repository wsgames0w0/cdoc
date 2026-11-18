"""論文再現実験（ex001）用のデータ読み込み・学習・評価・可視化ユーティリティ．"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


CLASS_NAMES = ["Nobody cheats", "White cheats", "Black cheats"]
CLASS_NAMES_JA = ["チートなし (0)", "白番チート (1)", "黒番チート (2)"]


def configure_japanese_font() -> None:
    """Matplotlib で日本語フォントが正しく描画されるように設定する．"""
    preferred_fonts = ["Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP", "IPAexGothic"]
    available = {f.name for f in fm.fontManager.ttflist}
    for font_name in preferred_fonts:
        if font_name in available:
            plt.rcParams["font.family"] = font_name
            break
    plt.rcParams["axes.unicode_minus"] = False


def set_seed(seed: int = 42) -> None:
    """再現性確保のため乱数シードを固定する．"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ChessCheatDataset(Dataset):
    """盤面テンソルと解析特徴量テンソルを保持する PyTorch Dataset．"""

    def __init__(
        self,
        x_board: np.ndarray,
        x_analyzed: np.ndarray,
        y: np.ndarray,
    ) -> None:
        self.x_board = x_board
        self.x_analyzed = x_analyzed
        self.y = y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        board = torch.from_numpy(self.x_board[idx].astype(np.float32, copy=False))
        analyzed = torch.from_numpy(self.x_analyzed[idx].astype(np.float32, copy=False))
        label = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return board, analyzed, label


def load_and_split_dataset(
    npz_path: str | Path,
    seed: int = 42,
    use_extended_analyzed: bool = False,
) -> dict[str, np.ndarray]:
    """データセットを読み込み，論文と同じ 80% Train (うち 10% Val) / 20% Test に分割する．"""
    data = np.load(npz_path)
    x_board = data["X_board"]
    if use_extended_analyzed and "X_analyzed_ext" in data:
        x_analyzed = data["X_analyzed_ext"]
    else:
        x_analyzed = data["X_analyzed"]
    y = data["y"]

    indices = np.arange(len(y))
    # 論文の「randomly shuffled and split into 80% training data and 20% testing data」に準拠
    # クラス比率を維持するため stratify=y を使用（各クラスが均等なので実質同じ比率）
    train_full_idx, test_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=seed,
        shuffle=True,
        stratify=y,
    )
    # 学習データ 4,000 件のうち 10% (400 件) を Validation，残り 3,600 件を Train に使用
    train_idx, val_idx = train_test_split(
        train_full_idx,
        test_size=0.10,
        random_state=seed,
        shuffle=True,
        stratify=y[train_full_idx],
    )

    return {
        "X_board_train": x_board[train_idx],
        "X_analyzed_train": x_analyzed[train_idx],
        "y_train": y[train_idx],
        "X_board_val": x_board[val_idx],
        "X_analyzed_val": x_analyzed[val_idx],
        "y_val": y[val_idx],
        "X_board_test": x_board[test_idx],
        "X_analyzed_test": x_analyzed[test_idx],
        "y_test": y[test_idx],
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }


def create_dataloaders(
    splits: dict[str, np.ndarray],
    train_batches_per_epoch: int = 10,
    eval_batch_size: int = 200,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """論文の「10 batches in each epoch」に合わせた DataLoader を生成する．"""
    train_ds = ChessCheatDataset(
        splits["X_board_train"], splits["X_analyzed_train"], splits["y_train"]
    )
    val_ds = ChessCheatDataset(
        splits["X_board_val"], splits["X_analyzed_val"], splits["y_val"]
    )
    test_ds = ChessCheatDataset(
        splits["X_board_test"], splits["X_analyzed_test"], splits["y_test"]
    )

    # 3,600件 / 10バッチ = 360件/バッチ
    train_batch_size = max(1, (len(train_ds) + train_batches_per_epoch - 1) // train_batches_per_epoch)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """検証またはテストローダで損失・正解率・予測ラベル・確率を算出する．"""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for x_board, x_analyzed, targets in loader:
        x_board = x_board.to(device, non_blocking=True)
        x_analyzed = x_analyzed.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(x_board, x_analyzed)
        loss = criterion(logits, targets)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_targets)
    avg_loss = total_loss / max(1, total_samples)
    acc = float(accuracy_score(y_true, y_pred))
    return avg_loss, acc, y_true, y_pred, y_prob


def train_and_evaluate(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 15,
    lr: float = 0.001,
) -> dict[str, Any]:
    """論文の設定（Adam lr=0.001, 15 epochs）で学習し，履歴とテスト評価結果を返す．"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: list[dict[str, float]] = []
    start_time = time.perf_counter()

    pbar = tqdm(range(1, epochs + 1), desc=f"[{model_name}]", leave=True)
    for epoch in pbar:
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for x_board, x_analyzed, targets in train_loader:
            x_board = x_board.to(device, non_blocking=True)
            x_analyzed = x_analyzed.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(x_board, x_analyzed)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            preds = torch.argmax(logits, dim=1)
            running_correct += int((preds == targets).sum().item())
            running_total += batch_size

        train_loss = running_loss / max(1, running_total)
        train_acc = running_correct / max(1, running_total)
        val_loss, val_acc, _, _, _ = evaluate_model(model, val_loader, criterion, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        pbar.set_postfix(
            train_acc=f"{train_acc:.4f}",
            val_acc=f"{val_acc:.4f}",
            val_loss=f"{val_loss:.4f}",
        )

    elapsed_sec = time.perf_counter() - start_time
    test_loss, test_acc, y_true, y_pred, y_prob = evaluate_model(
        model, test_loader, criterion, device
    )
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    report_dict = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )

    history_df = pd.DataFrame(history)
    best_val_acc = float(history_df["val_acc"].max())
    final_val_acc = float(history_df["val_acc"].iloc[-1])
    final_train_acc = float(history_df["train_acc"].iloc[-1])

    return {
        "model_name": model_name,
        "num_params": sum(p.numel() for p in model.parameters()),
        "history": history_df,
        "final_train_acc": final_train_acc,
        "best_val_acc": best_val_acc,
        "final_val_acc": final_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(prec_weighted),
        "recall_weighted": float(rec_weighted),
        "f1_weighted": float(f1_weighted),
        "confusion_matrix": cm,
        "classification_report": report_dict,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "elapsed_sec": elapsed_sec,
    }


def summarize_results(results: list[dict[str, Any]]) -> pd.DataFrame:
    """複数モデルの結果をまとめた DataFrame を作成する．"""
    rows = []
    for res in results:
        rows.append(
            {
                "Model": res["model_name"],
                "Params": res["num_params"],
                "Final Train Acc (%)": res["final_train_acc"] * 100.0,
                "Best Val Acc (%)": res["best_val_acc"] * 100.0,
                "Final Val Acc (%)": res["final_val_acc"] * 100.0,
                "Test Acc (%)": res["test_acc"] * 100.0,
                "Macro F1 (%)": res["f1_macro"] * 100.0,
                "Train-Val Gap (%pt)": (res["final_train_acc"] - res["final_val_acc"]) * 100.0,
                "Time (s)": res["elapsed_sec"],
            }
        )
    return pd.DataFrame(rows)


def plot_validation_curves(
    results: list[dict[str, Any]],
    save_path: str | Path,
) -> None:
    """論文 Fig. 6 に対応する各エポックの Validation Accuracy 推移グラフを保存する．"""
    configure_japanese_font()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for res in results:
        df = res["history"]
        axes[0].plot(df["epoch"], df["val_acc"], marker="o", linewidth=2, label=res["model_name"])
        axes[1].plot(df["epoch"], df["train_acc"], marker="s", linewidth=2, label=res["model_name"])

    axes[0].set_title("検証データ正解率の推移 (Validation Accuracy)")
    axes[0].set_xlabel("エポック (Epoch)")
    axes[0].set_ylabel("正解率 (Accuracy)")
    axes[0].set_ylim(0.25, 1.02)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].set_title("訓練データ正解率の推移 (Training Accuracy)")
    axes[1].set_xlabel("エポック (Epoch)")
    axes[1].set_ylabel("正解率 (Accuracy)")
    axes[1].set_ylim(0.25, 1.02)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_comparison(
    summary_df: pd.DataFrame,
    save_path: str | Path,
) -> None:
    """Train / Best Val / Test Accuracy の比較棒グラフを保存する．"""
    configure_japanese_font()
    melted = summary_df.melt(
        id_vars="Model",
        value_vars=["Final Train Acc (%)", "Best Val Acc (%)", "Test Acc (%)"],
        var_name="Metric",
        value_name="Accuracy (%)",
    )
    label_map = {
        "Final Train Acc (%)": "訓練正解率 (Final Train)",
        "Best Val Acc (%)": "最高検証正解率 (Best Val)",
        "Test Acc (%)": "テスト正解率 (Test)",
    }
    melted["Metric"] = melted["Metric"].map(label_map)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(data=melted, x="Model", y="Accuracy (%)", hue="Metric", ax=ax, palette="viridis")
    ax.set_title("ex001: モデル別 訓練・検証・テスト正解率の比較（過学習の確認）")
    ax.set_xlabel("モデル")
    ax.set_ylabel("正解率 (%)")
    ax.set_ylim(0, 110)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", fontsize=8, padding=2)

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(
    results: list[dict[str, Any]],
    save_path: str | Path,
) -> None:
    """4モデルのテストデータ混同行列を並べて可視化・保存する．"""
    configure_japanese_font()
    n_models = len(results)
    cols = 2
    rows = (n_models + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(11, 4.5 * rows))
    axes_flat = np.array(axes).reshape(-1)

    for idx, res in enumerate(results):
        ax = axes_flat[idx]
        sns.heatmap(
            res["confusion_matrix"],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=CLASS_NAMES_JA,
            yticklabels=CLASS_NAMES_JA,
            ax=ax,
            cbar=False,
        )
        ax.set_title(f"{res['model_name']} (Test Acc: {res['test_acc']*100:.2f}%)")
        ax.set_xlabel("予測クラス (Predicted)")
        ax.set_ylabel("正解クラス (True)")

    for idx in range(n_models, len(axes_flat)):
        axes_flat[idx].axis("off")

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

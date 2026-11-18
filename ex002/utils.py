"""改善実験（ex002）用の前処理・学習・評価・特徴量重要度分析・可視化ユーティリティ．"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ex001.utils import (
    CLASS_NAMES,
    CLASS_NAMES_JA,
    ChessCheatDataset,
    configure_japanese_font,
    create_dataloaders,
    evaluate_model,
    load_and_split_dataset,
    plot_confusion_matrices,
    set_seed,
    summarize_results,
)


EXTENDED_FEATURE_NAMES = [
    "評価値 (tanh圧縮)",
    "評価値損失 (CPL)",
    "評価値変動幅 (|Δeval|)",
    "残り時間割合 (clk/init)",
    "消費時間 (log Δt)",
    "手番フラグ (White/Black)",
]


def standardize_extended_features(splits: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Train 分割の有効手のみから平均・標準偏差を算出し，Train/Val/Test を Z-score 標準化する．"""
    out = dict(splits)
    x_tr = splits["X_analyzed_train"].copy()
    x_va = splits["X_analyzed_val"].copy()
    x_te = splits["X_analyzed_test"].copy()

    # 盤面に白駒が1つ以上存在するタイムステップを有効手とみなす
    mask_tr = splits["X_board_train"][:, :, 6].sum(axis=(2, 3)) > 0
    mask_va = splits["X_board_val"][:, :, 6].sum(axis=(2, 3)) > 0
    mask_te = splits["X_board_test"][:, :, 6].sum(axis=(2, 3)) > 0

    valid_vals = x_tr[mask_tr]  # (num_valid_plies, C)
    mean = valid_vals.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(valid_vals.std(axis=0, dtype=np.float64), 1e-6).astype(np.float32)

    for arr, mask in [(x_tr, mask_tr), (x_va, mask_va), (x_te, mask_te)]:
        arr[mask] = (arr[mask] - mean) / std
        arr[~mask] = 0.0

    out["X_analyzed_train"] = x_tr
    out["X_analyzed_val"] = x_va
    out["X_analyzed_test"] = x_te
    return out, mean, std


def train_and_evaluate_improved(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 25,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.05,
) -> dict[str, Any]:
    """AdamW + CosineAnnealingLR + ベスト検証モデル復元を用いて学習とテスト評価を行う．"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    eval_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history: list[dict[str, float]] = []
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            preds = torch.argmax(logits, dim=1)
            running_correct += int((preds == targets).sum().item())
            running_total += batch_size

        scheduler.step()

        train_loss = running_loss / max(1, running_total)
        train_acc = running_correct / max(1, running_total)
        val_loss, val_acc, _, _, _ = evaluate_model(model, val_loader, eval_criterion, device)

        # 検証正解率最大（同率なら検証損失最小）の重みを保存
        if (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

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
            best_val=f"{best_val_acc:.4f}",
        )

    elapsed_sec = time.perf_counter() - start_time

    # ベスト検証重みでテスト評価
    model.load_state_dict(best_state)
    test_loss, test_acc, y_true, y_pred, y_prob = evaluate_model(
        model, test_loader, eval_criterion, device
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
    final_val_acc = float(history_df["val_acc"].iloc[-1])
    final_train_acc = float(history_df["train_acc"].iloc[-1])

    return {
        "model_name": model_name,
        "model": model,
        "num_params": sum(p.numel() for p in model.parameters()),
        "history": history_df,
        "final_train_acc": final_train_acc,
        "best_val_acc": float(best_val_acc),
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


@torch.no_grad()
def compute_channel_ablation_importance(
    model: nn.Module,
    splits: dict[str, np.ndarray],
    device: torch.device,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """テストデータで拡張解析特徴量の各チャンネルをゼロマスクし，正解率低下量から重要度を測定する．"""
    if feature_names is None:
        feature_names = EXTENDED_FEATURE_NAMES

    model.eval()
    eval_criterion = nn.CrossEntropyLoss()

    def run_eval(x_analyzed_arr: np.ndarray) -> float:
        ds = ChessCheatDataset(splits["X_board_test"], x_analyzed_arr, splits["y_test"])
        loader = DataLoader(ds, batch_size=200, shuffle=False)
        _, acc, _, _, _ = evaluate_model(model, loader, eval_criterion, device)
        return acc

    base_acc = run_eval(splits["X_analyzed_test"])
    rows = []
    for ch_idx, fname in enumerate(feature_names):
        masked = splits["X_analyzed_test"].copy()
        masked[:, :, ch_idx] = 0.0
        masked_acc = run_eval(masked)
        rows.append(
            {
                "Feature": fname,
                "Base Acc (%)": base_acc * 100.0,
                "Masked Acc (%)": masked_acc * 100.0,
                "Acc Drop (%pt)": (base_acc - masked_acc) * 100.0,
            }
        )

    return pd.DataFrame(rows).sort_values("Acc Drop (%pt)", ascending=False).reset_index(drop=True)


def plot_improved_learning_curves(
    results: list[dict[str, Any]],
    save_path: str | Path,
) -> None:
    """実験2モデルの学習・検証正解率および損失推移グラフを保存する．"""
    configure_japanese_font()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for res in results:
        df = res["history"]
        axes[0].plot(df["epoch"], df["val_acc"], marker="o", linewidth=2, label=f"{res['model_name']} (Val)")
        axes[0].plot(df["epoch"], df["train_acc"], linestyle="--", alpha=0.65, label=f"{res['model_name']} (Train)")
        axes[1].plot(df["epoch"], df["val_loss"], marker="o", linewidth=2, label=f"{res['model_name']} (Val)")

    axes[0].set_title("ex002: 訓練・検証正解率の推移（正則化と過学習抑制）")
    axes[0].set_xlabel("エポック (Epoch)")
    axes[0].set_ylabel("正解率 (Accuracy)")
    axes[0].set_ylim(0.30, 1.02)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=8, ncol=2)

    axes[1].set_title("ex002: 検証損失の推移 (Validation Loss)")
    axes[1].set_xlabel("エポック (Epoch)")
    axes[1].set_ylabel("交差エントロピー損失")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_ex001_vs_ex002_comparison(
    combined_df: pd.DataFrame,
    save_path: str | Path,
) -> None:
    """ex001 と ex002 のテスト精度および過学習ギャップを比較可視化する．"""
    configure_japanese_font()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    sns.barplot(
        data=combined_df,
        x="Test Acc (%)",
        y="Model",
        hue="Experiment",
        ax=axes[0],
        palette="Set2",
    )
    axes[0].set_title("テスト正解率の比較 (ex001 vs ex002)")
    axes[0].set_xlabel("テスト正解率 (%)")
    axes[0].set_ylabel("")
    axes[0].set_xlim(0, 100)
    axes[0].grid(True, axis="x", linestyle="--", alpha=0.5)
    for c in axes[0].containers:
        axes[0].bar_label(c, fmt="%.2f%%", padding=3, fontsize=8)

    sns.barplot(
        data=combined_df,
        x="Train-Val Gap (%pt)",
        y="Model",
        hue="Experiment",
        ax=axes[1],
        palette="Set2",
    )
    axes[1].set_title("過学習ギャップの比較 (Final Train - Final Val %pt)")
    axes[1].set_xlabel("訓練−検証正解率ギャップ (%pt, 低いほど良好)")
    axes[1].set_ylabel("")
    axes[1].grid(True, axis="x", linestyle="--", alpha=0.5)
    for c in axes[1].containers:
        axes[1].bar_label(c, fmt="%.1f", padding=3, fontsize=8)

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(
    imp_df: pd.DataFrame,
    save_path: str | Path,
    model_name: str,
) -> None:
    """チャンネルアブレーションによる拡張特徴量重要度グラフを保存する．"""
    configure_japanese_font()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(data=imp_df, x="Acc Drop (%pt)", y="Feature", ax=ax, palette="magma", hue="Feature", legend=False)
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_title(f"{model_name} における解析特徴量チャンネル別アブレーション重要度")
    ax.set_xlabel("特徴量マスク時のテスト正解率低下量 (%pt)")
    ax.set_ylabel("特徴量")
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", padding=3, fontsize=9)

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_extended_feature_distributions(
    x_board: np.ndarray,
    x_analyzed_ext: np.ndarray,
    y: np.ndarray,
    save_path: str | Path,
) -> pd.DataFrame:
    """チーター手番 vs 非チーター手番での CPL・消費時間等の統計と分布図を保存する．"""
    configure_japanese_font()
    valid_mask = x_board[:, :, 6].sum(axis=(2, 3)) > 0
    is_white_ply = (np.arange(x_board.shape[1])[None, :] % 2) == 0

    cheater_mask = np.zeros_like(valid_mask, dtype=bool)
    human_mask = np.zeros_like(valid_mask, dtype=bool)

    # Label 0: 両者とも人間, Label 1: 白番のみチーター, Label 2: 黒番のみチーター
    human_mask[y == 0] = valid_mask[y == 0]
    cheater_mask[y == 1] = valid_mask[y == 1] & is_white_ply[0]
    human_mask[y == 1] = valid_mask[y == 1] & (~is_white_ply[0])
    cheater_mask[y == 2] = valid_mask[y == 2] & (~is_white_ply[0])
    human_mask[y == 2] = valid_mask[y == 2] & is_white_ply[0]

    cpl_human = x_analyzed_ext[:, :, 1][human_mask]
    cpl_cheat = x_analyzed_ext[:, :, 1][cheater_mask]
    dt_human = np.expm1(x_analyzed_ext[:, :, 4][human_mask])
    dt_cheat = np.expm1(x_analyzed_ext[:, :, 4][cheater_mask])
    deval_human = x_analyzed_ext[:, :, 2][human_mask]
    deval_cheat = x_analyzed_ext[:, :, 2][cheater_mask]

    stats_df = pd.DataFrame(
        [
            {
                "Group": "非チーター手番 (Human)",
                "Plies": int(human_mask.sum()),
                "Mean CPL (pawn)": float(cpl_human.mean()),
                "Median CPL (pawn)": float(np.median(cpl_human)),
                "Zero-Loss Ratio (%)": float((cpl_human < 0.05).mean() * 100.0),
                "Mean Think Time (s)": float(dt_human.mean()),
                "Std Think Time (s)": float(dt_human.std()),
                "Mean |Δeval| (pawn)": float(deval_human.mean()),
            },
            {
                "Group": "チーター手番 (Cheater)",
                "Plies": int(cheater_mask.sum()),
                "Mean CPL (pawn)": float(cpl_cheat.mean()),
                "Median CPL (pawn)": float(np.median(cpl_cheat)),
                "Zero-Loss Ratio (%)": float((cpl_cheat < 0.05).mean() * 100.0),
                "Mean Think Time (s)": float(dt_cheat.mean()),
                "Std Think Time (s)": float(dt_cheat.std()),
                "Mean |Δeval| (pawn)": float(deval_cheat.mean()),
            },
        ]
    )

    # 分布図（対局単位の平均 CPL と消費時間の標準偏差の比較）
    game_rows = []
    for idx in range(len(y)):
        m = valid_mask[idx]
        if m.sum() < 4:
            continue
        w_m = m & is_white_ply[0]
        b_m = m & (~is_white_ply[0])
        if w_m.sum() < 2 or b_m.sum() < 2:
            continue
        w_cpl = float(x_analyzed_ext[idx, w_m, 1].mean())
        b_cpl = float(x_analyzed_ext[idx, b_m, 1].mean())
        w_dt = float(np.expm1(x_analyzed_ext[idx, w_m, 4]).mean())
        b_dt = float(np.expm1(x_analyzed_ext[idx, b_m, 4]).mean())
        game_rows.append(
            {
                "Label": CLASS_NAMES_JA[int(y[idx])],
                "White-Black Mean CPL Diff": w_cpl - b_cpl,
                "White-Black Mean ThinkTime Diff (s)": w_dt - b_dt,
            }
        )
    gdf = pd.DataFrame(game_rows)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.boxplot(
        data=gdf,
        x="Label",
        y="White-Black Mean CPL Diff",
        ax=axes[0],
        palette="Set2",
        hue="Label",
        legend=False,
        showfliers=False,
    )
    axes[0].axhline(0.0, color="gray", linestyle="--", linewidth=1)
    axes[0].set_title("クラス別の白黒平均評価値損失差 (White CPL - Black CPL)")
    axes[0].set_xlabel("クラス")
    axes[0].set_ylabel("平均 CPL 差（ポーン単位）")
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.5)

    sns.boxplot(
        data=gdf,
        x="Label",
        y="White-Black Mean ThinkTime Diff (s)",
        ax=axes[1],
        palette="Set2",
        hue="Label",
        legend=False,
        showfliers=False,
    )
    axes[1].axhline(0.0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_title("クラス別の白黒平均消費時間差 (White Δt - Black Δt)")
    axes[1].set_xlabel("クラス")
    axes[1].set_ylabel("平均消費時間差（秒）")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return stats_df

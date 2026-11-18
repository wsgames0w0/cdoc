"""改善実験（ex002）用のニューラルネットワークモデル定義モジュール．

ex001 で確認された論文モデルの課題：
1. 盤面テンソルの高次元性（71,680次元 / 37,888次元）による深刻な過学習
2. 解析特徴量（評価値・残り時間）の未正規化とブランチ間の重み・表現力不均衡
を解決するため，BatchNorm・Dropout強化・プーリング圧縮・正規化および
拡張解析特徴量（Centipawn Loss・消費時間等 6次元時系列）を活用するモデル群を実装する．
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RegularizedBoardDenseEncoder(nn.Module):
    """正則化（BatchNorm1d + 高Dropout）を導入した盤面全結合エンコーダ．"""

    def __init__(
        self,
        in_features: int = 80 * 14 * 8 * 8,
        out_dim: int = 64,
        dropout_p: float = 0.5,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )

    def forward(self, x_board: torch.Tensor) -> torch.Tensor:
        return self.net(x_board)


class RegularizedBoardConv3DEncoder(nn.Module):
    """BatchNorm3d・空間Dropout・適応的プーリングにより過学習を抑制した 3D CNN エンコーダ．"""

    def __init__(self, out_dim: int = 64, dropout_p: float = 0.4) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(14, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Dropout3d(p=0.1),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),  # (40, 4, 4)
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Dropout3d(p=0.15),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),  # (20, 2, 2)
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            # 時間軸を 4 セグメント（序盤・中盤1・中盤2・終盤），空間を 1x1 に集約して丸暗記を防止
            nn.AdaptiveAvgPool3d((4, 1, 1)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_p),
            nn.Linear(128 * 4, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )

    def forward(self, x_board: torch.Tensor) -> torch.Tensor:
        # (B, 80, 14, 8, 8) -> (B, 14, 80, 8, 8)
        x = x_board.permute(0, 2, 1, 3, 4).contiguous()
        x = self.conv(x)
        return self.fc(x)


class ExtendedAnalyzedMLPEncoder(nn.Module):
    """正規化済み拡張解析特徴量 (B, 80, C) 用の多層 MLP エンコーダ．"""

    def __init__(
        self,
        seq_len: int = 80,
        num_features: int = 6,
        out_dim: int = 128,
        dropout_p: float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )

    def forward(self, x_analyzed: torch.Tensor) -> torch.Tensor:
        return self.net(x_analyzed)


class TemporalConvEncoder(nn.Module):
    """拡張解析特徴量 (B, 80, C) から手番・連続手の局所推移を抽出する 1D-CNN エンコーダ．"""

    def __init__(
        self,
        num_features: int = 6,
        out_dim: int = 128,
        dropout_p: float = 0.25,
    ) -> None:
        super().__init__()
        # 白番・黒番の交互性（2手周期）と数手の連続一貫性を捉える畳み込みブロック
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(num_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=dropout_p * 0.5),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            # AvgPool と MaxPool の結合（128 * 2 = 256次元）で平均的傾向とピーク異常値の両方を捕捉
            nn.Linear(128 * 2, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )

    def forward(self, x_analyzed: torch.Tensor) -> torch.Tensor:
        # (B, 80, C) -> (B, C, 80)
        x = x_analyzed.permute(0, 2, 1).contiguous()
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        avg_pool = x.mean(dim=-1)
        max_pool = x.amax(dim=-1)
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        return self.head(pooled)


class AnalyzedOnlyMLP(nn.Module):
    """アブレーション用：盤面を使用せず拡張解析特徴量のみを入力とする MLP モデル．"""

    def __init__(
        self,
        num_classes: int = 3,
        num_analyzed_features: int = 6,
        dropout_p: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = ExtendedAnalyzedMLPEncoder(
            num_features=num_analyzed_features, out_dim=128, dropout_p=dropout_p
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        x_board: torch.Tensor,
        x_analyzed: torch.Tensor,
    ) -> torch.Tensor:
        feat = self.encoder(x_analyzed)
        return self.classifier(feat)


class TemporalConvNet(nn.Module):
    """アブレーション用：拡張解析特徴量のみを 1D-CNN で処理する時系列モデル．"""

    def __init__(
        self,
        num_classes: int = 3,
        num_analyzed_features: int = 6,
        dropout_p: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = TemporalConvEncoder(
            num_features=num_analyzed_features, out_dim=128, dropout_p=dropout_p
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        x_board: torch.Tensor,
        x_analyzed: torch.Tensor,
    ) -> torch.Tensor:
        feat = self.encoder(x_analyzed)
        return self.classifier(feat)


class ImprovedAnalyzedDense(nn.Module):
    """正則化と表現バランスを改善した Analyzed Dense モデル．"""

    def __init__(
        self,
        num_classes: int = 3,
        num_analyzed_features: int = 6,
        dropout_p: float = 0.4,
    ) -> None:
        super().__init__()
        self.board_encoder = RegularizedBoardDenseEncoder(out_dim=64, dropout_p=0.5)
        self.analyzed_encoder = ExtendedAnalyzedMLPEncoder(
            num_features=num_analyzed_features, out_dim=128, dropout_p=0.3
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(64 + 128),
            nn.Dropout(p=dropout_p),
            nn.Linear(64 + 128, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_p * 0.5),
            nn.Linear(64, num_classes),
        )

    def forward(
        self,
        x_board: torch.Tensor,
        x_analyzed: torch.Tensor,
    ) -> torch.Tensor:
        b_feat = self.board_encoder(x_board)
        a_feat = self.analyzed_encoder(x_analyzed)
        combined = torch.cat([b_feat, a_feat], dim=1)
        return self.fusion(combined)


class ImprovedAnalyzedCNN(nn.Module):
    """3D CNN 盤面特徴と 1D Temporal CNN 拡張解析特徴を融合した改善モデル．"""

    def __init__(
        self,
        num_classes: int = 3,
        num_analyzed_features: int = 6,
        dropout_p: float = 0.35,
    ) -> None:
        super().__init__()
        self.board_encoder = RegularizedBoardConv3DEncoder(out_dim=64, dropout_p=0.4)
        self.analyzed_encoder = TemporalConvEncoder(
            num_features=num_analyzed_features, out_dim=128, dropout_p=0.25
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(64 + 128),
            nn.Dropout(p=dropout_p),
            nn.Linear(64 + 128, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_p * 0.5),
            nn.Linear(64, num_classes),
        )

    def forward(
        self,
        x_board: torch.Tensor,
        x_analyzed: torch.Tensor,
    ) -> torch.Tensor:
        b_feat = self.board_encoder(x_board)
        a_feat = self.analyzed_encoder(x_analyzed)
        combined = torch.cat([b_feat, a_feat], dim=1)
        return self.fusion(combined)


def build_improved_model(
    model_name: str,
    num_classes: int = 3,
    num_analyzed_features: int = 6,
) -> nn.Module:
    """モデル名から実験2のモデルインスタンスを生成する．"""
    registry: dict[str, type[nn.Module]] = {
        "Analyzed-Only MLP": AnalyzedOnlyMLP,
        "Temporal ConvNet": TemporalConvNet,
        "Improved Analyzed Dense": ImprovedAnalyzedDense,
        "Improved Analyzed CNN": ImprovedAnalyzedCNN,
    }
    if model_name not in registry:
        raise ValueError(f"未知のモデル名です: {model_name}. 候補: {list(registry.keys())}")
    return registry[model_name](
        num_classes=num_classes,
        num_analyzed_features=num_analyzed_features,
    )

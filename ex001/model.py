"""論文再現実験（ex001）用のニューラルネットワークモデル定義モジュール．

論文「Cheat Detection on Online Chess Games using Convolutional and Dense Neural Network」
(Patria et al., ISRITI 2021) の Section III.B および Fig. 5 に基づき，
4種類のモデル（Unanalyzed Dense, Unanalyzed CNN, Analyzed Dense, Analyzed CNN）を実装する．
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BoardDenseEncoder(nn.Module):
    """盤面テンソル (B, 80, 14, 8, 8) を平坦化して特徴抽出する全結合エンコーダ．"""

    def __init__(self, in_features: int = 80 * 14 * 8 * 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
        )

    def forward(self, x_board: torch.Tensor) -> torch.Tensor:
        return self.net(x_board)


class BoardConv3DEncoder(nn.Module):
    """盤面テンソル (B, 80, 14, 8, 8) から3次元畳み込みにより時空間特徴を抽出するエンコーダ．"""

    def __init__(self) -> None:
        super().__init__()
        # Keras の (N, 80, 8, 8, 14) に対する Conv3D と等価にするため，
        # PyTorch の (N, C=14, D=80, H=8, W=8) 形式で畳み込みを適用する．
        self.conv = nn.Sequential(
            nn.Conv3d(14, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, padding=0),
            nn.ReLU(),
        )
        # 80 -> 78 -> 76 -> 74, 8 -> 6 -> 4 -> 2 のため平坦化後は 74 * 2 * 2 * 128 = 37888 次元
        self.fc = nn.Sequential(
            nn.Linear(74 * 2 * 2 * 128, 32),
            nn.ReLU(),
        )

    def forward(self, x_board: torch.Tensor) -> torch.Tensor:
        # (B, 80, 14, 8, 8) -> (B, 14, 80, 8, 8)
        x = x_board.permute(0, 2, 1, 3, 4).contiguous()
        x = self.conv(x)
        # Keras の channels_last Flatten と並びを揃えるため (B, D, H, W, C) へ並べ替えて平坦化
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = x.flatten(start_dim=1)
        return self.fc(x)


class AnalyzedFeatureEncoder(nn.Module):
    """解析特徴量テンソル (B, 80, 2)（Stockfish評価値・残り持ち時間）用の全結合エンコーダ．"""

    def __init__(self, seq_len: int = 80, num_features: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
        )

    def forward(self, x_analyzed: torch.Tensor) -> torch.Tensor:
        return self.net(x_analyzed)


class UnanalyzedDense(nn.Module):
    """未解析データ（盤面のみ）を入力とする Dense Neural Network（Fig. 5c）．"""

    def __init__(self, num_classes: int = 3, dropout_p: float = 0.2) -> None:
        super().__init__()
        self.encoder = BoardDenseEncoder()
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(64, num_classes),
        )

    def forward(
        self,
        x_board: torch.Tensor,
        x_analyzed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feat = self.encoder(x_board)
        return self.classifier(feat)


class UnanalyzedCNN(nn.Module):
    """未解析データ（盤面のみ）を入力とする 3D Convolutional Neural Network（Fig. 5a）．"""

    def __init__(self, num_classes: int = 3, dropout_p: float = 0.2) -> None:
        super().__init__()
        self.encoder = BoardConv3DEncoder()
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(32, num_classes),
        )

    def forward(
        self,
        x_board: torch.Tensor,
        x_analyzed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feat = self.encoder(x_board)
        return self.classifier(feat)


class AnalyzedDense(nn.Module):
    """解析済みデータ（盤面 + Stockfish評価値・持ち時間）を入力とする Dense Neural Network（Fig. 5d）．"""

    def __init__(self, num_classes: int = 3, dropout_p: float = 0.2) -> None:
        super().__init__()
        self.board_encoder = BoardDenseEncoder()
        self.analyzed_encoder = AnalyzedFeatureEncoder()
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(64 + 64, num_classes),
        )

    def forward(
        self,
        x_board: torch.Tensor,
        x_analyzed: torch.Tensor,
    ) -> torch.Tensor:
        board_feat = self.board_encoder(x_board)
        analyzed_feat = self.analyzed_encoder(x_analyzed)
        combined = torch.cat([board_feat, analyzed_feat], dim=1)
        return self.classifier(combined)


class AnalyzedCNN(nn.Module):
    """解析済みデータ（盤面 + Stockfish評価値・持ち時間）を入力とする 3D CNN（Fig. 5b）．"""

    def __init__(self, num_classes: int = 3, dropout_p: float = 0.2) -> None:
        super().__init__()
        self.board_encoder = BoardConv3DEncoder()
        self.analyzed_encoder = AnalyzedFeatureEncoder()
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(32 + 64, num_classes),
        )

    def forward(
        self,
        x_board: torch.Tensor,
        x_analyzed: torch.Tensor,
    ) -> torch.Tensor:
        board_feat = self.board_encoder(x_board)
        analyzed_feat = self.analyzed_encoder(x_analyzed)
        combined = torch.cat([board_feat, analyzed_feat], dim=1)
        return self.classifier(combined)


def build_model(model_name: str, num_classes: int = 3, dropout_p: float = 0.2) -> nn.Module:
    """モデル名から該当する実験1モデルインスタンスを生成する．"""
    registry: dict[str, type[nn.Module]] = {
        "Unanalyzed Dense": UnanalyzedDense,
        "Unanalyzed CNN": UnanalyzedCNN,
        "Analyzed Dense": AnalyzedDense,
        "Analyzed CNN": AnalyzedCNN,
    }
    if model_name not in registry:
        raise ValueError(f"未知のモデル名です: {model_name}. 候補: {list(registry.keys())}")
    return registry[model_name](num_classes=num_classes, dropout_p=dropout_p)

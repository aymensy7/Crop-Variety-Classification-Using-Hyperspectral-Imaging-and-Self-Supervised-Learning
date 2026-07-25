"""
Model architectures shared by the self-supervised (SSL) and supervised
training stages.

Previously `SpectralAttention` and `HyperspectralEncoder` were copy-pasted
into both `selfsupervised.py` and `supervised.py` with small differences
(the supervised copy added dropout and dropped the projection head). Both
variants are unified here behind constructor flags so there is a single
source of truth, while producing state_dicts identical in structure to the
original two versions (safe to load the existing checkpoints in
results/models/).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralAttention(nn.Module):
    """Squeeze-and-excitation style channel (spectral band) attention."""

    def __init__(self, num_bands, reduction_ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_bands, num_bands // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_bands // reduction_ratio, num_bands, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x.view(b, c, -1)).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class HyperspectralEncoder(nn.Module):
    """CNN backbone with spectral attention, used by both training stages.

    Args:
        input_bands: number of spectral bands in the input patch.
        patch_size: spatial size of the input patch (unused directly, kept
            for interface parity with the original scripts).
        projection_dim: if set, attaches a SimCLR projection head on top of
            the pooled features (used only during SSL pre-training).
        use_dropout: whether to apply dropout after the attention and second
            conv block (used during supervised fine-tuning; the original SSL
            script trained without dropout in the encoder).
        dropout_p: dropout probability when `use_dropout=True`.
    """

    def __init__(
        self,
        input_bands=200,
        patch_size=5,
        projection_dim=None,
        use_dropout=False,
        dropout_p=0.15,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(input_bands, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.spectral_attn = SpectralAttention(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_p)
        self.use_dropout = use_dropout

        self.projection_head = None
        if projection_dim is not None:
            self.projection_head = nn.Sequential(
                nn.Linear(256, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Linear(512, projection_dim),
            )

    def forward(self, x, use_projection=False):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.spectral_attn(x)
        if self.use_dropout:
            x = self.dropout(x)

        x = F.relu(self.bn2(self.conv2(x)))
        if self.use_dropout:
            x = self.dropout(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)

        if use_projection and self.projection_head is not None:
            return self.projection_head(x)
        return x


class HyperspectralClassifier(nn.Module):
    """Supervised classification head on top of the (optionally SSL
    pre-trained) HyperspectralEncoder."""

    def __init__(self, input_bands=200, patch_size=5, num_classes=7):
        super().__init__()
        self.encoder = HyperspectralEncoder(input_bands, patch_size, use_dropout=True)

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

        self.freeze_encoder()

    def freeze_encoder(self, freeze=True):
        for param in self.encoder.parameters():
            param.requires_grad = not freeze

    def forward(self, x):
        features = self.encoder(x, use_projection=False)
        return self.classifier(features)


class SimCLRLoss(nn.Module):
    """Standard NT-Xent contrastive loss used for SimCLR pre-training."""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, features):
        """
        Args:
            features: Tensor of shape [2*batch_size, projection_dim], where
                the first half and second half are the two augmented views.
        """
        batch_size = features.shape[0] // 2
        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=features.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, float("-inf"))

        labels = torch.cat(
            [
                torch.arange(batch_size, device=features.device) + batch_size,
                torch.arange(batch_size, device=features.device),
            ],
            dim=0,
        )

        return self.criterion(similarity_matrix, labels)


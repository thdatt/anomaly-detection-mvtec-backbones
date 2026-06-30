"""
Minimal unsupervised anomaly detection system.

Philosophy: Simple, explicit, minimal abstractions.
- One feature extractor class per backbone type
- Memory bank handles normalization and selection
- Scoring is a pure function
- Config groups parameters by semantic meaning (not by phase)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import transforms
from pathlib import Path
import numpy as np


# ============================================================================
# CONFIG
# ============================================================================

class Config:
    """Hyperparameters for anomaly detection pipeline."""
    def __init__(self):
        # Dataset paths
        self.base_path = Path('mvtec_anomaly_detection')
        self.dataset_name = 'toothbrush'

        # Device & batch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = 4

        # Feature extraction (set by get_feature_extractor)
        self.image_size = 224
        self.feature_dim = 1536

        # Memory bank compression
        self.coreset_ratio = 0.1

        # Scoring: image-level score = max over the (smoothed) patch score map
        self.smooth_sigma = 1.0

        # Threshold: where to draw decision boundary on held-out good scores
        self.threshold_method = 'sigma'    # 'sigma', 'quantile', or 'max'
        self.threshold_param = 2.0         # k (for sigma) or q (for quantile)

    def get_paths(self):
        """Returns train_good, test_good, test_defective paths."""
        return {
            'train_good': self.base_path / self.dataset_name / 'train' / 'good',
            'test_good': self.base_path / self.dataset_name / 'test' / 'good',
            'test_defective': self.base_path / self.dataset_name / 'test' / 'defective',
        }


# ============================================================================
# IMAGE TRANSFORMS
# ============================================================================

def get_transform(image_size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Resize, normalize to ImageNet stats."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


# ============================================================================
# FEATURE EXTRACTORS
# ============================================================================

class FeatureExtractor(nn.Module):
    """Base class. Forward returns (batch, features) after pooling patches."""
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.transform = None  # set by subclass

    def forward(self, x):
        """Expects (B, C, H, W). Returns (B*n_patches, feature_dim)."""
        raise NotImplementedError


class ResNet50Extractor(FeatureExtractor):
    """ResNet50 conv5 features, avg pooled, normalized."""
    def __init__(self, device='cuda'):
        super().__init__(device)

        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        self.backbone = model.to(device)

        # Extract hooks from layer2 and layer3
        self.feats = []
        self.backbone.layer2[-1].register_forward_hook(lambda m, i, o: self.feats.append(o))
        self.backbone.layer3[-1].register_forward_hook(lambda m, i, o: self.feats.append(o))

        self.transform = get_transform(image_size=224)

    def forward(self, x):
        x = x.to(self.device)
        self.feats = []
        with torch.no_grad():
            _ = self.backbone(x)

        # Resize both feature maps to same spatial size, concatenate, flatten
        avg_pool = nn.AvgPool2d(3, stride=1)
        fmap_size = self.feats[0].shape[-2]
        adaptive_pool = nn.AdaptiveAvgPool2d(fmap_size)

        resized = [adaptive_pool(avg_pool(f)) for f in self.feats]
        patch = torch.cat(resized, dim=1)  # (B, C, H, W)
        patch = patch.reshape(patch.shape[1], -1).T  # (H*W, C)
        return patch


class ViTExtractor(FeatureExtractor):
    """Vision Transformer (DINO or DINOv2) patch tokens."""
    def __init__(self, device='cuda', model_name='vit_small_patch8_224.dino', img_size=None):
        super().__init__(device)

        import timm
        from timm.data import resolve_model_data_config

        kwargs = dict(pretrained=True)
        if img_size:
            kwargs['img_size'] = img_size
            kwargs['dynamic_img_size'] = True

        model = timm.create_model(model_name, **kwargs)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        self.backbone = model.to(device)

        cfg = resolve_model_data_config(self.backbone)
        size = img_size or cfg['input_size'][-1]
        self.transform = get_transform(image_size=size, mean=cfg['mean'], std=cfg['std'])
        self.num_prefix_tokens = getattr(self.backbone, 'num_prefix_tokens', 1)

    def forward(self, x):
        x = x.to(self.device)
        with torch.no_grad():
            tokens = self.backbone.forward_features(x)
        # Skip CLS/prefix tokens, keep patch tokens only
        return tokens[:, self.num_prefix_tokens:, :].squeeze(0)


def get_feature_extractor(name, device='cuda', img_size=None):
    """Factory: return configured extractor + transform.

    img_size: optional override of the input resolution for ViT backbones
    (uses dynamic position-embedding interpolation). Ignored for ResNet50,
    which is fixed at 224. Used to reproduce the resolution sweep in the
    report (e.g. DINO at 320/384) without editing code.
    """
    if name == 'resnet50':
        return ResNet50Extractor(device=device)
    elif name == 'dino_vit':
        return ViTExtractor(device=device, model_name='vit_small_patch8_224.dino', img_size=img_size)
    elif name == 'dinov2':
        return ViTExtractor(device=device, model_name='vit_small_patch14_dinov2.lvd142m', img_size=img_size)
    else:
        raise ValueError(f"Unknown backbone: {name}")


# ============================================================================
# MEMORY BANK
# ============================================================================

class MemoryBank:
    """Stores normalized feature vectors, supports k-center greedy selection."""
    def __init__(self, device='cuda'):
        self.device = device
        self.features = None

    def build(self, feature_list):
        """Concatenate features and L2-normalize."""
        bank = torch.cat(feature_list, dim=0)
        bank = bank / (torch.norm(bank, dim=1, keepdim=True) + 1e-8)
        self.features = bank.to(self.device)

    def select_k_center_greedy(self, ratio=0.1, max_samples=50000):
        """Reduce bank size via farthest-point-first greedy selection."""
        if self.features is None:
            raise ValueError("Call build() first")

        # Pre-pool if too large (for efficiency)
        if self.features.shape[0] > max_samples:
            idx = torch.linspace(0, self.features.shape[0] - 1, max_samples).long()
            self.features = self.features[idx.to(self.device)]

        n = self.features.shape[0]
        k = max(1, int(n * ratio))
        if k >= n:
            return

        # Greedy: pick farthest point each iteration
        selected = [0]
        dists = torch.cdist(self.features, self.features[0:1]).squeeze(1)
        dists[0] = -1

        for _ in range(1, k):
            farthest = torch.argmax(dists).item()
            selected.append(farthest)
            new_dists = torch.cdist(self.features, self.features[farthest:farthest+1]).squeeze(1)
            dists = torch.minimum(dists, new_dists)
            dists[farthest] = -1

        self.features = self.features[selected]

    @property
    def size(self):
        """Return number of features in bank."""
        return self.features.shape[0] if self.features is not None else 0

    def get(self):
        """Return the feature bank."""
        return self.features


# ============================================================================
# SCORING FUNCTIONS
# ============================================================================

def smooth_map(score_map, sigma):
    """Apply 2D Gaussian smoothing to a score map."""
    if sigma is None or sigma <= 0:
        return score_map

    radius = max(1, int(round(3 * sigma)))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=score_map.device)
    kernel = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()

    m = score_map.unsqueeze(0).unsqueeze(0)
    m = F.conv2d(m, kernel.view(1, 1, 1, -1), padding=(0, radius))
    m = F.conv2d(m, kernel.view(1, 1, -1, 1), padding=(radius, 0))
    return m.squeeze(0).squeeze(0)


def image_anomaly_score(patch_scores, smooth_sigma=0.0):
    """Reduce patch-level scores to an image-level anomaly score: the maximum over
    the (optionally Gaussian-smoothed) spatial score map.

    Args:
        patch_scores: (n_patches,) tensor
        smooth_sigma: if > 0, apply Gaussian smoothing to the spatial map first

    Returns: scalar anomaly score
    """
    if smooth_sigma > 0:
        side = int(round(patch_scores.numel() ** 0.5))
        if side * side == patch_scores.numel():
            smoothed = smooth_map(patch_scores.view(side, side), smooth_sigma)
            patch_scores = smoothed.reshape(-1)

    return patch_scores.max()


def compute_threshold(calib_scores, method='sigma', param=2.0):
    """Set decision threshold from held-out GOOD scores only (leakage-free).

    Args:
        calib_scores: 1D array of anomaly scores from held-out good images
        method: 'sigma' (mean + k*std), 'quantile' (q-quantile), or 'max'
        param: k for sigma, q for quantile

    Returns: float threshold
    """
    s = np.asarray(calib_scores, dtype=np.float64)

    if method == 'sigma':
        return float(s.mean() + param * s.std())
    elif method == 'quantile':
        return float(np.quantile(s, param))
    elif method == 'max':
        return float(s.max())
    else:
        raise ValueError(f"Unknown threshold method: {method}")

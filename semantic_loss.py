import torch
import torch.nn as nn
import torchvision.models


class CachedLPIPS(nn.Module):
    """
    Implements a LPIPS-like perceptural loss, with a single cached target feature map.
    For saving compute in cases where the target image is always the same.
    """

    def __init__(self, normalize=True):
        super(CachedLPIPS, self).__init__()
        self.feature_extractor = CachedLPIPS._FeatureExtractor()
        self.normalize = normalize
        self.target = None

    def set_target(self, target_img):
        # precompute target feature map
        self.target = self.feature_extractor(target_img)
        self.target = [self._l2_normalize_features(f) for f in self.target]

    def _l2_normalize_features(self, x, eps=1e-10):
        nrm = torch.sqrt(torch.sum(x*x, dim=1, keepdim=True))
        return x / (nrm + eps)

    def diffs_map(self, pred):
        # Get VGG features
        pred = self.feature_extractor(pred)

        # L2 normalize features
        if self.normalize:
            pred = [self._l2_normalize_features(f) for f in pred]

        diffs = [torch.sum((p - t) ** 2, 1) for (p, t) in zip(pred, self.target)]

        return diffs

    def forward(self, pred):
        diffs = self.diffs_map(pred)

        # Spatial average per feature map (different resolutions)
        diffs = [diff.mean([1, 2]) for diff in diffs]

        return sum(diffs).mean(0)

    class _FeatureExtractor(nn.Module):
        def __init__(self):
            super(CachedLPIPS._FeatureExtractor, self).__init__()
            vgg_pretrained = torchvision.models.vgg16(pretrained=True).features

            # Split at the maxpools, before ReLU
            self.breakpoints = [0, 3, 8, 15, 22, 29]

            # Split after ReLU
            # self.breakpoints = [0, 4, 9, 16, 23, 30]

            for i, b in enumerate(self.breakpoints[:-1]):
                ops = nn.Sequential()
                for idx in range(b, self.breakpoints[i+1]):
                    op = vgg_pretrained[idx]
                    ops.add_module(str(idx), op)
                self.add_module("group{}".format(i), ops)

            # No gradients
            for p in self.parameters():
                p.requires_grad = False

            # Torchvision's normalization: <https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101>
            self.register_buffer("shift", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("scale", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        def forward(self, x):
            feats = []
            x = (x - self.shift) / self.scale
            for idx in range(len(self.breakpoints)-1):
                m = getattr(self, "group{}".format(idx))
                x = m(x)
                feats.append(x)

            return feats

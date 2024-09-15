import torch
import math

from gaussian_splatting.datatypes.gaussians import Gaussian2D
from gsplat.rasterize import rasterize_gaussians

TILE_SIZE = 16


class GSplatGaussianSplattingRenderer:
    def __init__(self, M, N):
        self.M = M
        self.N = N
        self.tile_size = TILE_SIZE

    def __call__(
        self,
        gaussians: Gaussian2D,
        background: torch.Tensor,
        depth=None,
        active: torch.Tensor = None,
    ):
        device = gaussians.position.device

        if background is None:
            background = torch.zeros(3, device=device)

        if active is not None:
            gaussians = gaussians[active]

        icov = gaussians.inv_covariance
        cov = gaussians.covariance
        det = cov[:, 1, 1] * cov[:, 0, 0] - cov[:, 0, 1] * cov[:, 1, 0]

        b = 0.5 * (cov[:, 0, 0] + cov[:, 1, 1])
        v1 = b + torch.clamp(b * b - det, min=1e-1).sqrt()
        v2 = b - torch.clamp(b * b - det, min=1e-1).sqrt()
        radii = torch.ceil(3.0 * torch.max(v1, v2).sqrt())

        tile_center = gaussians.position / float(TILE_SIZE)

        tile_radius = radii / float(TILE_SIZE)

        tile_bounds_x = math.ceil(self.N / float(TILE_SIZE))
        tile_bounds_y = math.ceil(self.M / float(TILE_SIZE))

        bb_min_x = torch.clamp((tile_center[:, 0] - tile_radius).int(), min=0, max=tile_bounds_x)
        bb_max_x = torch.clamp(
            (tile_center[:, 0] + tile_radius + 1).int(), min=0, max=tile_bounds_x
        )
        bb_min_y = torch.clamp((tile_center[:, 1] - tile_radius).int(), min=0, max=tile_bounds_y)
        bb_max_y = torch.clamp(
            (tile_center[:, 1] + tile_radius + 1).int(), min=0, max=tile_bounds_y
        )

        num_tiles_hit = torch.clamp((bb_max_x - bb_min_x) * (bb_max_y - bb_min_y), min=0.0)
        conics = torch.stack((icov[:, 0, 0], icov[:, 0, 1], icov[:, 1, 1]), dim=-1)

        if depth is None:
            depth = torch.arange(len(gaussians), device=device)

        out, alpha = rasterize_gaussians(
            gaussians.position.float(),
            depth.float(),
            radii.int(),
            conics.float(),
            num_tiles_hit.int(),
            gaussians.color.float(),
            gaussians.opacity.float().view(-1, 1),
            background,
            int(self.M),
            int(self.N),
            int(TILE_SIZE),
            return_alpha=True,
        )

        return out, alpha

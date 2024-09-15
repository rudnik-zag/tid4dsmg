import torch
import torch.nn.functional as F

from typing import Union, Optional

from ..datatypes import PinholeCamera
from ..datatypes import CloudSegments3D, CloudSegments2D
from ..datatypes import Gaussian3D, RiggedGaussian3D

from ..utils.culling import get_visible_gaussians

# from .backends.tiled import TaichiTiledGaussianSplattingRenderer
from .backends.gsplat_renderer import GSplatGaussianSplattingRenderer


class BaseSplatter:
    """Splatter that works with Taichi-based rendering backends

    Splatters peform projection and visibility culling. It also handles
    depth ordering of segments with segments that come earlier in the CloudSegment
    being drawn first in back to front order.
    """

    def __init__(self, renderer):
        self.renderer = renderer
        self.M = renderer.M
        self.N = renderer.N

    def splat(
        self,
        camera: PinholeCamera,
        gaussians: CloudSegments3D,
        near_plane: float = 0.01,
        train: bool = False,
    ) -> tuple[torch.Tensor, CloudSegments2D, int]:
        """Splat 3D Gaussians to an image

        Args:
            camera (PinholeCamera): The camera to splat to
            gaussians (CloudSegments3D): 3D Gaussians to splat.
            near_plane (float, optional): depth cut-off for near-plane culling. Defaults to 0.01.
            train (bool, optional): enable training mode. If true, makes sure that gradients that
                                    are needed for densification are retained.

        Returns:
            tuple[torch.Tensor, CloudSegments2D, int]: splatted image, final alpha value,
                                                       2D Gaussians, number of visible gaussians
        """
        draw_list = []
        gauss2d_segments = CloudSegments2D()
        num_active = 0

        for segment_name, gauss3d in gaussians.items():
            gauss2d, depth = camera.project(gauss3d)

            # retain position gradients if in training mode
            # this is needed for densification
            if train:
                gauss2d._position.retain_grad()

            gauss2d_segments[segment_name] = gauss2d

            # cull based on visibility
            active = get_visible_gaussians(gauss2d, depth, (self.N, self.M), near_plane=near_plane)
            num_active += len(active)

            # depth sort segment
            gauss2d_active = gauss2d[active]
            depth_active = depth[active]

            sorted_idx = torch.argsort(depth_active, stable=True)

            draw_list.append(gauss2d_active[sorted_idx])

        # draw
        pixels, alpha = self.renderer(torch.cat(draw_list, 0))

        return pixels, alpha, gauss2d_segments, num_active


class GSplatRenderer:
    """Splatter that works with the GSplat rendering backend

    Splatters peform projection and visibility culling. It also handles
    depth ordering of segments with segments that come earlier in the CloudSegment
    being drawn first in back to front order.
    """

    def __init__(self, M: int, N: int, device):
        self.renderer_backend = GSplatGaussianSplattingRenderer(M, N)

        self.M = M
        self.N = N
        self.device = device
        self.depth_offset = 1.0

    def render(
        self,
        camera: PinholeCamera,
        gauss3d: Union[RiggedGaussian3D, Gaussian3D],
        background: torch.Tensor,
        base_depth: Optional[torch.Tensor] = None,
        near_plane: float = 0.01,
        train: bool = False,
    ):  # -> tuple[torch.Tensor, torch.Tensor, CloudSegments2D, int]:
        """Splat 3D Gaussians to an image

        Args:
            camera (PinholeCamera): The camera to splat to
            gaussians (CloudSegments3D): 3D Gaussians to splat.
            background (torch.Tensor): background color
            near_plane (float, optional): depth cut-off for near-plane culling. Defaults to 0.01.
            train (bool, optional): enable training mode. If true, makes sure that gradients that
                                    are needed for densification are retained.
        Returns:
            tuple[torch.Tensor, CloudSegments2D, int]: splatted image, final alpha value,
                                                       2D Gaussians, number of visible gaussians
        """
        num_active = 0

        # if isinstance(gauss3d, RiggedGaussian3D):
        #     gauss3d = cull_backfaces(gauss3d, camera)

        # gauss2d, depth = camera.project(gauss3d)
        gauss2d, depth = camera.project(gauss3d)

        # retain position gradients if in training mode
        # this is needed for densification
        if train:
            gauss2d._position.retain_grad()

        # cull based on visibility
        active = get_visible_gaussians(gauss2d, depth, (self.N, self.M), near_plane=near_plane)

        # depth sort segment
        gauss2d_active = gauss2d[active]
        depth_active = depth[active]

        # cull based on base depth
        # we only draw gaussians that are in front of the base geometry
        if base_depth is not None:
            ndc = gauss2d_active.position / torch.tensor([self.N, self.M], device=self.device).view(
                1, 2
            )
            ndc = 2 * ndc - 1

            base_depth_at_position = F.grid_sample(
                base_depth.view(1, 1, *base_depth.shape), ndc.view(1, 1, -1, 2)
            ).view(-1)
            base_depth_at_position[base_depth_at_position == -1] = 1000000
            non_occluded = depth_active <= base_depth_at_position + self.depth_offset
            gauss2d_active = gauss2d_active[non_occluded]
            depth_active = depth_active[non_occluded]

        num_active = len(gauss2d_active)

        # draw_list.append(gauss2d_active)
        # draw_list_depth.append(depth_active + min_depth)

        # if len(draw_list_depth[-1] > 0):
        #     min_depth = torch.max(draw_list_depth[-1] + 1e-4)

        pixels, alpha = self.renderer_backend(gauss2d_active, background, depth=depth_active)

        return pixels, 1 - alpha, (gauss2d, num_active)

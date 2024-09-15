import torch
import torch.nn.functional as F

from typing import Union

from dataclasses import dataclass

from .gaussians import Gaussian2D, Gaussian3D


@dataclass
class PinholeCameraConfig:
    """Representation of the parameters of a Pinhole camera.

    Offers sanity checks and convenience functions for reading configs and
    manipulating the camera intrinsics.
    """

    name: str
    focal_x: float
    focal_y: float
    pp_x: float
    pp_y: float
    res_x: int
    res_y: int
    R: torch.Tensor
    t: torch.Tensor

    def __post_init__(self):
        """Sanity checks fore initialization"""
        assert self.focal_x > 0
        assert self.focal_y > 0

        assert self.res_x > 0
        assert self.res_y > 0

        assert self.pp_x > 0 and self.pp_x < self.res_x
        assert self.pp_y > 0 and self.pp_y < self.res_y

        assert self.R.shape == (3, 3)
        assert self.t.shape == (3,)

    @staticmethod
    def from_matrices(
        name: str, res_x: int, res_y: int, K: torch.Tensor, world_2_cam: torch.Tensor
    ):
        """Instantiate config from from instrinsic and extrinsic matrics

        Args:
            name (str): name of the camera
            res_x (int): horizontal resolution
            res_y (int): vertical resolution
            K (torch.Tensor): intrinsic matrix [3, 3]
            world_2_cam (torch.Tensor): world to camera transform [4, 4]

        Returns:
            PinholeCameraConfig: the configuration
        """
        return PinholeCameraConfig(
            name,
            K[0, 0].item(),
            K[1, 1].item(),
            K[0, -1].item(),
            K[1, -1].item(),
            res_x,
            res_y,
            world_2_cam[:3, :3],
            world_2_cam[:3, -1],
        )

    def resize(self, factor: float):
        """Resize camera by factor.

        Rescales resolution, principal point, and focal length.

        Return:
            PinholeCameraConfig: the resized configuraiont
        """
        assert (
            self.res_x % factor == 0 and self.res_y % factor == 0
        ), f"""Unable to resize camera '{self.name}':
             resolution ({self.res_x, self.res_y}) is not divisible by {factor}"""

        fx = self.focal_x / factor
        fy = self.focal_y / factor
        pp_x = self.pp_x / factor
        pp_y = self.pp_y / factor

        res_x = int(self.res_x / factor)
        res_y = int(self.res_y / factor)

        return PinholeCameraConfig(
            self.name, fx, fy, pp_x, pp_y, res_x, res_y, self.R.clone(), self.t.clone()
        )


@dataclass
class PinholeCamera:
    """Representation of a Pinhole camera.

    Used to project 3D Gaussians to the 2D image plane
    """

    position: torch.Tensor  # 3D position (x, y, z)
    rotation: torch.Tensor  # 3D rotation (SO(3))
    focal_x: float  # focal length x
    focal_y: float  # focal length y
    pp_x: float  # principal point x
    pp_y: float  # principal point y
    res_x: float  # horizontal resolution
    res_y: float  # vertical resolution

    def __post_init__(self):
        """Sanity checks for initialization"""
        assert self.position.dim() == 2
        assert self.position.shape[0] == 1
        assert self.position.shape[1] == 3

        assert self.rotation.dim() == 2
        assert self.rotation.shape[0] == 3
        assert self.rotation.shape[1] == 3

    def to(self, device: torch.device):
        """Push camera to device"""
        self.position = self.position.to(device)
        self.rotation = self.rotation.to(device)
        return self

    @property
    def K(self) -> torch.Tensor:
        """Get intrinsic camera matrix

        Returns:
            torch.Tensor: intrinsic camera matrix [3, 3]
        """
        K = torch.eye(3, device=self.position.device)
        K[0, 0] = self.focal_x
        K[1, 1] = self.focal_y
        K[0, -1] = self.pp_x
        K[1, -1] = self.pp_y

        return K

    @property
    def T_world2cam(self) -> torch.Tensor:
        """Get world to camera transform

        Returns:
            torch.Tensor: transformation matrix [4, 4]
        """
        T = torch.eye(4, device=self.position.device)
        T[:3, :3] = self.rotation
        T[:3, -1] = self.position
        return T

    @property
    def T_cam2world(self) -> torch.Tensor:
        """Get camera to world transform

        Returns:
            torch.Tensor: transformation matrix [4, 4]
        """
        T = self.T_world2cam
        return torch.linalg.inv(T)

    @staticmethod
    def from_camera_config(config: PinholeCameraConfig):
        """Construct camera from config

        Args:
            config (PinholeCameraConfig): the config

        Returns:
            PinholeCamera: the camera object
        """
        return PinholeCamera(
            config.t.unsqueeze(0),
            config.R,
            config.focal_x,
            config.focal_y,
            config.pp_x,
            config.pp_y,
            config.res_x,
            config.res_y,
        )

    @property
    def center(self) -> torch.Tensor:
        """Get camera center

        Returns:
            torch.Tensor: camera center in world coordinates
        """
        return -self.position @ self.rotation

    def world2cam(self, vertices: torch.Tensor) -> torch.Tensor:
        """Transform vertices from world to camera coordinate system

        Args:
            locations (torch.Tensor): vertices [n_verts, 3]

        Returns:
            torch.Tensor: vertices in camera coordinate system [n_verts, 3]
        """
        return vertices @ self.rotation.transpose(-2, -1) + self.position

    def project(
        self, gaussians: Gaussian3D
    ) -> tuple[Gaussian2D, torch.Tensor]:
        """Projects a 3D Gaussian to the camera image plane

        Args:
            gaussians (Union[Gaussian3D, RiggedGaussian3D]): input Gaussians

        Returns:
            tuple[Gaussian2D, torch.Tensor]: Resulting 2D Gaussians and their depth in
                                             the camera view
        """
        n_gaussians = len(gaussians)

        g_position = gaussians.position
        g_opacity = gaussians.opacity
        g_L_cov = gaussians.L_covariance
        view_direction = F.normalize(self.center - g_position)
        g_color = gaussians.color(view_direction)

        cam_pos = self.world2cam(g_position)

        depth = cam_pos[:, 2]
        inv_depth = 1.0 / depth

        J = self._jacobian(cam_pos, inv_depth)
        cov2D_L = J @ (self.rotation @ g_L_cov)

        img_pos = torch.stack(
            (
                self.focal_x * cam_pos[:, 0] * inv_depth + self.pp_x,
                self.focal_y * cam_pos[:, 1] * inv_depth + self.pp_y,
            ),
            -1,
        )

        return (
            Gaussian2D(
                img_pos,
                cov2D_L,
                g_opacity,
                g_color,
                batch_size=[n_gaussians],
            ),
            depth,
        )

    def _jacobian(self, pos_cam: torch.Tensor, inv_depth: torch.Tensor) -> torch.Tensor:
        """Compute Jacobian of perspective projection

        Args:
            pos_cam (torch.Tensor): position in camera coordinate frame
            inv_depth (torch.Tensor): inverse depth

        Returns:
            torch.Tensor: associated Jacobian J [n_gaussians, 2, 3]
        """
        # TODO: Optimize once the rasterizer is optimized
        zero = torch.zeros(len(pos_cam), device=pos_cam.device)

        inv_z_square = inv_depth.square()
        J_1 = torch.stack(
            [
                self.focal_x * inv_depth,
                zero,
                -(self.focal_x * pos_cam[:, 0]) * inv_z_square,
            ],
            -1,
        )
        J_2 = torch.stack(
            [
                zero,
                self.focal_y * inv_depth,
                -(self.focal_y * pos_cam[:, 1]) * inv_z_square,
            ],
            -1,
        )
        J = torch.cat((J_1, J_2), -1).view(-1, 2, 3)

        return J

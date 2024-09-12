import torch

import numpy as np
import kornia as K

from tensordict import tensorclass
from collections import namedtuple

from gaussian_splatting.utils.spherical_harmonics import (
    rsh_cart_0,
    rsh_cart_1,
    rsh_cart_2,
)

SphericalHarmonicsFun = namedtuple("SphericalHarmonicsFun", "num_coeff fun")

sh_map = {
    0: SphericalHarmonicsFun(1, rsh_cart_0),
    1: SphericalHarmonicsFun(4, rsh_cart_1),
    2: SphericalHarmonicsFun(9, rsh_cart_2),
}

sh_fun_map = {sh_fun.num_coeff: sh_fun.fun for sh_fun in sh_map.values()}
sh_order_map = {sh_fun.num_coeff: order for order, sh_fun in sh_map.items()}


@tensorclass
class Gaussian3D:
    """Represents a set of Gaussian in 3D space.

    3D Gaussians are defined by position, covariance, opacity, and color.
    Covariance is indirectly represented by rotation and scaling. Color
    is view-dependend and represented using spherical harmonics of order at most 2

    Atttributes:
        _position: 3D position
        _rotation: Unnormalized uaternion that represents the rotation
        _log_scaling: Logarithm of scaling (log to ensure positive values)
        _opacity: Raw opacity value. Passed through a sigmoid to to ensure [0, 1] range
        _sh_coeff: Spherical Harmonics coefficients. Order of the spherical harmonics
                  is detected automatically from shape. Maximum supported order is 2.
    """

    _position: torch.Tensor  # 3D position (x,y,z)
    _rotation: torch.Tensor  # Rotation quaternion (q1, q2, q3, q4)
    _log_scaling: torch.Tensor  # 3D scaling (sx, sy, sz)
    _opacity: torch.Tensor  # Blend weight (alpha)
    _sh_coeff: torch.Tensor  # Spherical harmonics coefficients (n_coeff, 3)

    def __post_init__(self):
        """Sanity check shapes and detect order of spherical harmonics"""
        assert self._position.dim() == 2
        assert self._rotation.dim() == 2
        assert self._log_scaling.dim() == 2
        assert self._opacity.dim() == 2
        assert self._sh_coeff.dim() == 3

        assert self._position.shape[-1] == 3
        assert self._rotation.shape[-1] == 4
        assert self._log_scaling.shape[-1] == 3
        assert self._opacity.shape[-1] == 1
        assert self._sh_coeff.shape[-1] == 3

        supported_sh_orders = [num_coeff for num_coeff in sh_fun_map.keys()]
        assert self._sh_coeff.shape[-2] in supported_sh_orders

    @staticmethod
    def random_init_at(
        position: torch.Tensor,
        covariance_scale: float = 1.0,
        opacity_scale: float = 1.0,
        sh_order: int = 2,
    ):
        """Initialize Gaussians at a given position.

        Rotation, scaling, opacity, and color are initialized randomly to sane
        values.

        Args:
            position (torch.Tensor): position of initialization
            covariance_scale (float, optional): uniform scaling multiplier for covariance.
                                                Defaults to 1.0.
            opacity_scale (float, optional): uniform scaling factor for opacity. Defaults to 1.0.
            sh_order (int, optional): order of spherial harmonics. Defaults to 2.

        Returns:
           Gaussian3D: randomly initialized 3D Gaussians at the given position
        """
        n_gaussians = len(position)

        rotation = torch.randn(n_gaussians, 4)
        log_scaling = torch.log(covariance_scale * torch.randn(n_gaussians, 3))
        opacity = (
            opacity_scale * torch.rand(n_gaussians, 1) / np.sqrt(n_gaussians)
            + 1.0 / 255.0
        )
        opacity = -4 + opacity_scale * torch.rand(n_gaussians, 1) / np.sqrt(
            n_gaussians
        )  # + 1e-4

        sh_coeff = torch.randn(n_gaussians, sh_map[sh_order].num_coeff, 3)

        return Gaussian3D(
            position, rotation, log_scaling, opacity, sh_coeff, batch_size=[n_gaussians]
        )

    @torch.no_grad()
    def reset_opacity(self, opacity_scale: float = 1.0):
        """Resets opacity values to be close to 0.

        This is sometimes useful to periodically execute to get rid of floaters.

        Args:
            opacity_scale (float, optional): uniform scaling factor for opacities. Defaults to 1.0.
        """
        self._opacity.fill_(-4 * opacity_scale)
        self._opacity.requires_grad_(True)

    @property
    def opacity(self) -> torch.Tensor:
        """Get opacity squashed to the range [0, 1]

        This should be the default accessor to the opacity values.

        Returns:
            torch.Tensor: squashed opacity [n_gaussians, 1]
        """
        return torch.sigmoid(self._opacity)

    def color(self, view_direction: torch.Tensor) -> torch.Tensor:
        """Compute color of Gaussian

        Evaluates spherical harmonics for given view directions. View direction is assumed
        to be normalized to length 1.

        Args:
            view_direction (torch.Tensor): per-gaussian view directions [n_gaussians, 3]

        Returns:
            torch.Tensor: Resulting view-dependend color [n_gaussians, 3]
        """
        basis = sh_fun_map[self._sh_coeff.shape[-2]](view_direction)
        color = torch.bmm(basis.unsqueeze(1), self._sh_coeff).squeeze(1)
        return torch.sigmoid(color)
        # return torch.clamp(color, min=0.0, max=1.0)

    @property
    def position(self) -> torch.Tensor:
        """Get position

        Returns:
            torch.Tensor: position [n_gaussians, 3]
        """
        return self._position

    @property
    def quaternion(self) -> torch.Tensor:
        """Get rotation as normalized quaternion

        Returns:
            torch.Tensor: rotation as normalized quaternion [n_gaussians, 4]
        """
        return K.geometry.conversions.normalize_quaternion(self._rotation)

    @property
    def rotation_matrix(self) -> torch.Tensor:
        """Get rotation as rotation matrix

        Returns:
            torch.Tensor: rotation as rotation matrix [n_gaussians, 3, 3]
        """
        return K.geometry.conversions.quaternion_to_rotation_matrix(self.quaternion)

    @property
    def scaling(self) -> torch.Tensor:
        """Convert log scaling to proper scaling

        Returns:
            torch.Tensor: proper scline in 3D space [n_gaussians, 3]
        """
        return torch.exp(self._log_scaling)

    @property
    def L_covariance(self) -> torch.Tensor:
        """Get factorized covariance

        Constructs factor of covariance from rotation and scale. Final covariance
        can be computed as

        C = L' L

        This function computes L. We opt for this representation to ensure that
        the matrix always stays positive semi-definite (NOTE: wouldn't be necessary
        with our scaling + rotation representation) and to reduce compute.

        See also: method 'covariance' below.

        Returns:
            torch.Tensor: L such that covariance = L' L  [n_gaussians, 3, 3]
        """
        S = torch.diag_embed(self.scaling)
        R = self.rotation_matrix
        return R @ S

    @property
    def covariance(self) -> torch.Tensor:
        """Construct the full covariance.

        Returns:
            torch.Tensor: covariance matrices [n_gaussians, 3, 3]
        """
        cov = self.L_covariance @ self.L_covariance.transpose(-2, -1)
        return (
            cov  # + torch.eye(3, device=self.L_covariance.device).unsqueeze(0) * 1e-6
        )

    @property
    def packed_gradient(self) -> torch.Tensor:
        """Return all gradients as a single packed tensor

        Returns:
            torch.Tensor: gradients
        """
        all_grads = torch.cat(
            (
                self._position.grad,
                self._rotation.grad,
                self._log_scaling.grad,
                self._sh_coeff.grad.view(len(self), -1),
            ),
            dim=-1,
        )
        return all_grads

    def requires_grad_(self, val: bool):
        """Enable or disable gradient computation for internal parameters

        Calling this wiht val=True will make the internal properties a leaf node
        in the autograd graph, which enables optimization with the standard
        PyTorch facilities.

        Args:
            val (bool): enable or disable gradient computation
        """
        self._position.requires_grad_(val)
        self._rotation.requires_grad_(val)
        self._log_scaling.requires_grad_(val)
        self._opacity.requires_grad_(val)
        self._sh_coeff.requires_grad_(val)


@tensorclass
class Gaussian2D:
    """Represents a set of Gaussian in 2D space.

    2D Gaussians are defined by position, covariance, opacity, and color. Note that
    initializers are responsible to ensure valide values (eg. opacity and color are
    NOT clamped to valid ranges, instead need to provided as clamped values on
    initialization)

    Atttributes:
        _position: 2D position
        _L_covariance: Factorized covariance L. Covariance C = L' L
        _opacity: Raw opacity
        _color: Raw RGB values
    """

    _position: torch.Tensor  # 2D position (x,y)
    _L_covariance: torch.Tensor  # full covariance matrix 2d [[c1, c2], [c3, c4]]
    _opacity: torch.Tensor  # Blend weight (alpha)
    _color: torch.Tensor  # RGB color (r, g, b)

    def __post_init__(self):
        """Sanity check of shapes"""
        assert self._position.dim() == 2
        assert self._L_covariance.dim() == 3
        assert self._opacity.dim() == 2
        assert self._color.dim() == 2

        assert self._position.shape[-1] == 2
        assert self._opacity.shape[-1] == 1
        assert self._color.shape[-1] == 3

    @staticmethod
    def random_init_at(position: torch.Tensor, covariance_scale: float = 1.0):
        """Randomly initialize Gaussians at given position.

        Covariance, opacity and color are initialized randomly to sane values. This function
        is meant for testing purposes, as 2D Gaussians never need to be initialized randomly
        in practice.

        Args:
            position (torch.Tensor): position of initalization
            covariance_scale (float, optional): uniform scaling multiplier for covariance.
                                                Defaults to 1.0.

        Returns:
            Gaussian2D: randomly initialized 2D Gaussians at the given position.
        """
        n_gaussians = len(position)
        L_covariance = covariance_scale * torch.randn(n_gaussians, 2, 2)
        opacity = torch.rand(n_gaussians, 1) / np.sqrt(n_gaussians) + 1e-4

        color = torch.rand(n_gaussians, 3)

        return Gaussian2D(
            position, L_covariance, opacity, color, batch_size=[n_gaussians]
        )

    @property
    def covariance(self) -> torch.Tensor:
        """Compute 2D covariance.

        Adds a small diagonal regularization to ensure that the 2D Gaussians
        don't shrink to be too small in screen space and to avoid numerical
        issues.

        Returns:
            torch.Tensor: regularized covariance [n_gaussians, 2, 2]
        """
        cov = self._L_covariance @ self._L_covariance.transpose(-2, -1)
        return cov + torch.eye(2, device=self._L_covariance.device).unsqueeze(0) * 1e-4

    @property
    def position(self) -> torch.Tensor:
        return self._position

    @property
    def opacity(self) -> torch.Tensor:
        return self._opacity

    @property
    def color(self) -> torch.Tensor:
        return self._color

    @property
    def inv_covariance(self) -> torch.Tensor:
        """Get inverse covariance (= precision matrix)

        Returns:
            torch.Tensor: precision matrix [n_gaussians, 2, 2]
        """
        cov = self.covariance

        inv_cov = torch.stack(
            (cov[:, 1, 1], -cov[:, 1, 0], -cov[:, 0, 1], cov[:, 0, 0]), -1
        )
        inv_cov = inv_cov.reshape(-1, 2, 2) / self.determinant.view(-1, 1, 1)

        return inv_cov

    @property
    def determinant(self) -> torch.Tensor:
        """Compute determinant of covariance matrix

        Returns:
            torch.Tensor: determinant [n_gaussians]
        """
        cov = self.covariance
        det = cov[:, 1, 1] * cov[:, 0, 0] - cov[:, 0, 1] * cov[:, 1, 0]

        return det

    @property
    def packed(self) -> torch.Tensor:
        """Pack all attributes into a single array

        Returns:
            torch.Tensor: All attributes that are relevant for rendering [n_gaussians, 10]
        """
        inv_cov = self.inv_covariance
        return torch.cat(
            (
                self.position,
                inv_cov[:, 0, 0].unsqueeze(-1),
                inv_cov[:, 0, 1].unsqueeze(-1),
                inv_cov[:, 1, 1].unsqueeze(-1),
                self.opacity.unsqueeze(-1),
                self.color,
            ),
            -1,
        )

    @property
    def packed_geometry(self) -> torch.Tensor:
        """Pack all geometric attributes into a single array

        Returns:
            torch.Tensor: pack position and inverse covariance into a single array [n_gaussians, 6]
        """
        inv_cov = self.inv_covariance
        return torch.cat(
            (
                self.position,
                inv_cov[:, 0, 0].unsqueeze(-1),
                inv_cov[:, 0, 1].unsqueeze(-1),
                inv_cov[:, 1, 1].unsqueeze(-1),
            ),
            -1,
        )

    def requires_grad_(self, val: bool):
        """Enable or disable gradient compuatation for internal parameters

        Calling this wiht val=True will make the internal properties a leaf node
        in the autograd graph, which enables optimization with the standard
        PyTorch facilities.

        Args:
            val (bool): enable or disable gradient computation
        """
        self._position.requires_grad_(val)
        self._L_covariance.requires_grad_(val)
        self._opacity.requires_grad_(val)
        self._color.requires_grad_(val)

import math

import numpy as np

import torch
import torch.nn.functional as F

import OpenGL.GL as gl
import OpenGL.GLU as glu


class OrbitCamera:
    def __init__(self, center, offset=150.0, rotation=[0.0, 0.0]):
        self.device = center.device
        self.center = center.clone()
        self.init_center = center.clone()

        self.radius = torch.tensor(offset, device=self.device)
        self.theta = torch.tensor(rotation[0], device=self.device)
        self.phi = torch.tensor(rotation[1], device=self.device)

        self.init_radius = self.radius.clone()
        self.init_theta = self.theta.clone()
        self.init_phi = self.phi.clone()

        self.world_up = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        # This works for troop:
        # self.world_up = torch.tensor([-1.0, 0.0, 0.0], device=self.device)

        self.strafe = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        self.updated = True

    @property
    def position(self):
        rad2deg = 180.0 / math.pi

        theta = self.theta / rad2deg
        phi = self.phi / rad2deg

        x = self.radius * torch.sin(theta) * torch.cos(phi)
        y = self.radius * torch.sin(theta) * torch.sin(phi)
        z = self.radius * torch.cos(theta)

        return torch.stack((x, y, z)) + self.center

    @property
    def rotation_matrix(self):
        forward = F.normalize((self.center - self.position).unsqueeze(0)).view(-1)
        side = F.normalize(torch.cross(self.world_up, forward).unsqueeze(0)).view(-1)
        look_up = torch.cross(forward, side)

        return torch.stack((side, look_up, forward), 0)

    def reset(self):
        self.center = self.init_center.clone()
        self.radius = self.init_radius.clone()
        self.theta = self.init_theta.clone()
        self.phi = self.init_phi.clone()
        self.strafe = torch.zeros_like(self.strafe)
        self.updated = True

    def mouse_motion(self, dx, dy):
        sensitivity = 0.1
        dx = float(dx)
        dy = float(dy)

        self.theta -= dx * sensitivity
        self.phi += 5.0 * dy * sensitivity

        self.theta = self.theta % 360
        self.phi = self.phi % 360
        self.updated = True

    def update(self, dt):
        motion_vector = self.get_motion_vector()
        speed = 10 * dt

        # self.center = [x + y * speed for x, y in zip(self.position, motion_vector)]
        if torch.linalg.norm(motion_vector) > 0.0:
            self.center += speed * motion_vector
            self.updated = True

    def get_motion_vector(self):
        x, y, z = self.strafe.unbind()
        if x or z:
            strafe = torch.rad2deg(torch.atan2(x, z))
            yaw = self.theta
            x_angle = torch.deg2rad(yaw + strafe)
            x = torch.cos(x_angle)
            z = torch.sin(x_angle)

        return torch.tensor([x, y, z], device=self.device)

import math
import numpy as np

import torch

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from sh_utils import eval_sh, SH2RGB, RGB2SH

from gaussian_model_4d import GaussianModel, BasicPointCloud

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 1 / tanHalfFovX
    P[1, 1] = 1 / tanHalfFovY
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


class MiniCam:
    def __init__(self, c2w, width, height, fovy, fovx, znear, zfar, time=0, gs_convention=True):
        # c2w (pose) should be in NeRF convention.

        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar

        w2c = np.linalg.inv(c2w)

        if gs_convention:
            # rectify...
            w2c[1:3, :3] *= -1
            w2c[:3, 3] *= -1

        self.world_view_transform = torch.tensor(w2c).transpose(0, 1).cuda()
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = -torch.tensor(c2w[:3, 3]).cuda()

        self.time = time


class Renderer:
    def __init__(self, opt, sh_degree=3, white_background=True, radius=1):
        
        self.sh_degree = sh_degree
        self.white_background = white_background
        self.radius = radius
        self.opt = opt
        self.T = self.opt.batch_size

        self.gaussians = GaussianModel(sh_degree, opt.deformation)

        self.bg_color = torch.tensor(
            [1, 1, 1] if white_background else [0, 0, 0],
            dtype=torch.float32,
            device="cuda",
        )
        self.means3D_deform_T = None
        self.opacity_deform_T = None
        self.scales_deform_T = None
        self.rotations_deform_T = None


    
    def initialize(self, input=None, num_pts=5000, radius=0.5):
        # load checkpoint
        if input is None:
            # init from random point cloud
            
            phis = np.random.random((num_pts,)) * 2 * np.pi
            costheta = np.random.random((num_pts,)) * 2 - 1
            thetas = np.arccos(costheta)
            mu = np.random.random((num_pts,))
            radius = radius * np.cbrt(mu)
            x = radius * np.sin(thetas) * np.cos(phis)
            y = radius * np.sin(thetas) * np.sin(phis)
            z = radius * np.cos(thetas)
            xyz = np.stack((x, y, z), axis=1)
            # xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3

            shs = np.random.random((num_pts, 3)) / 255.0
            pcd = BasicPointCloud(
                points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
            )
            # self.gaussians.create_from_pcd(pcd, 10)
            self.gaussians.create_from_pcd(pcd, 10, 1)
        elif isinstance(input, BasicPointCloud):
            # load from a provided pcd
            self.gaussians.create_from_pcd(input, 1)
        else:
            # load from saved ply
            self.gaussians.load_ply(input)

    def prepare_render(
        self,
    ):
        means3D = self.gaussians.get_xyz
        opacity = self.gaussians._opacity
        scales = self.gaussians._scaling
        rotations = self.gaussians._rotation

        means3D_T = []
        opacity_T = []
        scales_T = []
        rotations_T = []
        time_T = []

        for t in range(self.T):
            time = torch.tensor(t).to(means3D.device).repeat(means3D.shape[0],1)
            time = ((time.float() / self.T) - 0.5) * 2

            means3D_T.append(means3D)
            opacity_T.append(opacity)
            scales_T.append(scales)
            rotations_T.append(rotations)
            time_T.append(time)

        means3D_T = torch.cat(means3D_T)
        opacity_T = torch.cat(opacity_T)
        scales_T = torch.cat(scales_T)
        rotations_T = torch.cat(rotations_T)
        time_T = torch.cat(time_T)


        means3D_deform_T, scales_deform_T, rotations_deform_T, opacity_deform_T = self.gaussians._deformation(means3D_T, scales_T, 
                                                            rotations_T, opacity_T,
                                                            time_T) #  time is not none
        self.means3D_deform_T = means3D_deform_T.reshape([self.T, means3D_deform_T.shape[0]//self.T, -1])
        self.opacity_deform_T = opacity_deform_T.reshape([self.T, means3D_deform_T.shape[0]//self.T, -1])
        self.scales_deform_T = scales_deform_T.reshape([self.T, means3D_deform_T.shape[0]//self.T, -1])
        self.rotations_deform_T = rotations_deform_T.reshape([self.T, means3D_deform_T.shape[0]//self.T, -1])
        

    def render(
        self,
        viewpoint_camera,
        scaling_modifier=1.0,
        bg_color=None,
        override_color=None,
        compute_cov3D_python=False,
        convert_SHs_python=False,
    ):
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                self.gaussians.get_xyz,
                dtype=self.gaussians.get_xyz.dtype,
                requires_grad=True,
                device="cuda",
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.bg_color if bg_color is None else bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.gaussians.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = self.gaussians.get_xyz
        time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
        time = ((time.float() / self.T) - 0.5) * 2

        means2D = screenspace_points
        opacity = self.gaussians._opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if compute_cov3D_python:
            cov3D_precomp = self.gaussians.get_covariance(scaling_modifier)
        else:
            scales = self.gaussians._scaling
            rotations = self.gaussians._rotation

        means3D_deform, scales_deform, rotations_deform, opacity_deform = self.means3D_deform_T[viewpoint_camera.time], self.scales_deform_T[viewpoint_camera.time], self.rotations_deform_T[viewpoint_camera.time], self.opacity_deform_T[viewpoint_camera.time]


        means3D_final =  means3D + means3D_deform
        rotations_final =  rotations + rotations_deform
        scales_final =  scales + scales_deform
        opacity_final = opacity + opacity_deform



        scales_final = self.gaussians.scaling_activation(scales_final)
        rotations_final = self.gaussians.rotation_activation(rotations_final)
        opacity = self.gaussians.opacity_activation(opacity)


        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if colors_precomp is None:
            if convert_SHs_python:
                shs_view = self.gaussians.get_features.transpose(1, 2).view(
                    -1, 3, (self.gaussians.max_sh_degree + 1) ** 2
                )
                dir_pp = self.gaussians.get_xyz - viewpoint_camera.camera_center.repeat(
                    self.gaussians.get_features.shape[0], 1
                )
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(
                    self.gaussians.active_sh_degree, shs_view, dir_pp_normalized
                )
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = self.gaussians.get_features
        else:
            colors_precomp = override_color

        rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
        means3D = means3D_final,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = cov3D_precomp)


        rendered_image = rendered_image.clamp(0, 1)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "image": rendered_image,
            "depth": rendered_depth,
            "alpha": rendered_alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }

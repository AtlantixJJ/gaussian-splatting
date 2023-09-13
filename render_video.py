#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import math
import torch
import numpy as np
from argparse import ArgumentParser
from kaolin.render.camera import Camera
from tqdm import tqdm
import imageio.v2 as iio

from gaussian_renderer import render
from utils.general_utils import safe_state
from scene.cameras import MiniCam, getProjectionMatrix
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel


def kaolin_cam_to_colmap(kaolin_cam2world: torch.Tensor, inplace: bool=True):
    inplace = 0
    colmap_cam2world = kaolin_cam2world if inplace else kaolin_cam2world.clone()
    colmap_cam2world[..., 1] *= -1
    colmap_cam2world[..., 2] *= -1
    return colmap_cam2world


def FOV_to_intrinsics(fov_degrees, device='cpu'):
    """
    Creates a 3x3 camera intrinsics matrix from the camera field of view, specified in degrees.
    Note the intrinsics are returned as normalized by image size, rather than in pixel units.
    Assumes principal point is at image center.
    """

    focal_length = float(1 / (math.tan(fov_degrees * 3.14159 / 360) * 1.414))
    intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device)
    return intrinsics


def get_ffhq_camera(azim, elev):
    """Return a camera with FFHQ intrinsics.
    """
    fov_deg = 14
    FOVx = FOVy = math.pi * fov_deg / 180
    radius = 2.7
    look_at = torch.Tensor([0., 0., 0.]).float()
    up_vec = torch.Tensor([0., 1., 0.]).float()
    proj_matrix = getProjectionMatrix(znear=1e-2, zfar=1e2, fovX=FOVx, fovY=FOVy).transpose(0,1)

    x = radius * math.cos(azim)
    y = radius * math.sin(elev)
    z = radius * math.sin(azim)
    cam_center = torch.Tensor([x, y, z])
    kaolin_camera = Camera.from_args(
        eye=cam_center, at=look_at, up=up_vec,
        width=512, height=512, fov=14 * math.pi / 180)
    kaolin_cam2world = kaolin_camera.extrinsics.inv_view_matrix()[0]
    colmap_cam2world = kaolin_cam_to_colmap(kaolin_cam2world)
    colmap_world2cam = colmap_cam2world.inverse().transpose(0, 1)
    colmap_fullproj = colmap_world2cam @ proj_matrix
    return MiniCam(512, 512, FOVx, FOVy, 1e-2, 1e2, colmap_world2cam.cuda(), colmap_fullproj.cuda())


def write_video(output_path, frames, fps=30):
    """Write a video to output path from the frames.
    """
    writer = iio.get_writer(output_path, 
        format='FFMPEG', mode='I', fps=fps,
        codec='h264',
        pixelformat='yuv420p')
    for f in frames:
        writer.append_data(f)
    writer.close()


def get_rotating_angles(
        n_steps=360,
        n_rot=3,
        elev_low=-math.pi/4,
        elev_high=math.pi/4):
    """Return the elevation and azimus angle of 360 rotation.
    """
    half_steps = n_steps // 2
    rot_steps = n_steps // n_rot
    elevs = np.linspace(elev_low, elev_high, half_steps)
    elevs = np.concatenate([elevs, elevs[::-1]])
    azims = np.concatenate([np.linspace(-np.pi, np.pi, rot_steps)] * n_rot)
    return azims, elevs


def render_rotate(gaussians, pipeline, background):
    frames = []
    azims, elevs = get_rotating_angles()
    for azim, elev in tqdm(zip(azims, elevs), total=azims.shape[0]):
        cam = get_ffhq_camera(azim, elev)
        rendering = render(cam, gaussians, pipeline, background)["render"]
        image = rendering.clamp(0, 1).permute(1, 2, 0) * 255
        frames.append(image.cpu().numpy().astype("uint8"))
    return frames


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--output_path", default="../../expr/gaussian_splatting/rendering", type=str)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    dataset = model.extract(args)
    pipeline = pipeline.extract(args)
    ply_path = os.path.join(dataset.model_path, "point_cloud", f"iteration_{args.iteration}", "point_cloud.ply")
    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.load_ply(ply_path)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    with torch.no_grad():
        frames = render_rotate(gaussians, pipeline, background)
    os.makedirs(args.output_path, exist_ok=True)
    model_name = dataset.model_path.split("/")[-1]
    output_path = f"{args.output_path}/{model_name}_rotate.mp4"
    write_video(output_path, frames)

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
import imageio
import torchvision
import numpy as np
from os import makedirs
from argparse import ArgumentParser
from kaolin.render.camera import Camera
from tqdm import tqdm
import imageio.v2 as iio

from scene import Scene
from gaussian_renderer import render
from utils.general_utils import safe_state
from scene.cameras import MiniCam, getProjectionMatrix
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene.colmap_loader import qvec2rotmat, rotmat2qvec


def normalize_vecs(vectors: torch.Tensor) -> torch.Tensor:
    """
    Normalize vector lengths.
    """
    return vectors / (torch.norm(vectors, dim=-1, keepdim=True))


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


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    fov_deg = 14
    FOVx = FOVy = math.pi * fov_deg / 180
    radius = 2.7
    
    look_at = torch.Tensor([0., 0., 0.]).float()
    up_vec = torch.Tensor([0., 1., 0.]).float()
    proj_matrix = getProjectionMatrix(znear=1e-2, zfar=1e2, fovX=FOVx, fovY=FOVy).transpose(0,1)
    
    writer = iio.get_writer(f"{render_path}/my_video.mp4", 
        format='FFMPEG', mode='I', fps=30,
        codec='h264',
        pixelformat='yuv420p')

    frames = []
    elevs = np.linspace(-np.pi / 4, np.pi / 4, 180)
    elevs = np.concatenate([elevs, elevs[::-1]])
    azims = np.concatenate([np.linspace(-np.pi, np.pi, 120)] * 3)
    for idx, (azim, elev) in enumerate(tqdm(zip(azims, elevs))):
        x = radius * math.cos(azim)
        y = radius * math.sin(elev)
        z = radius * math.sin(azim)
        cam_center = torch.Tensor([x, y, z])
        kaolin_camera = Camera.from_args(eye=cam_center, at=look_at, up=up_vec, width=512, height=512, fov=14 * math.pi / 180)
        kaolin_cam2world = kaolin_camera.extrinsics.inv_view_matrix()[0]
        colmap_cam2world = kaolin_cam_to_colmap(kaolin_cam2world)
        colmap_world2cam = colmap_cam2world.inverse().transpose(0, 1)
        colmap_fullproj = colmap_world2cam @ proj_matrix
        cam = MiniCam(512, 512, FOVx, FOVy, 1e-2, 1e2, colmap_world2cam.cuda(), colmap_fullproj.cuda())
        rendering = render(cam, gaussians, pipeline, background)["render"]

        image = rendering.clamp(0, 1).permute(1, 2, 0) * 255
        frames.append(image.cpu().numpy().astype("uint8"))
        writer.append_data(frames[-1])
    writer.close()
    #imageio.mimsave(f"{render_path}/rotate.mp4", frames)


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
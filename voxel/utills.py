import sys
import torch
import os
import glob
import numpy as np
from tqdm import tqdm
import imageio
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
from pytorch3d.utils import ico_sphere

# io utils
from pytorch3d.io import load_obj, save_obj
from pytorch3d.ops import cubify

# datastructures
from pytorch3d.structures import Meshes, Volumes

# 3D transformations functions
from pytorch3d.transforms import Rotate, Translate

# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    FoVOrthographicCameras, 
    VolumeRenderer,
    NDCGridRaysampler,
    EmissionAbsorptionRaymarcher,
    look_at_view_transform,
    RasterizationSettings, MeshRenderer,
    MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader,
    PointLights, TexturesVertex,SoftPhongShader,
)

from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)

import utills
import cv2

#################### Image Processing Utills  ##################

def get_props(img, img_size = (256, 256)):
    
    rows, cols = img_size
    h0, w0 = (int(rows*0.7) , int(cols*0.7))
    cx0, cy0 = (img_size[0]//2, img_size[1]//2)

    conts, _ = cv2.findContours(img,
                                cv2.RETR_TREE,
                                cv2.CHAIN_APPROX_SIMPLE)    
    conts = sorted(conts,
                   key=lambda ctr: cv2.contourArea(ctr),
                   reverse=True)
    holes = len(conts) -1

    cnt = conts[0]
    x,y,w,h = cv2.boundingRect(cnt)
    cx1 = x+w/2
    cy1 = y+h/2

    sx = w0*1.0/w
    sy = h0*1.0/h
    tx = cx0*(1 - sx) + (cx0 - cx1)
    ty = cy0*(1 - sy) + (cy0 - cy1)

    M = np.float32([[sx, 0, tx],[0, sy, ty]])
    img = cv2.warpAffine(img, M, img_size)

    return img, holes

def get_silhouette_renderer(device, cameras, image_size):

    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

    # Initialize silhouette renderer
    raster_settings = RasterizationSettings(
        image_size=image_size, 
        blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma, 
        faces_per_pixel=100, 
    )
    silhouette_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=SoftSilhouetteShader(blend_params=blend_params)
    )

    return silhouette_renderer

def get_phong_renderer(device, cameras, image_size):

    # Initialize phong render
    raster_settings = RasterizationSettings(
        image_size=image_size, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
    )
    # We can add a point light in front of the object. 
    lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))
    phong_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(device=device, cameras=cameras, lights=lights)
    )

    return phong_renderer

def get_voxel_renderer(device, cameras, image_size, volume_extent_world):

    volume_extent_world = volume_extent_world
    raysampler = NDCGridRaysampler(
            image_width=image_size,
            image_height=image_size,
            n_pts_per_ray=150,
            min_depth=0.1,
            max_depth=volume_extent_world*1.5,
        )

    raymarcher = EmissionAbsorptionRaymarcher()
    renderer = VolumeRenderer(
        raysampler=raysampler, raymarcher=raymarcher,
    )

    return renderer

def create_cameras(num_views, device, zdist, camera_mode, mirror_mode):
    
    if mirror_mode:
        elev = torch.linspace(0, 0, num_views*2)
        azim = torch.linspace(0, (360 - 360//(num_views*2)), num_views*2)
    else:
        elev = torch.linspace(0, 0, num_views)
        azim = torch.linspace(0, (180 - 180//num_views), num_views)

    R, T = look_at_view_transform(dist=zdist, elev=elev, azim=azim)

    if camera_mode == "perp":
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    else:
        cameras = FoVOrthographicCameras(device=device, R=R, T=T)
    
    return cameras, R, T

def create_cameras_TFS_mode(device, zdist, mirror_mode, camera_mode):

    if mirror_mode:
        elev = torch.tensor([0.0, 0.0, 90.0, 0.0, 0.0, 270.0])
        azim = torch.tensor([0.0, 90.0, 0.0, 180.0, 270.0, 0.0])
    else:
        elev = torch.tensor([0.0, 0.0, 90.0])
        azim = torch.tensor([0.0, 90.0, 0.0])

    R, T = look_at_view_transform(dist=zdist, elev=elev, azim=azim)
    
    if camera_mode == "perp":
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    else:
        cameras = FoVOrthographicCameras(device=device, R=R, T=T)
    
    return cameras, R, T

def create_cameras_4VTFS_mode(device, zdist, mirror_mode, camera_mode):

    if mirror_mode:
        elev = torch.tensor([0.0, 0.0, 0.0, 90.0, 0.0, 0.0, 0.0, 270.0])
        azim = torch.tensor([0.0, 60.0, 120.0, 0.0, 180.0, 240.0, 300.0, 0.0])
    else:
        elev = torch.tensor([0.0, 0.0, 0.0, 90.0])
        azim = torch.tensor([0.0, 60.0, 120.0, 0.0])

    R, T = look_at_view_transform(dist=zdist, elev=elev, azim=azim)
    
    if camera_mode == "perp":
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    else:
        cameras = FoVOrthographicCameras(device=device, R=R, T=T)
    
    return cameras, R, T

def render_voxels(voxels2, volume_extent_world, volume_size, zdist, renderer, device):

    colors = np.zeros((3,voxels2.shape[1],voxels2.shape[2],voxels2.shape[3]))
    colors[0] = 0.0
    colors[1] = 1.0
    colors[2] = 1.0
    print(voxels2.shape)
    print(colors.shape)

    colors = torch.from_numpy(colors.astype(np.float32)).to(device)
    voxels = torch.from_numpy(voxels2.astype(np.float32)).to(device)


    for azim in range(0,360,10):
        R, T = look_at_view_transform(dist=zdist, elev=0, azim=azim)
        cameras = FoVOrthographicCameras(device=device, R=R, T=T)
        volumes = Volumes(densities=voxels.unsqueeze(0),features=colors.unsqueeze(0), voxel_size= volume_extent_world / volume_size)
        rendered_images, rendered_silhouettes = renderer(cameras, volumes=volumes)[0].split([3, 1], dim=-1)
        img = rendered_images[0].cpu().numpy()
        silh = rendered_silhouettes[0].cpu().numpy()
        cv2.imshow("image",img)
        cv2.imshow("silh",silh)
        cv2.waitKey(0)

def save_vox2obj(vox_file_name, thresh, device, output_obj_file_name="None"):

    BlendParams.background_color = (0,0,0)

    with open(vox_file_name, "rb") as f:
        voxels = np.load(f)

    # colors = np.zeros((3,voxels.shape[1],voxels.shape[2],voxels.shape[3]))
    # colors[0] = 0.0
    # colors[1] = 1.0
    # colors[2] = 1.0

    # colors = torch.from_numpy(colors.astype(np.float32)).to(device)   
    voxels = torch.from_numpy(voxels.astype(np.float32)).to(device)

    mesh1 = cubify(voxels,thresh)
    final_verts = mesh1.verts_packed() 
    final_faces = mesh1.faces_packed()

    verts_features=torch.ones_like(final_verts, device=device).unsqueeze(0)
    textures = TexturesVertex(
        verts_features=verts_features
    )

    mesh1.textures = textures

    mesh1 = mesh1.to(device)

    cameras = FoVOrthographicCameras(device=device)

    ########### rasterization setting ############
    # Settings for rasterization
    raster_settings = RasterizationSettings(
        image_size=512, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
    )

    ############## Lighting setting ##############
    # # Placing point light at -> location
    lights = PointLights(
        ambient_color=((1., 1., 1.), ),
        diffuse_color=((0.2, 0.2, 0.2), ),
        specular_color=((0.2, 0.2, 0.2), ),
        device=device,
        location=[[0.0, 0.0, 1.5]])

    ############# Phong Renderer  ################
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device, 
            cameras=cameras,
            lights=lights
        )
    )


    for azim in range(0,360,10):
        R, T = look_at_view_transform(dist=2, elev=0, azim=azim)
        cameras = FoVOrthographicCameras(device=device, R=R, T=T)
        rendered_images, rendered_silhouettes = renderer(mesh1,cameras=cameras)[0].split([3, 1], dim=-1)
        img = rendered_images.cpu().numpy()
        silh = rendered_silhouettes.cpu().numpy()
        cv2.imshow("image",img)
        cv2.imshow("silh",silh)
        cv2.waitKey(0)

    if output_obj_file_name != "None":
        save_obj(output_obj_file_name, final_verts, final_faces)


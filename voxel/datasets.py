import sys
import torch
import os
import glob
import numpy as np
from utills import get_props
import cv2
import random

from skimage import img_as_ubyte
from pytorch3d.utils import ico_sphere

# io utils
from pytorch3d.io import load_obj, save_obj
from pytorch3d.ops import cubify
# datastructures
from pytorch3d.structures import Meshes, Volumes

from utills import (render_voxels,get_silhouette_renderer)


from pytorch3d.renderer import (
    FoVOrthographicCameras, FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,HardPhongShader,
    PointLights, TexturesVertex,SoftPhongShader
)


################# Functions for loading different datasets ############

def load_data(root_dir, file_name, mirror_mode, img_size, device, num_views, debug_mode):

    files = []
    
    f = open(os.path.join(root_dir,"data/",file_name),"r")
    for x in f:
        files.append(os.path.join(root_dir,"data/",x[:-1]))
    
    random.shuffle(files)
    files = files[:num_views]

    images = []
    img_tensors = []

    for file_ in files:
        _ , img = cv2.threshold(cv2.imread(file_,0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img = img.astype(np.uint8)
        img = cv2.resize(img, img_size)
        img, holes = get_props(img, img_size)        
        if debug_mode:
            cv2.imshow("img", img)
            cv2.waitKey(0)
        images.append(img)
    
    if mirror_mode:
        for file_ in files:
            _ , img = cv2.threshold(cv2.imread(file_,0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            img = img.astype(np.uint8)
            img = cv2.resize(img, img_size)
            img = cv2.flip(img,1)
            img, holes = get_props(img, img_size)
            images.append(img)

    img_tensors = torch.from_numpy(np.array([img/255.0 for img in images])).to(device)
         
    return images, img_tensors

def load_test_data(root_dir,main_exp_id,sub_exp_id,exp_sample_id,cameras,device, renderer, debug_mode, thresh_density):

    folder_path = root_dir+"results/%s/%s/"%(
        main_exp_id,
        sub_exp_id)
    obj_path = folder_path+"sample_%d_output.obj"%exp_sample_id
    voxel_path = folder_path+"sample_%d_final_voxels.npy"%exp_sample_id
    
    with open(voxel_path, "rb") as f:
        voxels = np.load(f)
    
    colors = np.zeros((3,voxels.shape[1],voxels.shape[2],voxels.shape[3]))
    colors[0] = 1.0
    colors[1] = 1.0
    colors[2] = 1.0

    colors = torch.from_numpy(colors.astype(np.float32)).to(device)
    voxels = torch.from_numpy(voxels.astype(np.float32)).to(device)

    # voxels = 1/(1 + torch.exp(-10000.0*(voxels - thresh_density*1.05)))
    # print(torch.max(voxels))
    # print(torch.min(voxels))


    volumes = Volumes(densities=voxels.unsqueeze(0),
        features=colors.unsqueeze(0),
        voxel_size= 1.7 / 128
        )

    # nviews = len(glob.glob(folder_path+"sample_%dview_*.png"%exp_sample_id))
    nviews = len(cameras)
    # print(nviews)

    pred_list = []
    target_list = []

    for i in range(nviews):
        file_name = folder_path+"sample_%dview_%d.png"%(exp_sample_id, i)
        _ , img = cv2.threshold(
            cv2.imread(file_name,0), 
            0, 
            255, 
            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        if debug_mode:
            cv2.imshow("views",img)
            cv2.waitKey(0)

        R = cameras.R[i].unsqueeze(0)
        T = cameras.T[i].unsqueeze(0)
        camera = FoVOrthographicCameras(device=device, R=R, T=T)
        rendered_images, rendered_silhouettes = renderer(camera, volumes=volumes)[0].split([3, 1], dim=-1)
        # rimg = rendered_images[0]
        # silh = rendered_silhouettes[0].cpu().numpy()
        silh = rendered_images[0].cpu().numpy()
        img = img/255.0

        pred_list.append(silh)
        target_list.append(img)

        if debug_mode:
            cv2.imshow("silh",silh)
            cv2.waitKey(0)

    
    pred = torch.from_numpy(np.array(pred_list).astype(np.float32)).to(device)
    target = torch.from_numpy(np.array(target_list).astype(np.float32)).unsqueeze(-1).to(device)

    return pred, target


def load_test_data2(root_dir,main_exp_id,sub_exp_id,exp_sample_id,cameras,device, renderer, debug_mode, thresh, image_size):

    folder_path = root_dir+"results/%s/%s/"%(
        main_exp_id,
        sub_exp_id)
    obj_path = folder_path+"sample_%d_output.obj"%exp_sample_id
    voxel_path = folder_path+"sample_%d_final_voxels.npy"%exp_sample_id
    
    with open(voxel_path, "rb") as f:
        voxels = np.load(f)
    
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
    renderer = get_silhouette_renderer(device, cameras, image_size)

    nviews = len(glob.glob(folder_path+"sample_%dview_*.png"%exp_sample_id))
    print(nviews)

    pred_list = []
    target_list = []

    for i in range(nviews):
        file_name = folder_path+"sample_%dview_%d.png"%(exp_sample_id, i)
        _ , img = cv2.threshold(
            cv2.imread(file_name,0), 
            0, 
            255, 
            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        if debug_mode:
            cv2.imshow("views",img)
            cv2.waitKey(0)

        R = cameras.R[i].unsqueeze(0)
        T = cameras.T[i].unsqueeze(0)
        camera = FoVOrthographicCameras(device=device, R=R, T=T)
        rendered_images, rendered_silhouettes = renderer(mesh1,cameras=camera)[0].split([3, 1], dim=-1)
        print("rendered images shape",rendered_images.shape)
        print("rendered silhouettes shape",rendered_silhouettes.shape)
        silh = rendered_silhouettes.cpu().numpy()
        img = img/255.0

        pred_list.append(silh)
        target_list.append(img)

        if debug_mode:
            cv2.imshow("silh",silh)
            cv2.waitKey(0)

    
    pred = torch.from_numpy(np.array(pred_list).astype(np.float32)).to(device)
    target = torch.from_numpy(np.array(target_list).astype(np.float32)).unsqueeze(-1).to(device)

    return pred, target




# def load_data_from_list(files, mirror_mode, img_size, device, num_views, debug_mode):

def load_data_from_list(files, mirror_mode, img_size, device, num_views, debug_mode):

    assert len(files) == num_views

    root = ""
    images = []
    img_tensors = []

    for file_ in files:
        file_path = root + file_
        _ , img = cv2.threshold(cv2.imread(file_path,0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img = img.astype(np.uint8)
        img = cv2.resize(img, img_size)
        img, holes = get_props(img, img_size)        
        if debug_mode:
            cv2.imshow("img", img)
            cv2.waitKey(0)
        images.append(img)
    
    if mirror_mode:
        for file_ in files:
            file_path = root + file_
            _ , img = cv2.threshold(cv2.imread(file_path,0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            img = img.astype(np.uint8)
            img = cv2.resize(img, img_size)
            img = cv2.flip(img,1)
            img, holes = get_props(img, img_size)
            images.append(img)

    img_tensors = torch.from_numpy(np.array([img/255.0 for img in images])).to(device)
         
    return images, img_tensors
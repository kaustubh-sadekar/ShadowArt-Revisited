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
import argparse

# io utils
from pytorch3d.io import load_obj, save_obj
from pytorch3d.ops import cubify

# datastructures
from pytorch3d.structures import Meshes, Volumes

# 3D transformations functions
from pytorch3d.transforms import Rotate, Translate

import random

# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    FoVOrthographicCameras, 
    VolumeRenderer,
    NDCGridRaysampler,
    EmissionAbsorptionRaymarcher,
    look_at_view_transform,
    TexturesVertex
)

from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)

import cv2
from utills import (get_voxel_renderer,get_phong_renderer, create_cameras, create_cameras_TFS_mode, create_cameras_4VTFS_mode, render_voxels)

from datasets import load_data, load_data_from_list
from losses import (huber, silh_loss, MS_SSIM, l1_loss, iou_np, dice_np)
from models import VolumeModel

def train():

    global exp_sample_id
    #############################################################
    #             Setting the rendering parameters              #
    #############################################################

    if TFS_mode:
        cameras, Rs, Ts = create_cameras_TFS_mode(device, zdist, mirror_mode, camera_mode)
    elif FourVTSF_mode:
        cameras, Rs, Ts = create_cameras_4VTFS_mode(device, zdist, mirror_mode, camera_mode)
    else:
        cameras, Rs, Ts = create_cameras(num_views, device, zdist, camera_mode, mirror_mode)

    renderer = get_voxel_renderer(device, cameras, img_size, volume_extent_world)
    phong_renderer = get_phong_renderer(device, FoVOrthographicCameras(device=device), img_size)

    i = 0

    print("Experiment:",main_exp_id+sub_exp_id)
    print("Sample : ",i)

    try:
        os.mkdir(root_dir+main_exp_id)
    except:
        print("Directory already exists")

    try:
        os.mkdir(root_dir+main_exp_id+sub_exp_id)
    except:
        print("Directory already exists")
    
    output_vid_path = root_dir+main_exp_id+sub_exp_id+"/sample_%d_vid.gif"%i
    print(output_vid_path) 
    writer = imageio.get_writer(output_vid_path, mode='I', duration=0.1)

    silhs, silhs_tensors = load_data_from_list(shadow_files,
            mirror_mode,
            (img_size, img_size),
            device,
            num_views,
            debug_mode
            )
    
    volume_size = 128 # Voxel Resolution
    volume_model = VolumeModel(
        renderer,
        volume_size=[volume_size] * 3, 
        voxel_size = volume_extent_world / volume_size,
        thresh_density = thresh_density
    ).to(device)
    
    optimizer = torch.optim.Adam(volume_model.parameters(), lr=lr)
    batch_size = 1

    loop = tqdm(range(Niter))
    # loop = range(Niter)

    ms_ssim_list = []
    
    # for i in loop:
    for iteration in loop:
        print(iteration)

        # In case we reached the last 75% of iterations,
        # decrease the learning rate of the optimizer 10-fold.
        if iteration == round(Niter * 0.75):
            print('Decreasing LR 10-fold ...')
            optimizer = torch.optim.Adam(
                volume_model.parameters(), lr=lr * 0.1
            )
        
        # Sample random batch indices.
        # batch_idx = torch.randperm(len(cameras))[:batch_size]
        # print(len(cameras))
        for batch_idx in range(len(cameras)):

            # Zero the optimizer gradient.
            optimizer.zero_grad()

            if camera_mode == "ortho":
                batch_cameras = FoVOrthographicCameras(
                    R = cameras.R[batch_idx].unsqueeze(0), 
                    T = cameras.T[batch_idx].unsqueeze(0), 
                    znear = cameras.znear[batch_idx].unsqueeze(0),
                    zfar = cameras.zfar[batch_idx].unsqueeze(0),
                    device = device,
                )
            else:
                batch_cameras = FoVPerspectiveCameras(
                    R = cameras.R[batch_idx].unsqueeze(0), 
                    T = cameras.T[batch_idx].unsqueeze(0), 
                    znear = cameras.znear[batch_idx].unsqueeze(0),
                    zfar = cameras.zfar[batch_idx].unsqueeze(0),
                    device = device,
                )
            
            # Evaluate the volumetric model.
            rendered_images, rendered_silhouettes = volume_model(
                batch_cameras
            ).split([3, 1], dim=-1)

            pred_output = rendered_images[0][:,:,0]

            sil_err = silh_loss(
                pred_output, silhs_tensors[batch_idx],
            )

            l1_err = l1_loss(
                pred_output.view(1,img_size,img_size).type(torch.float32).to(device), 
                silhs_tensors[batch_idx].view(1,img_size,img_size).type(torch.float32).to(device)
            )

            ms_ssim_err = ms_ssim_loss(
                pred_output.view(1,1,img_size,img_size).type(torch.float32).to(device), 
                silhs_tensors[batch_idx].view(1,1,img_size,img_size).type(torch.float32).to(device)
            )

            loss = sil_err *silh_wt + l1_err*l1_wt + ms_ssim_err*ms_ssim_wt

            if iteration%10 == 0:
                print(
                        f'Iteration {iteration:04d}:'
                        + f' loss = {float(loss):1.2e}'
                    )
            
            # Take the optimization step.
            loss.backward()
            optimizer.step()
            silh_pth = root_dir + main_exp_id + sub_exp_id + "/sample_%dview_%d_silh.png"%(exp_sample_id,batch_idx)
            silh_view_img = (pred_output.detach().cpu().numpy()*255).astype(np.uint8)
            ret, silh_view_img = cv2.threshold(silh_view_img,np.max(silh_view_img)*0.8,255,cv2.THRESH_BINARY)
            cv2.imwrite(silh_pth,silh_view_img)

            ms_ssim_list.append(ms_ssim_err.item())
        
        if iteration%10 == 0:
            print(
                    f'Iteration {iteration:04d}:'
                    + f' loss = {float(loss):1.2e}'
                )
            R, T = look_at_view_transform(zdist, 0, iteration, device=device)
            volumes = Volumes(
                densities = volume_model.voxels[None].expand(
                    batch_size, *volume_model.log_densities.shape),
                features = volume_model.colors[None].expand(
                    batch_size, *volume_model.log_colors.shape),
                voxel_size=volume_model._voxel_size,
            )
            image, silhouette = renderer(cameras=FoVOrthographicCameras(R=R, T=T, device=device), volumes=volumes)[0].split([3, 1], dim=-1)
            image = image[0, ..., :3].detach().squeeze().cpu().numpy()
            image = img_as_ubyte(image)
            writer.append_data(image)

    writer.close()    
    
    mesh1 = cubify(volume_model.voxels,thresh_density)
    final_verts = mesh1.verts_packed() 
    final_faces = mesh1.faces_packed()
        
    # # Store the predicted mesh using save_obj
    final_obj_pth = root_dir + main_exp_id + sub_exp_id +"/sample_%d_output.obj"%exp_sample_id
    final_voxel_pth = root_dir + main_exp_id + sub_exp_id +"/sample_%d_final_voxels.npy"%exp_sample_id

    save_obj(final_obj_pth, final_verts, final_faces)
    
    voxels = volume_model.voxels.detach().cpu().numpy()
    colors = volume_model.colors.detach().cpu().numpy()
    
    with open(final_voxel_pth, 'wb') as f:
        np.save(f,voxels)
    
    folder_pth = root_dir + main_exp_id + sub_exp_id
    for i, img in enumerate(silhs):
        cv2.imwrite(folder_pth+"/sample_%dview_%d.png"%(exp_sample_id,i),img)
    
    if debug_mode:
        with open(final_voxel_pth, 'rb') as f:
            voxels = np.load(f)
        render_voxels(voxels, volume_extent_world, volume_size, zdist, renderer, device)
    
    ms_ssim_metric = 1 - np.array(ms_ssim_list).mean()

    mean_iou = 0.0
    mean_dice = 0.0
    for idx in range(num_views*(1+mirror_mode)):

        gt_shadow = cv2.imread(folder_pth+"/sample_%dview_%d.png"%(exp_sample_id,idx),0)
        pred_shadow = cv2.imread(folder_pth+"/sample_%dview_%d_silh.png"%(exp_sample_id,idx),0)
        diff_img = np.abs(gt_shadow - pred_shadow)

        mean_iou+= iou_np(gt_shadow,pred_shadow)
        mean_dice+= dice_np(gt_shadow, pred_shadow)

        gt_shadow = cv2.bitwise_not(gt_shadow)
        pred_shadow = cv2.bitwise_not(pred_shadow)

        cv2.imwrite(folder_pth+"/SHADOW_GT%d_view_%d.png"%(exp_sample_id,idx),gt_shadow)
        cv2.imwrite(folder_pth+"/SHADOW_PRED%d_view_%d.png"%(exp_sample_id,idx),pred_shadow)
        cv2.imwrite(folder_pth+"/SHADOW_DIFF%d_view_%d.png"%(exp_sample_id,idx),diff_img)


    
    mean_iou = mean_iou/(1.0*num_views*(1+mirror_mode))
    mean_dice = mean_dice/(1.0*num_views*(1+mirror_mode))    

    text = ""
    text += "\nEdge loss : " + str(mesh_edge_loss(mesh1).item())
    text += "\nLaplacian loss : " + str(mesh_laplacian_smoothing(mesh1).item())
    text += "\nNormal loss : " + str(mesh_normal_consistency(mesh1).item())
    text += "\nIOU metric: " + str(mean_iou)
    text += "\nDice metric: " + str(mean_dice)
    text += "\nMISSIM metric: " + str(ms_ssim_metric)

    text_file = open(folder_pth+"/log.txt", "w")
    n = text_file.write(cmd_input + text)
    text_file.close()

    exp_sample_id+=1

parser = argparse.ArgumentParser(description = "List of various parameters for experiments")
parser.add_argument("device", type=str, help="GPU number")
parser.add_argument("sub_exp_id", type=str, help="sub experiment id")
parser.add_argument("Niter", type=int, help="Number of iterations")
parser.add_argument("lr", type=float, help="Learning rate")

parser.add_argument("-vfl","--views_file_name", type=str, help="Name of file containing path to ground truth views",default="dataset1.txt")
parser.add_argument("-mr","--mirror_mode", type=bool, help="Mirror mode set to true if front and rear both views are to be regressed", default=1)
parser.add_argument("-mr2","--mirror_mode_2", type=bool, help="Mirror mode set to true if front and rear both views are to be regressed", default=1)
parser.add_argument("-tsf","--TSF_mode", type=bool, help="set true for Top-Side-Front view 3 view setup", default=0)
parser.add_argument("-tsf4","--TSF4V_mode", type=bool, help="set true for TSF with three side view setup", default=0)
parser.add_argument("-cam","--camera_mode", type=str, help="set camera mode as ortho or perspective", default="ortho")
parser.add_argument("-imsz","--img_size", type=int, help="set image size",default=512)
parser.add_argument("-swt","--silh_wt", type=float, help="Silhoutte loss weight", default=10.0)
parser.add_argument("-l1wt","--l1_wt", type=float, help="L1 loss weight", default=10.0)
parser.add_argument("-mwt","--ms_ssim_wt", type=float, help="MS_SSIM loss weight", default=0.0)
parser.add_argument("-ns","--num_samples", type=int, help="Number of samples", default=1)
parser.add_argument("-th","--thresh_density", type=float, help="Cubify function threshold", default=0.05)
parser.add_argument("-zd","--zdist", type=float, help="Cubify function threshold", default=1.7)
parser.add_argument("-sdlist", "--shadow_files", nargs="+", default=["None"])

args = parser.parse_args()



#############################################################
#                 Experiment Key Parameters                 #
#############################################################

# setting Device
if torch.cuda.is_available() and (args.device != "cpu"):
    device = torch.device(args.device)
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

print("Device: ", device)

ms_ssim_loss = MS_SSIM(device)

random.seed(43)
root_dir = "../"
main_exp_id = "voxel_results/"

sub_exp_id = args.sub_exp_id
file_name = args.views_file_name
mirror_mode = args.mirror_mode
FourVTSF_mode = args.TSF4V_mode
mirror_mode_2 = args.mirror_mode_2 # Make it false to get cameras as mirror view but rear view as not a mirror view but other obj view
thresh_density = args.thresh_density
TFS_mode = args.TSF_mode
camera_mode = args.camera_mode
img_size = args.img_size
Niter = args.Niter
zdist = args.zdist
debug_mode = False
num_samples = args.num_samples
volume_extent_world = 1.7
exp_sample_id = 0
lr = args.lr
l1_wt = args.l1_wt
silh_wt = args.silh_wt
ms_ssim_wt = args.ms_ssim_wt
shadow_files = args.shadow_files
num_views = len(shadow_files)


cmd_input = "The command line input string \n"+str(sys.argv)

train()



# python train.py cuda:1 temp_trial 30 0.01 -swt 10.0 -l1wt 10.0 -mwt 0.0 -ns 2
# python val.py cuda:0 output1 600 0.01 -swt 10.0 -l1wt 10.0 -sdlist duck.png mikey.png

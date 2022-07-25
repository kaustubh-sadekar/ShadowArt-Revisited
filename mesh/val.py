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

# datastructures
from pytorch3d.structures import Meshes

# 3D transformations functions
from pytorch3d.transforms import Rotate, Translate

import random

# rendering components
from pytorch3d.renderer import (
    FoVOrthographicCameras, FoVPerspectiveCameras,
    look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer,
    MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader,
    PointLights, TexturesVertex,
)

from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)

import cv2
from utills import (get_silhouette_renderer, get_phong_renderer, create_cameras, create_cameras_TFS_mode, create_cameras_4VTFS_mode)

from datasets import load_data, load_data_from_list
from losses import (update_mesh_shape_prior_losses, silh_loss,MS_SSIM, l1_loss, iou_np, dice_np)


def val():

    #############################################################
    #             Setting the rendering parameters              #
    #############################################################

    if TFS_mode:
        cameras, Rs, Ts = create_cameras_TFS_mode(device, zdist, mirror_mode, camera_mode)
    elif TFS4v_mode:
        cameras, Rs, Ts = create_cameras_4VTFS_mode(device, zdist, mirror_mode, camera_mode)
    else:
        cameras, Rs, Ts = create_cameras(num_views, device, zdist, camera_mode, mirror_mode)
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

    silhouette_renderer = get_silhouette_renderer(device, FoVOrthographicCameras(device=device), img_size)
    phong_renderer = get_phong_renderer(device, FoVOrthographicCameras(device=device), img_size)

    min_loss_log = []
    
    silhs, silhs_tensors = load_data_from_list(shadow_files,
            mirror_mode,
            (img_size, img_size),
            device,
            num_views,
            debug_mode
            )

    losses = {"silhouette": {"weight": silh_wt, "values": []},
                "l1": {"weight": l1_wt, "values": []},
                "i2v": {"weight": i2v_wt, "values": []},
                "ms_ssim": {"weight": ms_ssim_wt, "values": []},
                "edge": {"weight": edge_wt, "values": []},
                "normal": {"weight": norm_wt, "values": []},
                "laplacian": {"weight": lapl_wt, "values": []},
                }
    
    src_mesh = ico_sphere(4, device)

    verts = src_mesh.verts_packed()
    faces = src_mesh.faces_packed()
    faces = faces.t().contiguous()
    verts = verts.type(torch.float)
    # verts *= 1.0
    src_mesh.verts = verts
    faces = faces.type(torch.long)
    # edge_index = torch.cat([faces[:2], faces[1:], faces[::2]], dim=1)
    # edge_index = to_undirected(edge_index, num_nodes=verts.shape[0])

    verts_shape = verts.shape
    deform_verts = torch.full(verts_shape, 0.0, device=device, requires_grad=True)
    optimizer = torch.optim.SGD([deform_verts], lr=lr, momentum=0.9)

    try:
        os.mkdir(root_dir +main_exp_id)
    except:
        print("Directory already exists")

    try:
        os.mkdir(root_dir +main_exp_id+ sub_exp_id)
    except:
        print("Directory already exists")

    output_vid_path = root_dir+main_exp_id+sub_exp_id+"/sample_%d_vid.gif"%exp_sample_id
    folder_pth = root_dir + main_exp_id + sub_exp_id
    
    print(output_vid_path) 
    
    writer = imageio.get_writer(output_vid_path, mode='I', duration=0.1)


    optimization_loss = []
    ms_ssim_loss_list = []

    loop = tqdm(range(Niter))
    
    for i in loop:
        print("Iteration: ",i)
        optimizer.zero_grad()

        # print("silh gt shape",silh_gt.shape)

        # deform_verts = model((verts, edge_index),silh_gt)
        # new_src_mesh = src_mesh.offset_verts(deform_verts)
        loss = {k: torch.tensor(0.0, device=device) for k in losses}
        # update_mesh_shape_prior_losses(new_src_mesh, loss)

        n_silh_loss = torch.tensor(0.0, device=device)
        n_ms_ssim_loss = torch.tensor(0.0, device=device)
        n_l1_loss = torch.tensor(0.0, device=device)
        i2v_loss = torch.tensor(0.0, device=device)
        for n, silh_gt in enumerate(silhs_tensors):
            # optimizer.zero_grad()
            target_view = silh_gt.view(1,1,img_size,img_size).type(torch.FloatTensor).to(device)
            new_src_mesh = src_mesh.offset_verts(deform_verts)
            update_mesh_shape_prior_losses(new_src_mesh, loss)
            R = Rs[n].unsqueeze(0).to(device)
            T = Ts[n].unsqueeze(0).to(device)
            view_pred = silhouette_renderer(new_src_mesh, R = R, T = T)
            silh_pred = view_pred[..., 3][0]
            n_silh_loss += silh_loss(silh_pred, silh_gt)
            n_l1_loss += l1_loss(silh_pred, silh_gt)
            # i2v_loss = ((img2v_loss(silh_pred.view(1,img_size,img_size).type(torch.float32)) -  
            #     img2v_loss(silh_gt.view(1,img_size,img_size).type(torch.float32)))**2).mean()
            n_ms_ssim_loss += ms_ssim_loss(silh_pred.view(1,1,img_size,img_size).type(torch.float32), 
                silh_gt.view(1,1,img_size,img_size).type(torch.float32))
            
            silh_pred_img = view_pred[..., 3][0].clone()
            silh_pred_img = (silh_pred_img.detach().cpu().numpy()*255).astype(np.uint8)
            ret2,silh_pred_img = cv2.threshold(silh_pred_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            cv2.imwrite(folder_pth+"/sample_%d_pred_view_%d.png"%(exp_sample_id,n),silh_pred_img)

            # n_logprob_loss += logprob_loss(x_mu, x_var, deform_verts)
        
        loss["silhouette"] = n_silh_loss*1.0/num_views
        loss["l1"] = n_l1_loss*1.0/num_views
        loss["ms_ssim"] = n_ms_ssim_loss*1.0/num_views

        sum_loss = torch.tensor(0.0, device=device)
        for k,l in loss.items():
            # print(k,":",l)
            sum_loss += l*losses[k]["weight"]
            losses[k]["values"].append(l)
        
        sum_loss = sum_loss
        sum_loss.backward()
        optimizer.step()
        ms_ssim_loss_list.append(n_ms_ssim_loss.item()*1.0/num_views)

        # print("silhouette loss : ", loss["silhouette"])
        # print("logprob loss : ", loss["logprob"])
        # print("edge loss : ", loss["edge"])
        # print("normal loss : ", loss["normal"])
        # print("laplacian loss : ", loss["laplacian"])
        
                
        # loop.set_description("total_loss = %.6f"%sum_loss)
        print("total_loss = %.6f"%sum_loss)

        if sum_loss.item() < 0.00001:
            break
        
        optimization_loss.append(sum_loss.item())

        if i % 10 == 0:
            new_src_verts = new_src_mesh.verts_packed()
            new_src_faces = new_src_mesh.faces_packed()
            verts_rgb = torch.ones_like(new_src_verts)[None]  # (1, V, 3)
            verts_rgb[0,:,2]*=0.4
            # verts_rgb[0,:,1]*=0.2
            textures = TexturesVertex(verts_features=verts_rgb.to(device))

            # Create a Meshes object for the teapot. Here we have only one mesh in the batch.
            new_src_mesh1 = Meshes(
                verts=[new_src_verts.to(device)],   
                faces=[new_src_faces.to(device)], 
                textures=textures
            )
            R, T = look_at_view_transform(zdist, 0, i, device=device)
            image = phong_renderer(new_src_mesh1, R=R, T=T)
            image = image[0, ..., :3].detach().squeeze().cpu().numpy()
            image = img_as_ubyte(image)
            writer.append_data(image)
    
    writer.close
    
    min_loss_log.append(sum_loss.item())
    
    # Fetch the verts and faces of the final predicted mesh
    final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)
        
    # Store the predicted mesh using save_obj
    final_obj = root_dir + main_exp_id + sub_exp_id +"/sample_%d_output.obj"%exp_sample_id
    save_obj(final_obj, final_verts, final_faces)
    for i, img in enumerate(silhs):
        cv2.imwrite(folder_pth+"/sample_%dview_%d.png"%(exp_sample_id,i),img)
    """
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
    """
    print("Edge loss : ", mesh_edge_loss(new_src_mesh))
    print("Laplacian loss : ", mesh_laplacian_smoothing(new_src_mesh))
    print("Normal loss: ", mesh_normal_consistency(new_src_mesh))

    text = "\n"
    text += "\nOptimization Loss \n" + str(min_loss_log)
    text += "\nEdge loss : " + str(mesh_edge_loss(new_src_mesh).item())
    text += "\nLaplacian loss : " + str(mesh_laplacian_smoothing(new_src_mesh).item())
    text += "\nNormal loss : " + str(mesh_normal_consistency(new_src_mesh).item())
    
    mean_iou = 0.0
    mean_dice = 0.0
    for idx in range(num_views*(1+mirror_mode)):
        mean_iou+= iou_np(cv2.imread(folder_pth+"/sample_%dview_%d.png"%(exp_sample_id,idx),0),
        cv2.imread(folder_pth+"/sample_%d_pred_view_%d.png"%(exp_sample_id,idx),0))
        
        mean_dice+= dice_np(cv2.imread(folder_pth+"/sample_%dview_%d.png"%(exp_sample_id,idx),0),
        cv2.imread(folder_pth+"/sample_%d_pred_view_%d.png"%(exp_sample_id,idx),0))
    
    mean_iou = mean_iou/(1.0*num_views*(1+mirror_mode))
    mean_dice = mean_dice/(1.0*num_views*(1+mirror_mode))

    ms_ssim_metric = (1 - np.array(ms_ssim_loss_list).mean())

    text += "\nIOU : " + str(mean_iou)
    text += "\nDice : " + str(mean_dice)
    text += "\nMISSIM : " + str(ms_ssim_metric)
    

    text_file = open(folder_pth+"/log.txt", "w")
    n = text_file.write(cmd_input + text)
    text_file.close()



parser = argparse.ArgumentParser(description = "List of various parameters for experiments")
parser.add_argument("device", type=str, help="GPU number")
parser.add_argument("sub_exp_id", type=str, help="sub experiment id")
parser.add_argument("Niter", type=int, help="Number of iterations")
parser.add_argument("lr", type=float, help="Learning rate")
parser.add_argument("model_id", type=int, help="Model id 0,1,2")

parser.add_argument("-vfl","--views_file_name", type=str, help="Name of file containing path to ground truth views",default="dataset1.txt")
parser.add_argument("-mr","--mirror_mode", type=bool, help="Mirror mode set to true if front and rear both views are to be regressed", default=1)
parser.add_argument("-tsf","--TSF_mode", type=bool, help="set true for Top-Side-Front view 3 view setup", default=0)
parser.add_argument("-tsf4","--TSF4V_mode", type=bool, help="set true for TSF with three side view setup", default=0)
parser.add_argument("-cam","--camera_mode", type=str, help="set camera mode as ortho or perspective", default="ortho")
parser.add_argument("-imsz","--img_size", type=int, help="set image size",default=256)
parser.add_argument("-swt","--silh_wt", type=float, help="Silhoutte loss weight", default=1.6)
parser.add_argument("-l1wt","--l1_wt", type=float, help="L1 loss weight", default=1.6)
parser.add_argument("-i2vwt","--i2v_wt", type=float, help="L1 loss weight", default=0.0)
parser.add_argument("-mwt","--ms_ssim_wt", type=float, help="MS_SSIM loss weight", default=0.0)
parser.add_argument("-ewt","--edge_wt", type=float, help="Edge loss weight", default=1.2)
parser.add_argument("-nwt","--norm_wt", type=float, help="Normal loss weight", default=0.03)
parser.add_argument("-lwt","--lapl_wt", type=float, help="Laplacian loss weight", default=1.2)
parser.add_argument("-ns","--num_samples", type=int, help="Number of samples", default=1)
parser.add_argument("-sdlist", "--shadow_files", nargs="+", default=["None"])
args = parser.parse_args()

# setting GPU ID
if torch.cuda.is_available() and (args.device != "cpu"):
    device = torch.device(args.device)
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

print("Device: ", device)

random.seed(43)
root_dir = "../"
main_exp_id = "mesh_results/"
zdist = 2.5
debug_mode = False
exp_sample_id = 0


# Parameters to be parsed 
sub_exp_id = args.sub_exp_id
file_name = args.views_file_name
mirror_mode = args.mirror_mode
TFS_mode = args.TSF_mode
TFS4v_mode = args.TSF4V_mode
camera_mode = args.camera_mode
img_size = args.img_size
Niter = args.Niter
lr = args.lr
silh_wt = args.silh_wt
l1_wt = args.l1_wt
ms_ssim_wt = args.ms_ssim_wt
i2v_wt = args.i2v_wt
edge_wt = args.edge_wt
norm_wt = args.norm_wt
lapl_wt = args.lapl_wt
num_samples = args.num_samples
model_id = args.model_id
shadow_files = args.shadow_files
num_views = len(shadow_files)

cmd_input = "The command line input string \n"+str(sys.argv)

ms_ssim_loss = MS_SSIM(device)


val()

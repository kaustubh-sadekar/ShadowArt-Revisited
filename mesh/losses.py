import sys
import torch
import os
import glob
import numpy as np
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, sigma, channel):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


class Gaussian_Blur(torch.nn.Module):
    def __init__(self):
        super(Gaussian_Blur, self).__init__()
        self.channel = 3

    def _gauss_blur(self, img1, img2):
        _, c, w, h = img1.size()
        window_size = min(w, h, 11)
        # sigma = 1.5 * window_size / 11
        sigma = 0.5
        window = create_window(window_size, sigma, self.channel).cuda()
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=self.channel)
        return torch.tanh(mu1), torch.tanh(mu2)

    def forward(self, img1, img2):
        g_img1, g_img2 = self._gauss_blur(img1, img2)
        return g_img1, g_img2



class MS_SSIM(torch.nn.Module):
    def __init__(self, device, size_average = True, max_val = 255):
        super(MS_SSIM, self).__init__()
        self.size_average = size_average
        self.channel = 1
        self.max_val = max_val
        self.device = device

    def _ssim(self, img1, img2, size_average = True):

        _, c, w, h = img1.size()
        window_size = min(w, h, 11)
        sigma = 1.5 * window_size / 11
        window = create_window(window_size, sigma, self.channel).to(self.device)
        mu1 = F.conv2d(img1, window, padding = window_size//2, groups = self.channel)
        mu2 = F.conv2d(img2, window, padding = window_size//2, groups = self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = self.channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = self.channel) - mu1_mu2

        C1 = (0.01*self.max_val)**2
        C2 = (0.03*self.max_val)**2
        V1 = 2.0 * sigma12 + C2
        V2 = sigma1_sq + sigma2_sq + C2
        ssim_map = ((2*mu1_mu2 + C1)*V1)/((mu1_sq + mu2_sq + C1)*V2)
        mcs_map = V1 / V2
        if size_average:
            return ssim_map.mean(), mcs_map.mean()

    def SSIM(self,x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = F.avg_pool2d(x, 3, 1, 1)
        mu_y = F.avg_pool2d(y, 3, 1, 1)

        sigma_x = F.avg_pool2d(x ** 2, 3, 1, 1) - mu_x ** 2
        sigma_y = F.avg_pool2d(y ** 2, 3, 1, 1) - mu_y ** 2
        sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return torch.clamp((1-SSIM) / 2, 0, 1)

    def ms_ssim(self, img1, img2, levels=5):

        weight = Variable(torch.Tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(self.device))

        msssim = Variable(torch.Tensor(levels,).to(self.device))
        mcs = Variable(torch.Tensor(levels,).to(self.device))
        for i in range(levels):
            ssim_map, mcs_map = self._ssim(img1, img2)
            msssim[i] = ssim_map
            mcs[i] = mcs_map
            filtered_im1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
            filtered_im2 = F.avg_pool2d(img2, kernel_size=2, stride=2)
            img1 = filtered_im1
            img2 = filtered_im2

        value = (torch.prod(mcs[0:levels-1]**weight[0:levels-1])*(msssim[levels-1]**weight[levels-1]))
        return torch.clamp(value / 2, 0, 1)


    def forward(self, img1, img2):
        
        return self.ms_ssim(img1, img2)
        # return self.SSIM(img1,img2)

# Losses to smooth / regularize the mesh shape
def update_mesh_shape_prior_losses(mesh, loss):
    # and (b) the edge length of the predicted mesh
    loss["edge"] = mesh_edge_loss(mesh)
    
    # mesh normal consistency
    loss["normal"] = mesh_normal_consistency(mesh)
    
    # mesh laplacian smoothing
    loss["laplacian"] = mesh_laplacian_smoothing(mesh, method="uniform")


def silh_loss(silh_pred, silh_gt):
    return ((silh_pred - silh_gt) ** 2).mean()

def l1_loss(silh_pred, silh_gt):
    return (torch.abs(silh_pred - silh_gt)).mean()



def iou(pred, target):

    n = torch.sum(pred*target)
    d = torch.sum(pred + target - pred*target)
    
    return n/d

def dice(pred, target):

    n = torch.sum(pred*target)
    d = torch.sum(pred + target)
    
    
    return 2*n/d


def iou_np(pred, target):

    n = np.sum(pred*target)
    d = np.sum(pred + target - pred*target)
    
    return n/d

def dice_np(pred, target):

    n = np.sum(pred*target)
    d = np.sum(pred + target)
    
    
    return 2*n/d
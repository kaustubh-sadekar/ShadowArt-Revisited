import os
import sys
import time
import torch
import math
import numpy as np
from PIL import Image
from pytorch3d.ops import cubify
from pytorch3d.io import save_obj
import random
# from IPython import display

# Data structures and functions for rendering
from pytorch3d.structures import Volumes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    FoVOrthographicCameras, 
    VolumeRenderer,
    NDCGridRaysampler,
    EmissionAbsorptionRaymarcher,
    look_at_view_transform
)
from pytorch3d.transforms import so3_exponential_map


class VolumeModel(torch.nn.Module):
    def __init__(self, renderer, volume_size=[64] * 3, voxel_size=0.1, thresh_density = 0.005):
        super().__init__()
        # After evaluating torch.sigmoid(self.log_colors), we get 
        # densities close to zero.
        self.log_densities = torch.nn.Parameter(-4.0 * torch.ones(1, *volume_size))
        # After evaluating torch.sigmoid(self.log_colors), we get 
        # a neutral gray color everywhere.
        colors_temp = torch.zeros(3, *volume_size)
        colors_temp[0] = 1.0
        # print(colors_temp.shape)
        self.log_colors = torch.nn.Parameter(colors_temp,requires_grad=False)
        self._voxel_size = voxel_size
        # Store the renderer module as well.
        self._renderer = renderer
        self.voxels = None
        self.thresh_density = thresh_density
        
    def forward(self, cameras):
        batch_size = cameras.R.shape[0]

        # Convert the log-space values to the densities/colors
        densities = torch.sigmoid(self.log_densities)
        colors = torch.sigmoid(self.log_colors)

        # densities = torch.clamp(densities, min= 0.005, max=1.0)
        # densities = 1/(1+torch.exp(-10000.0*(densities - self.thresh_density))) # Thresholding does not help

        # Instantiate the Volumes object, making sure
        # the densities and colors are correctly
        # expanded batch_size-times.
        volumes = Volumes(
            densities = densities[None].expand(
                batch_size, *self.log_densities.shape),
            features = colors[None].expand(
                batch_size, *self.log_colors.shape),
            voxel_size=self._voxel_size,
        )

        self.voxels = densities
        self.colors = colors
        
        # Given cameras and volumes, run the renderer
        # and return only the first output value 
        # (the 2nd output is a representation of the sampled
        # rays which can be omitted for our purpose).
        return self._renderer(cameras=cameras, volumes=volumes)[0]

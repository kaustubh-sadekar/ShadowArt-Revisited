import sys
import torch
import os
import glob
import numpy as np
from utills import get_props
import cv2
import random

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")



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

    for img in images:
        img_tensors.append(torch.from_numpy(img/255.0).to(device))
        
    return images, img_tensors


def load_data_from_list(files, mirror_mode, img_size, device, num_views, debug_mode):

    assert len(files) == num_views
    
    images = []
    img_tensors = []

    root = ""

    for file_ in files:
        file_path = root + file_
        print(file_path)
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

    for img in images:
        img_tensors.append(torch.from_numpy(img/255.0).to(device))
        
    return images, img_tensors

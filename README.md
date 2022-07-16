# [ShadowArt-Revisited](https://kaustubh-sadekar.github.io/ShadowArt-Revisited/)
Shadow Art Revisited: A Differentiable Rendering Based Approach


## Setup Instructions

## Create Shadow Art Using Voxel Optimization

```text

usage: val.py [-h] [-vfl VIEWS_FILE_NAME] [-mr MIRROR_MODE]
              [-mr2 MIRROR_MODE_2] [-tsf TSF_MODE] [-tsf4 TSF4V_MODE]
              [-cam CAMERA_MODE] [-imsz IMG_SIZE] [-swt SILH_WT] [-l1wt L1_WT]
              [-mwt MS_SSIM_WT] [-ns NUM_SAMPLES] [-th THRESH_DENSITY]
              [-zd ZDIST] [-sdlist SHADOW_FILES [SHADOW_FILES ...]]
              device sub_exp_id Niter lr

List of various parameters for experiments

positional arguments:
  device                GPU number
  sub_exp_id            sub experiment id
  Niter                 Number of iterations
  lr                    Learning rate

optional arguments:
  -h, --help            show this help message and exit
  -vfl VIEWS_FILE_NAME, --views_file_name VIEWS_FILE_NAME
                        Name of file containing path to ground truth views
  -mr MIRROR_MODE, --mirror_mode MIRROR_MODE
                        Mirror mode set to true if front and rear both views
                        are to be regressed
  -mr2 MIRROR_MODE_2, --mirror_mode_2 MIRROR_MODE_2
                        Mirror mode set to true if front and rear both views
                        are to be regressed
  -tsf TSF_MODE, --TSF_mode TSF_MODE
                        set true for Top-Side-Front view 3 view setup
  -tsf4 TSF4V_MODE, --TSF4V_mode TSF4V_MODE
                        set true for TSF with three side view setup
  -cam CAMERA_MODE, --camera_mode CAMERA_MODE
                        set camera mode as ortho or perspective
  -imsz IMG_SIZE, --img_size IMG_SIZE
                        set image size
  -swt SILH_WT, --silh_wt SILH_WT
                        Silhoutte loss weight
  -l1wt L1_WT, --l1_wt L1_WT
                        L1 loss weight
  -mwt MS_SSIM_WT, --ms_ssim_wt MS_SSIM_WT
                        MS_SSIM loss weight
  -ns NUM_SAMPLES, --num_samples NUM_SAMPLES
                        Number of samples
  -th THRESH_DENSITY, --thresh_density THRESH_DENSITY
                        Cubify function threshold
  -zd ZDIST, --zdist ZDIST
                        Cubify function threshold
  -sdlist SHADOW_FILES [SHADOW_FILES ...], --shadow_files SHADOW_FILES [SHADOW_FILES ...]
```
**Example**
To create shadow art with two views with files puma.png and mikey.png use the following command.
```script
python val.py cuda:0 vox_2view_exp5 600 0.01 -swt 10.0 -l1wt 10.0 -sdlist puma.png mikey.png
```

To Do
- [ ] Need to add the default values for the cli




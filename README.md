# [ShadowArt-Revisited](https://kaustubh-sadekar.github.io/ShadowArt-Revisited/)
Shadow Art Revisited: A Differentiable Rendering Based Approach

## Setup Instructions

**Clone the repository**

```script
git clone https://github.com/kaustubh-sadekar/ShadowArt-Revisited.git
cd ShadowArt-Revisited
```

**Install required libraries**

```script
pip install -r requirements.txt
pip plotly
curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz
tar xzf 1.10.0.tar.gz
python -c "import os; os.environ["CUB_HOME"] = os.getcwd() + "/cub-1.10.0" "
pip install 'git+https://github.com/facebookresearch/pytorch3d.git@stable'
```

## Examples
### Create Shadow Art Using Voxel Optimization

To create shadow art with two views with files duck.png and mikey.png use the following command.
```script
python val.py cuda:0 output1 600 0.01 -swt 10.0 -l1wt 10.0 -sdlist duck.png mikey.png
```
*For better understanding of the input arguments type `python val.py -h`*

![Voxel Output](media/voxel_shadow_art.gif)

*Gif showing the optimization of the 3D voxel shadow art*


### Create Shadow Art Using Mesh Optimization

To create shadow art with two views with files duck.png and mikey.png use the following command.
```script
python val.py cuda:0 output1 2000 0.15 0 -swt 1.6 -l1wt 1.6 -mwt 0.0 -i2vwt 0.0 -ewt 1.6 -nwt 0.6 -lwt 1.2 -sdlist duck.png mikey.png
```
*For better understanding of the input arguments type `python val.py -h`*

![Mesh Output](media/mesh_shadow_art.gif)

*Gif showing the optimization of the 3D shadow art mesh*


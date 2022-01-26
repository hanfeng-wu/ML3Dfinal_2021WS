import torch
import numpy as np
from Networks.mesh2mesh import *
from Networks.rgbd2mesh import RGBD2Mesh
from Networks.obj2pointcloud import *
import OpenEXR
import trimesh
import os
import sys




obj = trimesh.load(f'./test.obj')

if isinstance(obj, trimesh.Trimesh):
    vertices = obj.vertices.view(np.ndarray)
    faces = obj.faces.view(np.ndarray)

else:
    obj = obj.dump().sum()
    vertices = obj.vertices.view(np.ndarray)
    faces = obj.faces.view(np.ndarray)
pcd_full = sample_point_cloud(vertices, faces, 4000)
export_pointcloud_to_obj(f'./pcd_gt.obj', pcd_full)    
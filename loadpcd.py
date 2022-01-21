
import torch
import numpy as np
from Networks.mesh2mesh import *
from Networks.rgbd2mesh import RGBD2Mesh
from Networks.obj2pointcloud import *
import OpenEXR
import trimesh
import os
import sys


start = sys.argv[1]
end = sys.argv[2]



depthIntrinsics = np.array([[711.111083984375, 0.0, 255.5], [0.0, 711.111083984375, 255.5], [0.0, 0.0, 1.0]])
depthExtrinsics = [
    np.array([[0.6859, 0.7277, -0.0, -0.0225], [0.324, -0.3054, -0.8954, -0.0405], [-0.6516, 0.6142, -0.4453, 1.4963], [0.0, 0.0, 0.0, 1.0]]),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, 1.5], [0.0, 0.0, 0.0, 1.0]]),
    np.array([[-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, -0.0], [0.0, -1.0, 0.0, 1.5], [0.0, 0.0, 0.0, 1.0]]),
    np.array([[-0.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [-1.0, 0.0, 0.0, 1.5], [0.0, 0.0, 0.0, 1.0]]),
    np.array([[-0.7424, 0.67, 0.0, 0.0], [-0.3634, -0.4027, -0.8401, -0.0], [-0.5628, -0.6237, 0.5425, 1.5], [0.0, 0.0, 0.0, 1.0]]),
    np.array([[0.2058, 0.9786, -0.0, 0.0], [-0.4049, 0.0852, -0.9104, -0.0], [-0.8909, 0.1874, 0.4137, 1.5], [0.0, 0.0, 0.0, 1.0]]),
    np.array([[-1.0, -0.007, 0.0, -0.0], [-0.0057, 0.8049, -0.5934, -0.0], [0.0042, -0.5933, -0.8049, 1.5], [0.0, 0.0, 0.0, 1.0]]),
    np.array([[1.0, -0.007, 0.0, 0.0], [0.0024, 0.3401, -0.9404, 0.0], [0.0066, 0.9404, 0.3401, 1.5], [0.0, 0.0, 0.0, 1.0]]),
    np.array([[-0.726, 0.6877, 0.0, -0.0], [0.3241, 0.3421, -0.882, 0.0], [-0.6066, -0.6403, -0.4712, 1.5], [0.0, 0.0, 0.0, 1.0]]),
    np.array([[0.5203, -0.854, -0.0, 0.0011], [-0.7902, -0.4814, -0.3792, 0.0048], [0.3239, 0.1973, -0.9253, 2.584], [0.0, 0.0, 0.0, 1.0]])
]

datadir = './Data/shapenetdata/'
subdirs = []
for x in os.walk(datadir):
    subdirs.append(x[0])
subdirs = sorted(subdirs)
subdirs = subdirs[1:]
for subdir in subdirs[int(start):int(end)]:
    sys.stderr.write(f'start loading{subdir} \n')
    file = open(f'{subdir}/model.obj', 'r')
    obj = trimesh.load(f'{subdir}/model.obj')

    if isinstance(obj, trimesh.Trimesh):
        vertices = obj.vertices.view(np.ndarray)
        faces = obj.faces.view(np.ndarray)

    else:
        obj = obj.dump().sum()
        vertices = obj.vertices.view(np.ndarray)
        faces = obj.faces.view(np.ndarray)

    pcd_full = sample_point_cloud(vertices, faces, 40000)
    export_pointcloud_to_obj(f'{subdir}/pcd_gt.obj', pcd_full)
    # for index in range(10):
    #     depthimg = OpenEXR.InputFile(f"{subdir}/depth000{index}.exr").channel('R')
    #     depth = np.frombuffer(depthimg, dtype=np.float32)
    #     rgb2mesh = RGBD2Mesh(depth, 512, 512, depthIntrinsics, depthExtrinsics[index])
    #     pcd_recon = rgb2mesh.to_pcd(10000)
    #     export_pointcloud_to_obj(f'{subdir}/pcd_{index}.obj', pcd_recon)
    sys.stderr.write(f'{subdir} done \n')

sys.stderr.write(f'all done \n')

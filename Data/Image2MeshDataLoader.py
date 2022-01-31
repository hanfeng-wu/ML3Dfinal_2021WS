"""
Dataloader for the image to mesh dataset
"""

import os
import torch
import numpy as np
import skimage.io as skio
from skimage.transform import resize
from skimage.color import rgb2gray
import trimesh
import pandas as pd
from pyntcloud import PyntCloud
from Data.binvox_rw import read_as_3d_array

class Image2MeshDataLoader():

    def __init__(self,images_path = "Assets/Data/Image2Mesh/train/images/",meshes_path = "Assets/Data/Image2Mesh/train/meshes/",image_size= 256, voxel_dims = (32,32,32), sample_rate = 4096):
        self.images_path = images_path
        self.meshes_path = meshes_path
        self.data = os.listdir(images_path)
        self.length = len(self.data)
        self.sample_rate =  sample_rate
        self.voxel_dims = voxel_dims
        self.image_size = image_size

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        available_images = os.listdir( self.images_path + "/"+ self.data[index] + "/")
        images = []
        for image_path in available_images:
            image = self.images_path + "/"+ self.data[index] + "/" + image_path
            image = skio.imread(image)
            image = resize(image,(self.image_size, self.image_size))
            image = np.expand_dims(rgb2gray(image),-1)
            image = (image/127.5) - 1  ## scale between -1 and 1
            images.append(image)
        images = torch.from_numpy(np.array(images)).permute(0,3,1,2)

        label = self.meshes_path + "/"+ self.data[index] + "/" + os.listdir( self.meshes_path + "/"+ self.data[index] + "/")[0]
        with open(label, "rb") as fptr:
            voxel = read_as_3d_array(fptr).astype(np.float32)
        if(voxel.shape!=self.voxel_dims):
            mesh = trimesh.voxel.ops.matrix_to_marching_cubes(voxel)
            label = self._point2vox_(mesh.sample(self.sample_rate),self.voxel_dims)
            label = torch.from_numpy(label.astype(np.uint8))
        else:
            label = torch.from_numpy(voxel.astype(np.uint8))
        return images.float(),label.float()

    def _point2vox_(self,points,dims=(32,32,32)):
        w,h,d = dims
        points = pd.DataFrame(points, columns=['x','y','z'])
        cloud = PyntCloud(points)
        voxelgrid_id = cloud.add_structure("voxelgrid", n_x=w, n_y=h, n_z=d)
        voxelgrid = cloud.structures[voxelgrid_id]
        x_cords = voxelgrid.voxel_x
        y_cords = voxelgrid.voxel_y
        z_cords = voxelgrid.voxel_z
        voxel = np.zeros((w, h, d)).astype(np.uint8)
        for x, y, z in zip(x_cords, y_cords, z_cords):
            voxel[x][y][z] = 1
        return voxel
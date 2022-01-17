"""
Dataloader for the image to mesh dataset
"""

import os
import torch
import numpy as np
import scipy.io as sio
import skimage.io as skio
from skimage.transform import resize
import trimesh
import pandas as pd
from pyntcloud import PyntCloud

class Image2MeshDataLoader():

    def __init__(self,images_path = "Assets/Data/Image2Mesh/images/",meshes_path = "Assets/Data/Image2Mesh/meshes/",image_size= 256, voxel_dims = (32,32,32), sample_rate = 4096):
        self.images_path = images_path
        self.meshes_path = meshes_path
        self.in_size =image_size
        self.data = os.listdir(images_path)
        self.length = len(self.data)
        self.sample_rate =  sample_rate
        self.voxel_dims = voxel_dims
        self.image_size = 128

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        
        available_images = os.listdir( self.images_path + "/"+ self.data[index] + "/")
        rand_indx = np.random.randint(0,len(available_images))
        image = self.images_path + "/"+ self.data[index] + "/" + available_images[rand_indx]
        label = self.meshes_path + "/"+ self.data[index] + "/model.mat"
        
        image = skio.imread(image)
        image = resize(image,(self.image_size, self.image_size))[:,:,:3] 
        image = torch.from_numpy(np.array(image,dtype=np.uint8))
        image = image.permute(2,0,1)

        voxel = sio.loadmat(label)['input'].astype(np.uint8)[0]
        mesh = trimesh.voxel.ops.matrix_to_marching_cubes(voxel)
        label = self._point2vox_(mesh.sample(self.sample_rate),self.voxel_dims)
        label = torch.from_numpy(label.astype(np.uint8))
        return image.float(),label.long()

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
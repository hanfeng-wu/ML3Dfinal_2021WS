"""PyTorch datasets for loading ShapeNet voxels and ShapeNet point clouds from disk"""
from typing import Dict, List
import torch
from pathlib import Path
import json
import numpy as np
import trimesh
import csv

from Data.binvox_rw import read_as_3d_array


class ShapeNetVox(torch.utils.data.Dataset):
    """
    Dataset for loading ShapeNet Voxels from disk
    """

    def __init__(self, shapenet_core_path: Path, shapenet_splits_csv_path: Path, split: str):
        """
        :param split: one of 'train', 'val' or 'overfit' - for training, validation or overfitting split
        """
        super().__init__()
        self._shapenet_core_path = shapenet_core_path
        
        # the format of model paths is f"{synsetId}/{modelId}"
        self._model_paths: List[str] = self._load_model_paths(shapenet_splits_csv_path=shapenet_splits_csv_path, split=split)

    def _load_model_paths(shapenet_splits_csv_path: Path, split: str) -> List[str]:
        assert split in ['train', 'val', 'overfit']

        with open(str(shapenet_splits_csv_path), 'r') as read_obj:
            csv_reader = csv.reader(read_obj)
            # skip header
            next(csv_reader)
            return [f"{row[1]}/{row[3]}" for row in csv_reader if row[4] == split]


    def __getitem__(self, index: int):
        """
        PyTorch requires you to provide a getitem implementation for your dataset.
        :param index: index of the dataset sample that will be returned
        :return: a dictionary of data corresponding to the shape. In particular, this dictionary has keys
                 "name", given as "<shape_category>/<shape_identifier>",
                 "voxel", a 1x32x32x32 numpy float32 array representing the shape
                 "label", a number in [0, 12] representing the class of the shape
        """
        # Get item associated with index, get class, load voxels with ShapeNetVox.get_shape_voxels
        model_path: str = self._model_paths[index]
        with open(self._shapenet_core_path / model_path / "model_normalized.solid.binvox", "rb") as fptr:
            voxels = read_as_3d_array(fptr).astype(np.float32)
        return {
            "name": model_path,
            "voxel": voxels[np.newaxis, :, :, :],  # we add an extra dimension as the channel axis, since pytorch 3d tensors are Batch x Channel x Depth x Height x Width
        }

    def __len__(self) -> int:
        """
        :return: length of the dataset
        """
        return len(self._model_paths)

    @staticmethod
    def move_batch_to_device(batch, device):
        """
        Utility method for moving all elements of the batch to a device
        :return: None, modifies batch inplace
        """
        batch['voxel'] = batch['voxel'].to(device)


class ShapeNetPoints(torch.utils.data.Dataset):
    num_classes = 13  # we'll be performing a 13 class classification problem
    dataset_path = Path("exercise_2/data/ShapeNetPointClouds/")  # path to point cloud data
    class_name_mapping = json.loads(Path("exercise_2/data/shape_info.json").read_text())  # mapping for ShapeNet ids -> names
    classes = sorted(class_name_mapping.keys())

    def __init__(self, split: str):
        """
        :param split: one of 'train', 'val' or 'overfit' - for training, validation or overfitting split
        """
        super().__init__()
        assert split in ['train', 'val', 'overfit']

        self.items = Path(f"exercise_2/data/splits/shapenet/{split}.txt").read_text().splitlines()  # keep track of shapes based on split

    def __getitem__(self, index):
        # Get item associated with index, get class, load points with ShapeNetPoints.get_point_cloud

        # Hint: Since shape names are in the format "<shape_class>/<shape_identifier>", the first part gives the class

        item = self.items[index]
        item_class = item.split("/")[0]

        return {
            "name": item,
            "points": ShapeNetPoints.get_point_cloud(item),
            "label": ShapeNetPoints.classes.index(item_class)  # Label is 0 indexed position in sorted class list, e.g. 02691156 is label 0, 02828884 is label 1 and so on.
        }

    def __len__(self):
        return len(self.items)

    @staticmethod
    def move_batch_to_device(batch, device):
        """
        Utility method for moving all elements of the batch to a device
        :return: None, modifies batch inplace
        """
        batch['points'] = batch['points'].to(device)
        batch['label'] = batch['label'].to(device)

    @staticmethod
    def get_point_cloud(shapenet_id):
        """
        Utility method for reading a ShapeNet point cloud from disk, reads points from obj files on disk as 3d numpy arrays
        :param shapenet_id: Shape ID of the form <shape_class>/<shape_identifier>, e.g. 03001627/f913501826c588e89753496ba23f2183
        :return: a numpy array representing the point cloud, in shape 3 x 1024
        """
        point_cloud = trimesh.load(ShapeNetPoints.dataset_path / f"{shapenet_id}.obj")
        return np.ndarray(point_cloud.vertices).T.astype(np.float32)

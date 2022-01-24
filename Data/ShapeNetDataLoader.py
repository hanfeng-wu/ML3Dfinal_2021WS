from typing import Dict, List, Optional
import torch
from pathlib import Path
import numpy as np
import csv
import os.path

from Data.binvox_rw import read_as_3d_array


class ShapeNetVoxelData(torch.utils.data.Dataset):
    """
    Dataset for loading ShapeNet Voxels from disk
    """

    def __init__(
        self, 
        shapenet_core_path: Path, 
        shapenet_splits_csv_path: Path,
        split: str, 
        overfit: bool = False, 
        synset_id_filter: Optional[List[str]] = None,
        voxel_filename: str = "models/model_normalized.solid.binvox"
    ):
        super().__init__()
        self._shapenet_core_path = shapenet_core_path
        self._overfit = overfit
        self._voxel_filename = voxel_filename
        
        # the format of model paths is f"{synsetId}/{modelId}"
        self._model_paths: List[str] = self._load_model_paths(shapenet_splits_csv_path, split, synset_id_filter)

    def _load_model_paths(self, shapenet_splits_csv_path: Path, split: str, synset_id_filter: Optional[List[str]]) -> List[str]:
        assert split in ['train', 'val', 'test']

        with open(str(shapenet_splits_csv_path), 'r') as read_obj:
            csv_reader = csv.reader(read_obj)
            # skip header
            next(csv_reader)
            paths = []
            for row in csv_reader:
                path = self._shapenet_core_path / f"{row[1]}/{row[3]}" / self._voxel_filename
                if (row[4] == split and (synset_id_filter is None or row[1] in synset_id_filter)):
                    if os.path.exists(path):
                        paths.append(path)
            return paths


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
        with open(model_path, "rb") as fptr:
            voxels = read_as_3d_array(fptr).astype(np.float32) 

        return voxels[np.newaxis, :, :, :]  # we add an extra dimension as the channel axis, since pytorch 3d tensors are Batch x Channel x Depth x Height x Width

    def __len__(self) -> int:
        """
        :return: length of the dataset
        """
        if self._overfit:
            return 16
        return len(self._model_paths)

    @staticmethod
    def move_batch_to_device(batch, device):
        """
        Utility method for moving all elements of the batch to a device
        :return: None, modifies batch inplace
        """
        batch['voxel'] = batch['voxel'].to(device)

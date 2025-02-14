import os
import cv2
import torch

import numpy as np

from torch.utils.data import Dataset

class Kermany(Dataset):

    def __init__(
        self,
        csv,
        root_dir,
        target,
        binary=True,
        transformations=None
    ):
        self.dataframe = csv
        self.root_dir = root_dir
        self.target = target
        self.binary = binary
        if self.binary:
            self.dataframe[
                self.target] = self.dataframe[self.target].map(
                    {
                        'normal': 0, 
                        'cnv':1, 
                        'drusen':1, 
                        'dme':1
                    }
                )
        self.transformations = transformations

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img, target = self._loading(idx)
        
        return img, target

    def _loading(self, idx):
        tgt = torch.tensor(
            self.dataframe[self.target][idx],
            dtype=torch.int64
        )
        
        img_name = os.path.join(
            self.root_dir,
            self.dataframe['p_fix'][idx],
            self.dataframe['filename'][idx]
        )
        
        img = cv2.cvtColor(
            cv2.imread(img_name),
            cv2.COLOR_BGR2RGB
        )

        img = self._pil2tensor(img, dtype=np.float32) / 255
        if self.transformations:
            img = self.transformations(img)

        return img, tgt

    def _pil2tensor(self, image, dtype):
        a = np.asarray(image)
        if a.ndim==2 : a = np.expand_dims(a, 2)
        a = np.transpose(a, (1, 0, 2))
        a = np.transpose(a, (2, 1, 0))
        return torch.from_numpy(a.astype(dtype, copy=False))
    
    def get_labels(self):
        """
        Required by ImbalancedDatasetSampler to retrieve labels.
        Returns a list (or array) of all labels in the dataset.
        """
        return self.dataframe[self.target].values
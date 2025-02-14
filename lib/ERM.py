import os
import cv2
import torch

import numpy as np
from torch.utils.data import Dataset


class ERMDataset(Dataset):
    
    def __init__(
        self, 
        csv, 
        root_dir, 
        target, 
        binary=False,
        transformations=None
    ):
        self.dataframe = csv
        self.root_dir = root_dir
        self.target = target
        self.binary = binary
        if self.binary:
            self.dataframe[
                self.target] = self.dataframe[self.target].map({0:0, 1:1, 2:1})
        self.transformations = transformations
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img, target = self._loading(idx)
        
        return img, target
        
    def _pil2tensor(self, image, dtype):
        a = np.asarray(image)
        if a.ndim==2 : a = np.expand_dims(a, 2)
        a = np.transpose(a, (1, 0, 2))
        a = np.transpose(a, (2, 1, 0))
        return torch.from_numpy(a.astype(dtype, copy=False))
    
    def square_padding(self, im: np.ndarray, desired_size: int = 384) -> np.ndarray:
        """Set image into the center and pad around.
        To better find edges and the corresponding circle in the fundus images.
        Args:
            im: Fundus image.
            add_pad: Constant border padding.
        Return:
            Padded image.
        """
        dim_y, dim_x = im.shape[:2]
        dim_larger = max(dim_x, dim_y)
        ratio = float(desired_size)/max(dim_x, dim_y)
        new_size = tuple([int(x*ratio) for x in [dim_y, dim_x]])
        
        im = cv2.resize(im, (new_size[1], new_size[0]))

        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)

        color = [1, 1, 1]
        new_im = cv2.copyMakeBorder(
            im, 
            top, 
            bottom, 
            left, 
            right, 
            cv2.BORDER_CONSTANT,
            value=color)
        
        return new_im
        
    def _loading(self, idx):
        tgt = torch.tensor(
            self.dataframe[self.target][idx],
            dtype=torch.int64
        )
        
        img_name = os.path.join(
            self.root_dir,
            self.dataframe['filenames'][idx]
        )
        
        img = cv2.cvtColor(
            cv2.imread(img_name),
            cv2.COLOR_BGR2RGB
        )

        img = self._pil2tensor(img, dtype=np.float32) / 255
        if self.transformations:
            img = self.transformations(img)

        return img, tgt
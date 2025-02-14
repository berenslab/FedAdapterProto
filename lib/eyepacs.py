import os
import cv2
import torch

import numpy as np

from torch.utils.data import Dataset

class FundusDataset(Dataset):
    '''
    Loading Retinal Fundus Dataset
    '''

    def __init__(self, data, tgt, root_dir, transformations):
        '''
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transformations (callable, optional): Optional transform to be 
            applied on a sample.
        '''
        self.data_frame = data
        self.root_dir = root_dir
        if tgt == 'diabetes':
            self.target = 'diagnosis_image_dr_level'
            self.data_frame[self.target] = self.data_frame[self.target].where(
                self.data_frame[self.target] == 0.0, 1.0
            )
        elif tgt == 'gender':
            self.target = 'patient_gender'
            self.data_frame[self.target] = self.data_frame[self.target].map(
                {"Female": 0, "Male": 1}
            )
        else:
            raise NotImplementedError()
        self.transformations = transformations
            
    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # images
        img_name = os.path.join(
            self.root_dir,
            self.data_frame['image_path'][idx]
        )
        try:
            img = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)

            img = self._pil2tensor(img, dtype=np.float32)/255
            if self.transformations:
                img = self.transformations(img)

            # labels
            target = torch.tensor(
                self.data_frame[self.target][idx],
                dtype=torch.int64
            )

            return img, target
        except cv2.error:
            print(img_name)

    def _pil2tensor(self, image, dtype):
        a = np.asarray(image)
        if a.ndim==2 : a = np.expand_dims(a,2)
        a = np.transpose(a, (1, 0, 2))
        a = np.transpose(a, (2, 1, 0))
        return torch.from_numpy(a.astype(dtype, copy=False))

    def get_labels(self):
        """
        Required by ImbalancedDatasetSampler to retrieve labels.
        Returns a list (or array) of all labels in the dataset.
        """
        return self.data_frame[self.target].values
import os
import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import random as ra
from utils import read_filepaths
from PIL import Image, ImageOps
import cv2
from matplotlib import cm
from torchvision import transforms
from batchgenerators.transforms import noise_transforms
from batchgenerators.transforms import spatial_transforms
from skimage.exposure import equalize_hist
from skimage.util import random_noise

COVIDxDICT = {'pneumonia': 0, 'normal': 1, 'COVID-19': 2}

def do_augmentation(image_tensor):
    
    array, _ = spatial_transforms.augment_mirroring(image_tensor, axes=(1, 2))
        
    # need to become [bs, c, x, y] before augment_spatial
    augmented = array[None, ...]
    r_range = (0, (3 / 360.) * 2 * np.pi)
    cval = 0.
        
    augmented, _ = spatial_transforms.augment_spatial(
        augmented, seg=np.ones_like(augmented), patch_size=[augmented.shape[2],augmented.shape[3]],
        do_elastic_deform=True, alpha=(0., 100.), sigma=(8., 13.),
        do_rotation=True, angle_x=r_range, angle_y=r_range, angle_z=r_range,
        do_scale=True, scale=(.9, 1.1),
        border_mode_data='constant', border_cval_data=cval,
        order_data=3,
        p_el_per_sample=0.5,
        p_scale_per_sample=.5,
        p_rot_per_sample=.5,
        random_crop=False
    )
    return augmented

class COVIDxDataset(Dataset):
    """
    Code for reading the COVIDxDataset
    """

    def __init__(self, mode, n_classes=3, dataset_path='/home/huaxu@gaps_domain.ssr.upm.es/COVID/COVIDNET/data', dim=(224, 224)):
        self.root = os.path.join(dataset_path,mode) 

        self.CLASSES = n_classes
        self.dim = dim
        self.COVIDxDICT = {'pneumonia': 0, 'normal': 1, 'COVID-19': 2}
        testfile = '/home/huaxu@gaps_domain.ssr.upm.es/COVID/COVIDNET/test_COVIDx_new.txt'
        trainfile = '/home/huaxu@gaps_domain.ssr.upm.es/COVID/COVIDNET/train_COVIDx_new.txt'
        if (mode == 'train'):
            self.paths, self.labels = read_filepaths(trainfile)
        elif (mode == 'test'):
            self.paths, self.labels = read_filepaths(testfile)
        print("{} examples =  {}".format(mode, len(self.paths)))
        self.mode = mode

    def __len__(self):
        
        return len(self.paths)
    
    def __getitem__(self, index):
        #index = int(index/2)
        image_tensor = self.load_image(os.path.join(self.root , self.paths[index]), self.dim)
        label_tensor = torch.tensor(self.COVIDxDICT[self.labels[index]], dtype=torch.long)
        image_tensor = image_tensor.numpy()

        if ra.random()>0.5:
            image_tensor = random_noise(image_tensor, mode='gaussian', mean=0.015, var = 0.015)
            
        if ((label_tensor.numpy() == 2 and ra.random()>0.17) or (label_tensor.numpy() ==0 and ra.random()>0.5)) and self.mode == 'train':
            

            augmented_tensor = do_augmentation(image_tensor)
            augmented_tensor = torch.from_numpy(augmented_tensor)
            augmented_tensor = torch.squeeze(augmented_tensor, dim=0)
              
            final_tensor = augmented_tensor
                
        else:
            final_tensor = torch.FloatTensor(image_tensor)
        return final_tensor, label_tensor

    def load_image(self, img_path, dim):
        if not os.path.exists(img_path):
            print("IMAGE DOES NOT EXIST {}".format(img_path))
        image = cv2.imread(img_path)
        image2 = np.copy(image)
        image2[image2>0]=255
        image2 = image2[:,:,0]
        mask = Image.fromarray(image2.astype('uint8'))
  
        img_adapteq = Image.fromarray(image.astype('uint8'), 'RGB')
        img_adapteq = ImageOps.equalize(img_adapteq,mask=mask)

        preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),#normaliza a [0,1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        image_tensor = preprocess(img_adapteq)
        
        return image_tensor



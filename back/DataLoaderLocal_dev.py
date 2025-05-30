from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np
import yaml
import torch.nn.functional as F
from skimage.util import view_as_blocks
from torchvision import datasets, transforms
import torch
import matplotlib.pyplot as plt




class DataLoaderLocal(Dataset):

    @staticmethod
    def yaml_loader(path):
        with open(path, 'r') as file:
            yaml_data = yaml.safe_load(file)
        file.close()
        return yaml_data

    def load_paths(self):
        images_folder = os.path.join(self.main_input_path, self.dataset_folder, 'images')
        masks_folder = os.path.join(self.main_input_path,self.dataset_folder, 'masks')
        self.images_list_paths = [os.path.join(images_folder, img) for img in np.sort(os.listdir(images_folder))]
        self.masks_list_paths = [os.path.join(masks_folder, mask) for mask in np.sort(os.listdir(masks_folder))]

    def convert_mask_to_one_hot(self, mask):
        #convert mask to one hot encoder
        label_seg = np.zeros(mask.size, dtype=np.uint8)
        for label, rgb_val in self.config['color_dict'].items():
            label_seg[np.all(rgb_val == np.array(mask), axis=-1)] = int(label)
        one_hot_encoding_mask = F.one_hot(torch.tensor(label_seg, dtype=torch.long), num_classes=self.n_classes).numpy()
        one_hot_encoding_mask_trans = np.transpose(one_hot_encoding_mask, (2, 0, 1))
        return one_hot_encoding_mask_trans

    def transform_to_nn(self, image, mask):
        transform = transforms.Compose([
            transforms.Resize((224, 244)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        one_hot_mask = self.convert_mask_to_one_hot(mask)
        image = transform(image)
        one_hot_mask = torch.tensor(one_hot_mask)

        return image, one_hot_mask


    def __getitem__(self, idx):
        image = Image.open(self.images_list_paths[idx]).convert('RGB')
        mask = Image.open(self.masks_list_paths[idx]).convert('RGB')

        # Preprocess for neural network
        image, one_hot_mask = self.transform_to_nn(image, mask)

        return image, one_hot_mask


    def __len__(self):
        return len(self.images_list_paths)


    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder
        self.main_input_path = "C:\Maor Nanikashvili\מאור - לימודים אקדמיים\בר אילן\תואר שני\תזה\datasets"
        self.data_loader_list = []
        self.images_list_paths = []
        self.masks_list_paths = []
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.config = self.yaml_loader('color_mapper_config.yml')
        self.n_classes = len(self.config['color_dict'].keys())
        self.load_paths()












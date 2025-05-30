from torchvision.models import vgg19
import torchvision.models as models
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np
import yaml
import torch.nn.functional as F
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import cv2


class SegmentationModel(nn.Module):
    """
    A segmentation model using VGG19 as the feature extractor, followed by upsampling layers
    to produce per-pixel class predictions. Suitable for semantic segmentation tasks.

    Args:
        num_classes (int): Number of output segmentation classes.
        pretrained_state (bool): Whether to use ImageNet-pretrained weights for VGG19.
    """
    def __init__(self, num_classes, pretrained_state):
        super(SegmentationModel, self).__init__()
        # Load VGG19 feature extractor with or without pretrained weights
        if pretrained_state == True:
            self.features = vgg19(pretrained=True).features
        else:
            self.features = vgg19(pretrained=False).features

        # Define upsampling path using transposed convolutions (deconvolution)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # Upsample by a factor of 2
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Upsample by a factor of 2
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Upsample by a factor of 2
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, num_classes, kernel_size=4, stride=2, padding=1),  # Upsample by a factor of 2
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        """
        Forward pass through feature extractor and upsampling layers.
        Args:
            x (torch.Tensor): Input image tensor of shape (B, 3, H, W)
        Returns:
            torch.Tensor: Output segmentation map of shape (B, num_classes, H, W)
        """
        x = self.features(x)
        x = self.upsample(x)
        return x



class DataLoaderLocal(Dataset):

    """
    Custom Dataset class for loading images and corresponding segmentation masks.
    Converts masks to one-hot encodings and applies VGG-style normalization and resizing.
    """

    @staticmethod
    def yaml_loader(path):
        """
        Loads a YAML configuration file.
        Args:
            path (str): Path to the YAML file.
        Returns:
            dict: Parsed YAML content.
        """
        with open(path, 'r') as file:
            yaml_data = yaml.safe_load(file)
        file.close()
        return yaml_data

    def load_paths(self):
        """
        Loads sorted lists of full paths for images and masks in the dataset.
        """
        # Build paths to images and masks subfolders
        images_folder = os.path.join(self.main_input_path, self.dataset_folder, 'images')
        masks_folder = os.path.join(self.main_input_path,self.dataset_folder, 'masks')
        self.images_list_paths = [os.path.join(images_folder, img) for img in np.sort(os.listdir(images_folder))]
        self.masks_list_paths = [os.path.join(masks_folder, mask) for mask in np.sort(os.listdir(masks_folder))]

    def convert_mask_to_one_hot(self, mask):
        """
        Converts a grayscale mask image to a one-hot encoded tensor.
        Args:
            mask (PIL.Image): Grayscale mask image.
        Returns:
            np.ndarray: One-hot encoded mask of shape (C, H, W).
        """
        # Convert to numpy grayscale
        label_seg = np.array(mask.convert('L'))
        # Apply one-hot encoding
        one_hot_encoding_mask = F.one_hot(torch.tensor(label_seg, dtype=torch.long), num_classes=self.n_classes).numpy()
        # Transpose from (H, W, C) → (C, H, W) for PyTorch
        one_hot_encoding_mask_trans = np.transpose(one_hot_encoding_mask, (2, 0, 1))
        return one_hot_encoding_mask_trans

    def transform_to_nn(self, image, mask):
        """
        Applies resizing, normalization, and tensor conversion to input image.
        Args:
            image (PIL.Image): RGB image.
            mask (PIL.Image): RGB or grayscale mask image.
        Returns:
            tuple: (torch.Tensor image, torch.Tensor one-hot encoded mask)
        """
        # Define VGG-style image transformations

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        one_hot_mask = self.convert_mask_to_one_hot(mask)
        image = transform(image)
        one_hot_mask = torch.tensor(one_hot_mask)

        return image, one_hot_mask


    def __getitem__(self, idx):
        """
        Retrieves and processes an image-mask pair by index.
        Args:
            idx (int): Index of sample.
        Returns:
            tuple: (image tensor, one-hot encoded mask tensor)
        """
        # Load and convert image and mask to PIL
        image_cv = cv2.imread(self.images_list_paths[idx], cv2.IMREAD_UNCHANGED)
        image = Image.fromarray(image_cv)
        mask_cv = cv2.imread(self.masks_list_paths[idx], cv2.IMREAD_UNCHANGED)
        mask = Image.fromarray(mask_cv).convert('RGB')

        # Preprocess for NN
        image, one_hot_mask = self.transform_to_nn(image, mask)

        return image, one_hot_mask


    def __len__(self):
        """
        Returns number of samples in dataset.
        """
        return len(self.images_list_paths)


    def __init__(self, dataset_folder):
        """
        Initializes the dataset by loading config and data paths.
        Args:
            dataset_folder (str): One of 'train', 'val', or 'test'.
        """
        self.dataset_folder = dataset_folder
        self.main_input_path = ".\\results\\sentilet_data\\full_dataset"
        self.data_loader_list = []
        self.images_list_paths = []
        self.masks_list_paths = []
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.config = self.yaml_loader(".\\input\\seg_mapping\\sentilet_color_mapper_config.yml")
        self.n_classes = len(self.config['labels_numbers_dict'].keys())
        self.load_paths()


def vgg_train_val_session(main_path, train_batch_size, val_batch_size, num_classes, lr, num_epochs):
    """
    Trains the VGG19-based segmentation model using training and validation datasets.
    Logs loss and accuracy metrics to TensorBoard and saves model weights periodically.
    Args:
        main_path (str): Base path to the dataset.
        train_batch_size (int): Batch size for training.
        val_batch_size (int): Batch size for validation.
        num_classes (int): Number of output segmentation classes.
        lr (float): Learning rate for the optimizer.
        num_epochs (int): Total number of training epochs.
    """

    # Set up TensorBoard logging directory
    log_dir = os.path.join(main_path.replace('sentilet_data\\full_dataset', ''), 'tensorboard_logs')
    writer = SummaryWriter(log_dir=log_dir)

    main_path = main_path
    # Initialize dataset and dataloaders
    dataset_train = DataLoaderLocal('train')
    dataloader_train = DataLoader(dataset_train, batch_size=train_batch_size, shuffle=True)
    dataset_val = DataLoaderLocal('val')
    dataloader_val = DataLoader(dataset_val, batch_size=val_batch_size, shuffle=True)

    # number of segmentation classes
    num_classes = num_classes
    # Initialize model with pretrained VGG19
    vgg19 = SegmentationModel(num_classes, True)

    # Define loss function
    criterion = nn.CrossEntropyLoss()
    # Define optimizer
    optimizer = torch.optim.Adam(vgg19.parameters(), lr=lr)
    # Training loop
    num_epochs = num_epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgg19.to(device)

    for epoch in tqdm(range(num_epochs)):
        vgg19.train()
        running_loss = 0.0
        correct_predictions = 0
        total_masks_predictions = 0
        total_indecies_predictions = 0

        for batch in tqdm(dataloader_train):
            inputs = batch[0]
            masks = batch[1]
            inputs, masks = inputs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = vgg19(inputs)
            loss = criterion(outputs, torch.argmax(masks, dim=1))

            # Compute accuracy
            predicted_labels = torch.argmax(outputs, dim=1)
            correct_predictions += torch.sum(predicted_labels == torch.argmax(masks, dim=1))
            total_masks_predictions += inputs.size(0)
            total_indecies_predictions += inputs.size(0)*inputs.size(2)*inputs.size(3)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Calculate average training loss and accuracy
        epoch_loss = running_loss / total_masks_predictions
        epoch_accuracy = correct_predictions / total_indecies_predictions

        # Validation loop
        vgg19.eval()
        val_running_loss = 0.0
        val_correct_predictions = 0
        val_total_masks_predictions = 0
        val_total_indecies_predictions = 0

        for val_batch in tqdm(dataloader_val):
            val_inputs = val_batch[0]
            val_masks = val_batch[1]
            val_inputs, val_masks = val_inputs.to(device), val_masks.to(device)

            with torch.no_grad():
                val_outputs = vgg19(val_inputs)
                val_loss =  criterion(val_outputs, torch.argmax(val_masks, dim=1))

                # Compute validation accuracy
                val_predicted_labels = torch.argmax(val_outputs, dim=1)
                val_correct_predictions += torch.sum(val_predicted_labels == torch.argmax(val_masks, dim=1))
                val_total_masks_predictions += val_inputs.size(0)
                val_total_indecies_predictions += val_inputs.size(0) * val_inputs.size(2) * val_inputs.size(3)

                val_running_loss += val_loss.item()

        val_epoch_loss = val_running_loss / val_total_masks_predictions
        val_epoch_accuracy = val_correct_predictions / val_total_indecies_predictions

        print(f"Epoch {epoch+1}, Training Loss: {epoch_loss}, Training Accuracy: {epoch_accuracy * 100}%, "
              f"Validation Loss: {val_epoch_loss}, Validation Accuracy: {val_epoch_accuracy * 100}%")
        
        # Save model every 5 epochs
        if epoch%5 == 0:
          torch.save(vgg19.state_dict(), f"//NN_weghits///new_model_weights_lr_0.0001_{str(epoch)}.pth")

    #Save final model
    torch.save(vgg19.state_dict(), "//NN_weghits//new_model_weights_lr_0.0001_final.pth")

    writer.close()



def vgg_test_session(num_classes, path_to_trained_weights, input_image_path, gt_mask_path):
    """
    Loads a trained model and evaluates it on a single input image.
    Displays the input frame, ground truth mask, and predicted segmentation mask.

    Args:
        num_classes (int): Number of segmentation classes.
        path_to_trained_weights (str): Path to model weights file (.pth).
        input_image_path (str): Path to input RGB image.
        gt_mask_path (str): Path to ground truth mask image.
    """
    # Load input image and ground truth mask
    num_classes = num_classes
    path_to_trained_weights = path_to_trained_weights
    path_to_image_pred = input_image_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_cv = cv2.imread(path_to_image_pred, cv2.IMREAD_UNCHANGED)
    image = Image.fromarray(image_cv)
    mask_cv_gt = cv2.imread(gt_mask_path, cv2.IMREAD_UNCHANGED)
    mask_gt = Image.fromarray(mask_cv_gt).convert('RGB')
    num_classes = num_classes
    # Load model without pretrained backbone
    vgg19 = SegmentationModel(num_classes, False)
    # Define transforms for inference
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image)

    vgg19.load_state_dict(torch.load(path_to_trained_weights, map_location=torch.device(device)))
    vgg19.eval()

    with torch.no_grad():
        predicted_output = vgg19(input_tensor)

    predicted_mask = torch.argmax(predicted_output, dim=0).numpy()

    # Plot input image, ground truth, and predicted mask
    plt.figure(figsize=(15, 5))

    # Input frame
    plt.subplot(1, 3, 1)
    plt.imshow(image_cv)  # Convert BGR to RGB
    plt.title("Input Frame")
    plt.axis("off")

    # Ground truth mask
    plt.subplot(1, 3, 2)
    plt.imshow(mask_cv_gt , cmap='jet')  # Adjust colormap if necessary
    plt.title("Ground Truth Mask")
    plt.axis("off")

    # Predicted mask
    plt.subplot(1, 3, 3)
    plt.imshow(predicted_mask,  cmap='jet')
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

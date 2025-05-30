from torchvision.models import vgg19
import os
import numpy as np
from PIL import Image
import torchvision.models as models
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
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
    def __init__(self, num_classes, pretrained_state):
        super(SegmentationModel, self).__init__()
        if pretrained_state == True:
            self.features = vgg19(pretrained=True).features
        else:
            self.features = vgg19(pretrained=False).features

        # Add upsampling layers to match the input spatial dimensions
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
            # Upsample by a factor of 2
        )
        # self.final_layer = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.features(x)
        x = self.upsample(x)
        # x = self.final_layer(x)
        return x





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
        # label_seg = np.zeros(mask.size, dtype=np.uint8)
        # for label, rgb_val in self.config['update_color_names_dict'].items():
        #     label_seg[np.all(rgb_val == np.array(mask), axis=-1)] = int(label)
        label_seg = np.array(mask.convert('L'))
        one_hot_encoding_mask = F.one_hot(torch.tensor(label_seg, dtype=torch.long), num_classes=self.n_classes).numpy()
        one_hot_encoding_mask_trans = np.transpose(one_hot_encoding_mask, (2, 0, 1))
        return one_hot_encoding_mask_trans

    def transform_to_nn(self, image, mask):
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

        image_cv = cv2.imread(self.images_list_paths[idx], cv2.IMREAD_UNCHANGED)#Image.open(self.images_list_paths[idx])
        image = Image.fromarray(image_cv)
        mask_cv = cv2.imread(self.masks_list_paths[idx], cv2.IMREAD_UNCHANGED)
        mask = Image.fromarray(mask_cv).convert('RGB')

        # Preprocess for neural network
        image, one_hot_mask = self.transform_to_nn(image, mask)

        return image, one_hot_mask


    def __len__(self):
        return len(self.images_list_paths)


    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder
        self.main_input_path = "C:\\Maor Nanikashvili\\thesis_pipeline\Pipeline\\results\\sentilet_data\\full_dataset"
        self.data_loader_list = []
        self.images_list_paths = []
        self.masks_list_paths = []
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.config = self.yaml_loader("C:\\Maor Nanikashvili\\thesis_pipeline\\Pipeline\\input\\seg_mapping\\sentilet_color_mapper_config.yml")
        self.n_classes = len(self.config['labels_numbers_dict'].keys())
        self.load_paths()



def vgg_train_val_session(main_path, train_batch_size,val_batch_size, num_classes, lr, num_epochs):
    #data loader
    main_path = main_path
    # load the data --> for the first time and initilize the loader class

    dataset_train = DataLoaderLocal('train')
    dataloader_train = DataLoader(dataset_train, batch_size=train_batch_size, shuffle=True)

    dataset_val = DataLoaderLocal('val')
    dataloader_val = DataLoader(dataset_val, batch_size=val_batch_size, shuffle=True)


    # number of segmentation classes
    num_classes = num_classes

    vgg19 = SegmentationModel(num_classes, True)

    # Define loss function (example: CrossEntropyLoss for segmentation)
    criterion = nn.CrossEntropyLoss()

    # Define optimizer (adjust learning rate and other parameters as needed)
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
        # loader_list = data_loader_train.load_data()



        for batch in tqdm(dataloader_train):
            inputs = batch[0]
            masks = batch[1]
            # print(masks.shape)
            inputs, masks = inputs.to(device), masks.to(device)
            # masks_resized = F.interpolate(masks, size=(7, 7), mode='nearest')
            optimizer.zero_grad()
            # outputs = seg_head(inputs)
            outputs = vgg19(inputs)
            loss = criterion(outputs, torch.argmax(masks, dim=1))
            # loss = criterion(outputs, F.interpolate
            # (torch.argmax(masks, dim=1).unsqueeze(1).float(), size=(244, 244), mode='nearest').squeeze(1).long())
            # interpolation above same as torch.argmax(masks, dim=1) only but the croosentropy class have to get the values like that

            # Compute accuracy
            predicted_labels = torch.argmax(outputs, dim=1)
            correct_predictions += torch.sum(predicted_labels == torch.argmax(masks, dim=1))
            # correct_predictions += torch.sum(predicted_labels == F.interpolate
            # (torch.argmax(masks, dim=1).unsqueeze(1).float(), size=(244, 244), mode='nearest').squeeze(1).long()).item()
            total_masks_predictions += inputs.size(0)  # the total number of inputs(or outputs) that the model have triend on in each epoch
            total_indecies_predictions += inputs.size(0)*inputs.size(2)*inputs.size(3) # the total number of inecies in each frame that the model tring to predict (224*224)


            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / total_masks_predictions
        epoch_accuracy = correct_predictions / total_indecies_predictions

        # Validation loop
        vgg19.eval()  # Set model to evaluation mode
        # val_loader_list = data_loader_val.load_data()
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
                # val_loss = criterion(val_outputs,
                #                      F.interpolate(torch.argmax(val_masks, dim=1).unsqueeze(1).float(), size=(244, 244),
                #                                    mode='nearest').squeeze(1).long())

                # Compute validation accuracy
                val_predicted_labels = torch.argmax(val_outputs, dim=1)
                val_correct_predictions += torch.sum(val_predicted_labels == torch.argmax(val_masks, dim=1))
                # val_correct_predictions += torch.sum(
                #     val_predicted_labels == F.interpolate(torch.argmax(val_masks, dim=1).unsqueeze(1).float(),
                #                                           size=(244, 244), mode='nearest').squeeze(1).long()).item()
                val_total_masks_predictions += val_inputs.size(0)
                val_total_indecies_predictions += val_inputs.size(0) * val_inputs.size(2) * val_inputs.size(3)

                val_running_loss += val_loss.item()



        val_epoch_loss = val_running_loss / val_total_masks_predictions
        val_epoch_accuracy = val_correct_predictions / val_total_indecies_predictions

        print(f"Epoch {epoch+1}, Training Loss: {epoch_loss}, Training Accuracy: {epoch_accuracy * 100}%, "
              f"Validation Loss: {val_epoch_loss}, Validation Accuracy: {val_epoch_accuracy * 100}%")

        if epoch%5 == 0:

          torch.save(vgg19.state_dict(), f"/content/drive/MyDrive/NN_weghits/new_model_weights_lr_0.0001_{str(epoch)}.pth")


    # Save the new weights of the model
    torch.save(vgg19.state_dict(), "/content/drive/MyDrive/NN_weghits/new_model_weights_lr_0.0001_final.pth")





def vgg_test_session(num_classes, path_to_trained_weights, input_image_path, gt_mask_path):


    num_classes = num_classes
    path_to_trained_weights = path_to_trained_weights
    path_to_image_pred = input_image_path

    image_cv = cv2.imread(path_to_image_pred, cv2.IMREAD_UNCHANGED)  # Image.open(self.images_list_paths[idx])
    image = Image.fromarray(image_cv)
    mask_cv_gt = cv2.imread(gt_mask_path, cv2.IMREAD_UNCHANGED)
    mask_gt = Image.fromarray(mask_cv_gt).convert('RGB')

    num_classes = num_classes

    vgg19 = SegmentationModel(num_classes, False)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    input_tensor = transform(image) # Add batch dimension


    vgg19.load_state_dict(torch.load(path_to_trained_weights, map_location=torch.device('cpu')))
    vgg19.eval()  # Set model to evaluation mode


    # Pass the input frame through the model to obtain the predicted mask
    with torch.no_grad():
        predicted_output = vgg19(input_tensor)

    predicted_mask = torch.argmax(predicted_output, dim=0).numpy()

    # predicted_mask_rgb = convert_mask_to_rgb(predicted_mask)

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

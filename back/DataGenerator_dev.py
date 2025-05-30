import cv2
import os
from torch.utils.data import Dataset
import numpy as np
from skimage.util import view_as_blocks
from sklearn.model_selection import train_test_split
import yaml
from yaml.loader import SafeLoader
import shutil
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import json
from glob import glob
from collections import defaultdict






class DataGenerator(Dataset):

    """
    Class: DataGenerator
    Description: A custom PyTorch dataset and utility class for handling image, mask, and multispectral data
    for machine learning workflows. Handles configuration loading, data augmentation, and dataset splitting.
    """

    def __init__(self):

        """
        Initializes default patch size, scene name, output paths, and loads configuration from YAML.
        """
        self.config = self.yaml_loader("\\back\\config_DataGenerator_dev.yaml")
        # Get root directory for input data from configuration
        self.root_dir = self.config['functions']['general_params']['root_dir']
        # Set patch size and scene name
        self.patch_size = self.config['functions']['general_params']['patch_size']
        self.general_scene_name = self.config['functions']['general_params']['general_scene_name']
        # Define output path and load configuration file
        self.main_output_path = self.config['functions']['general_params']['main_output_path']



    @staticmethod
    def yaml_dumper(data, path):
        """
        Dumps Python dictionary `data` into a YAML file at the given `path`.
        """
        # Save data to YAML file
        with open(path, 'w') as outfile:
            yaml.dump(data, outfile, sort_keys=False, default_flow_style=None)

    @staticmethod
    def yaml_loader(path):
        """
        Loads YAML configuration file from given path and returns it as a dictionary.
        """
        # Load YAML data into dictionary
        with open(path) as f:
            data = yaml.load(f, Loader=SafeLoader)
        f.close()
        return data


    @staticmethod
    def move_files(file_list, source_dir, destination_dir):
        """
        Moves (copies) list of files from source directory to destination directory.
        Creates destination directory if it doesn't exist.
        """
        # Iterate through files and copy them to destination
        for file_name in file_list:
            source_file = os.path.join(source_dir, file_name)
            destination_file = os.path.join(destination_dir, file_name)
            # Copy files after checking if destination exists

            if os.path.exists(destination_dir):
                shutil.copy2(source_file, destination_file)
            else:
                os.makedirs(destination_dir)
                shutil.copy2(source_file, destination_file)

    @staticmethod
    def move_candidates_to_dataset():
        """
        Splits image, mask, and multispectral files into train/val/test sets and moves them
        into organized folders. Uses paths defined in configuration.
        """
        # Load configuration and define paths from it
        data_generator_obj = DataGenerator()
        general_input_path = data_generator_obj.config['functions']['move_candidates_to_dataset']['full_path_general_input_dir']
        output_dataset_general_path = data_generator_obj.config['functions']['move_candidates_to_dataset']['full_path_to_dataset']

        # Define folder mappings for source and target
        input_folders_dict = dict.fromkeys(['images_dir', 'masks_dir', 'multispectral_dir'])
        new_folders_dict = dict.fromkeys(['train_images_dir', 'val_images_dir', 'test_images_dir', 'train_masks_dir',
                                          'val_masks_dir', 'test_masks_dir', 'train_multispectral_dir',
                                          'val_multispectral_dir', 'test_multispectral_dir'])

        # Paths to original folders
        for dir_name in ['images', 'masks', 'multispectral']:
            input_folders_dict[f'{dir_name}_dir'] = os.path.join(general_input_path, dir_name)

        for dir_name in ['train', 'val', 'test']:
            for sub_dir_name in ['images', 'masks', 'multispectral']:
                new_folders_dict[f'{dir_name}_{sub_dir_name}_dir'] = os.path.join(output_dataset_general_path, dir_name, sub_dir_name)


        # Get the list of files (assuming all folders have the same file names but different extensions)
        files = os.listdir(input_folders_dict['images_dir'])

        # Split the files into train, val, and test
        train_files, temp_files = train_test_split(files, test_size=0.2, random_state=42, shuffle=True)
        val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42, shuffle=True)


        # Move the files
        for file_list, folder_type in [(train_files, 'train'), (val_files, 'val'), (test_files, 'test')]:
            if folder_type == 'train':
                data_generator_obj.move_files(train_files, input_folders_dict['images_dir'], new_folders_dict['train_images_dir'])
                data_generator_obj.move_files(train_files, input_folders_dict['masks_dir'], new_folders_dict['train_masks_dir'])
                data_generator_obj.move_files([file.replace('.png', '.npy') for file in train_files], input_folders_dict['multispectral_dir'], new_folders_dict['train_multispectral_dir'])
            elif folder_type == 'val':
                data_generator_obj.move_files(val_files, input_folders_dict['images_dir'], new_folders_dict['val_images_dir'])
                data_generator_obj.move_files(val_files, input_folders_dict['masks_dir'], new_folders_dict['val_masks_dir'])
                data_generator_obj.move_files([file.replace('.png', '.npy') for file in val_files], input_folders_dict['multispectral_dir'], new_folders_dict['val_multispectral_dir'])
            elif folder_type == 'test':
                data_generator_obj.move_files(test_files, input_folders_dict['images_dir'], new_folders_dict['test_images_dir'])
                data_generator_obj.move_files(test_files, input_folders_dict['masks_dir'], new_folders_dict['test_masks_dir'])
                data_generator_obj.move_files([file.replace('.png', '.npy') for file in test_files], input_folders_dict['multispectral_dir'], new_folders_dict['test_multispectral_dir'])

    @staticmethod
    def bin_image(tup):
        """
        The function bins images using the kmeans model.
        :param tup: A tuple containing three fields:
        0: Image path
        1: kmeans model
        2: Location for saving
        evaluating metric for amount of rgb clusters on all data set
        """
        img_loc = tup[0]
        kmeans_model = tup[1]
        save_loc = tup[2]
        img = cv2.imread(img_loc)
        flat_img = np.reshape(img, (-1, 3))
        resulting_img_labels = kmeans_model.predict(flat_img)
        cluster_centers = kmeans_model.cluster_centers_

        # Translating from cluster id to centroid
        resulting_rgb_list = [cluster_centers[i] for i in resulting_img_labels]
        binned_img = np.reshape(resulting_rgb_list, img.shape)

        cv2.imwrite(save_loc, binned_img)



    @staticmethod
    def parallel_get_all_rgb(img_path):
        """
        The function calculates the unique base RGBs of
        images and allows for parallel calculation.
        :param img_path: Path to image.
        :return unique_brgbs: List of unique base RGBs in the frame.
        """
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        flat_img = np.reshape(img, (-1, 3))
        unique_rgbs = np.unique(flat_img, axis=0)

        return unique_rgbs



    def save_images(self, file_name, folder_type_name, image_type, image):
        """
        Saves an image or NumPy array to the appropriate folder structure, creating folders as needed.
        Parameters:
            file_name (str): The name to save the file under.
            folder_type_name (str): Folder category (e.g., 'patches_after_anno').
            image_type (str): Type of data ('images', 'masks', 'multispectral').
            image (np.ndarray): The image or data to save.
        """
        # Save standard images (masks or RGB images) using cv2
        if image_type == 'masks' or image_type == 'images':
            if os.path.exists(os.path.join(self.main_output_path, folder_type_name, image_type)):
                cv2.imwrite(os.path.join(self.main_output_path, folder_type_name, image_type, file_name), image)
            else:
                os.makedirs(os.path.join(self.main_output_path,  folder_type_name, image_type))
                cv2.imwrite(os.path.join(self.main_output_path, folder_type_name, image_type, file_name), image)

        # Save numpy arrays (e.g., multispectral data)
        else:
            if os.path.exists(os.path.join(self.main_output_path, folder_type_name, image_type)):
                np.save(os.path.join(self.main_output_path, folder_type_name, image_type, file_name), image)
            else:
                os.makedirs(os.path.join(self.main_output_path,  folder_type_name, image_type))
                np.save(os.path.join(self.main_output_path, folder_type_name, image_type, file_name), image)


    def crop_image_or_mask_to_closest_size(self, frame):
        """
        Crops an image or mask to the largest size divisible by the patch size (224).
        Parameters:
            frame (np.ndarray): Input image or mask.
        Returns:
            np.ndarray: Cropped image/mask.
        """
        # Calculate dimensions divisible by patch size
        SIZE_X = (frame.shape[1] // self.patch_size) * self.patch_size  # Nearest size divisible by our patch size
        SIZE_Y = (frame.shape[0] // self.patch_size) * self.patch_size  # Nearest size divisible by our patch size

        # Return cropped region of interest
        cropped_frame = frame[0:SIZE_Y, 0:SIZE_X]

        return cropped_frame


    def rotate_and_flip_frames(self, image, names_tup):
        """
        Applies rotation and flipping augmentations to a given image and saves the result.
        Parameters:
            image (np.ndarray): Image to be transformed.
            names_tup (tuple): Tuple containing (image_name, file_type, image_category).
        """
        # Unpack tuple of names
        image = image
        patch_image_name, image_file_type, image_type = names_tup
        angle_list = [0,90,180,270]
        # Save transformed patches
        for i, angle in enumerate(angle_list):  # 4 rotations
            if angle == 90:
                rotated_patch_image = np.rot90(image, k=-1)  # 90 degrees clockwise
            elif angle == 180:
                rotated_patch_image = np.rot90(image, k=2)  # 180 degrees
            elif angle == 270:
                rotated_patch_image = np.rot90(image, k=-3)  # 270 degrees clockwise
            else:
                rotated_patch_image = image  # No rotation for 0 degrees

            for j in range(4):  # 2 flips
                frame_number = str((i * len(angle_list)) + (j + 1))

                if j == 0:
                    flipped_image = np.fliplr(rotated_patch_image)  # Flip horizontally
                elif j == 1:
                    flipped_image = np.flipud(rotated_patch_image)  # Flip vertically
                elif j == 2:
                    flipped_image = np.flipud(np.fliplr(rotated_patch_image))  # Flip both vertically and horizontally
                else:
                    flipped_image = rotated_patch_image  # No flip, just use the rotated image

                self.save_images(patch_image_name +f'_{frame_number}' +image_file_type,
                                 'patches_after_anno', image_type, flipped_image)



    def create_data(self):
        """
        Loads specific RGB, multispectral, and mask images from a directory.
        Processes, normalizes, crops, and saves them. Finally, splits cropped images
        into 224x224 patches and saves each patch separately by type (image, mask, multispectral).
        """
        # Get sorted list of all files in root directory
        set_images_list = np.sort(os.listdir(self.root_dir))
        # Convert selected image files to grayscale for RGB composition
        b_graylevel = cv2.cvtColor(cv2.imread(os.path.join(self.root_dir, set_images_list[2])), cv2.COLOR_BGR2GRAY)
        g_graylevel =  cv2.cvtColor(cv2.imread(os.path.join(self.root_dir, set_images_list[3])), cv2.COLOR_BGR2GRAY)
        r_graylevel =  cv2.cvtColor(cv2.imread(os.path.join(self.root_dir, set_images_list[4])), cv2.COLOR_BGR2GRAY)
        # Stack BGR channels into an RGB image
        full_rgb_image = np.stack((r_graylevel, g_graylevel, b_graylevel), axis=-1)
        mask_gray_level = cv2.cvtColor(cv2.imread(os.path.join(self.root_dir, set_images_list[11])), cv2.COLOR_BGR2GRAY)

        full_multyspectral_list = []
        # Load and process multispectral bands from selected image indices
        for data_list in (set_images_list[2:8], set_images_list[10], set_images_list[8:10]):
            if isinstance(data_list, np.ndarray):
                for band_data_path in data_list:
                    band_data = cv2.imread(os.path.join(self.root_dir, band_data_path), cv2.IMREAD_UNCHANGED)
                    band_data = band_data * 0.0001
                    band_data = np.clip(band_data, a_min=0, a_max=0.95)
                    full_multyspectral_list.append(band_data)
            else:
                band_data = cv2.imread(os.path.join(self.root_dir, data_list), cv2.IMREAD_UNCHANGED)
                band_data = band_data * 0.0001
                band_data = np.clip(band_data, a_min=0, a_max=0.95)
                full_multyspectral_list.append(band_data)

        # Stack all normalized bands into a full multispectral image
        full_multyspectral_image = np.stack(tuple(full_multyspectral_list), axis=-1)

        # Crop region of interest for all image types (fixed area of image)
        full_multyspectral_image_crop = full_multyspectral_image[2100:5490, 0:5490]
        mask_gray_level_crop = mask_gray_level[2100:5490, 0:5490]
        full_rgb_image_crop = full_rgb_image[2100:5490, 0:5490]

        # Save full images (non-cropped)
        for image in [(full_multyspectral_image, '.npy', 'multispectral'), (mask_gray_level, '.png', 'masks'),
                      (full_rgb_image, '.png', 'images')]:
            self.save_images(self.general_scene_name + image[1],
                             'full_data', image[2], image[0])

        # Delete full images and list to save memory
        del full_multyspectral_image
        del mask_gray_level
        del full_rgb_image
        del full_multyspectral_list

        # Crop to nearest patch-aligned size and save patches
        for image in [(full_multyspectral_image_crop, '.npy', 'multispectral'), (mask_gray_level_crop, '.png', 'masks'), (full_rgb_image_crop, '.png', 'images')]:
            crop_image = self.crop_image_or_mask_to_closest_size(image[0])
            self.save_images(self.general_scene_name + image[1],
                             'roi_crop_anno', image[2], crop_image)
            if len(crop_image.shape) == 3:
                patches_image = view_as_blocks(crop_image,
                                           block_shape=(self.patch_size, self.patch_size, crop_image.shape[2])) # Multichannel patch
            else:
                patches_image = view_as_blocks(crop_image,
                                           block_shape=(self.patch_size, self.patch_size)) # Single-channel patch

            for i in range(patches_image.shape[0]):
                for j in range(patches_image.shape[1]):
                    frame_number = str((i * patches_image.shape[1]) + (j + 1))
                    if len(patches_image.shape) == 6:
                        patch_image = patches_image[i, j, 0] # Multichannel patch
                    else:
                        patch_image = patches_image[i, j] #  Single-channel patch
                    patch_image_name = self.general_scene_name + f'_patch_{frame_number}'
                    self.save_images(patch_image_name + image[1],
                                     'patches_anno', image[2], patch_image)



    def create_masks_from_json_files(self):
        """
        Converts polygon annotations from JSON files into 2D mask images.
        Uses label mappings from the config YAML to assign numeric values to each segment type.
        Saves each mask as a `.png` image with the same name as the corresponding JSON file.
        """
        # Load input/output paths and label dictionary

        json_input_folder = self.config['functions']['create_masks_from_json_files']['input_json_folder']
        mask_output_path = self.config['functions']['create_masks_from_json_files']['output_mask_folder']
        json_files = glob(os.path.join(json_input_folder, "*.json"))
        LABEL_MAP = self.yaml_loader(self.config['functions']['create_masks_from_json_files']['labels_dict'])['labels_numbers_dict']

        # Iterate over all JSON files in the directory
        for json_file in json_files:
            full_json_path = os.path.join(json_input_folder, json_file)
            with open(full_json_path, 'r') as file:
                data = json.load(file)

            # Create an empty mask with default class
            mask = np.full((self.patch_size, self.patch_size), LABEL_MAP["lands_compounds"], dtype=np.uint8)

            # Draw polygons into the mask based on their label
            for shape in data["shapes"]:
                label = shape["label"]
                points = np.array(shape["points"])
                points = np.round(points).astype(np.int32)

                if label in LABEL_MAP:
                    cv2.fillPoly(mask, [points], LABEL_MAP[label])

            # Save mask image with same basename as original JSON
            mask_filename = os.path.splitext(os.path.basename(full_json_path))[0] + ".png"
            cv2.imwrite(os.path.join(mask_output_path, mask_filename), mask)


    def create_augmantations_for_masks(self):
        """
        Applies augmentations (rotation + flip) to all patches in the dataset:
        RGB images, their corresponding masks, and multispectral patches.
        This generates transformed versions and saves them under `patches_after_anno`.
        """
        # Get path to image patches folder
        patches_input_path = os.path.join(self.main_output_path, 'patches_after_anno', 'images')
        for image_file in np.sort(os.listdir(patches_input_path)):
            full_image_patch_path = os.path.join(patches_input_path, image_file)
            full_mask_patch_path = os.path.join(patches_input_path.replace('images','masks'), image_file)
            full_multy_patch_path = os.path.join(patches_input_path.replace('images', 'multispectral'), image_file.replace('.png', '.npy'))
            # Apply rotation + flip augmentations on all three types (image/mask/multispectral)
            for path in [full_image_patch_path, full_mask_patch_path, full_multy_patch_path]:
                patch_image_name = os.path.basename(path).split('.')[0]
                file_type = '.' + os.path.basename(path).split('.')[-1]
                file_folder_name =   os.path.basename(os.path.dirname(path))
                # Load image or array depending on file type
                if file_type.split('.')[-1] == 'npy':
                    patch_image = np.load(path)
                else:
                    patch_image = cv2.imread(path, cv2.IMREAD_UNCHANGED)

                # Save 16 augmented versions (4 rotations × 4 flips) for each patch
                self.rotate_and_flip_frames(patch_image, (patch_image_name, file_type, file_folder_name))





    def extract_segment_colors(self,rgb_image, mask_image):
        """
        For each segment in the mask, collects all unique RGB values from the RGB image.
        Parameters:
            rgb_image (np.ndarray): RGB image of shape (H, W, 3).
            mask_image (np.ndarray): Grayscale mask with labeled segments.
        Returns:
            dict: Mapping of segment label -> unique RGB colors present.
        """
        # Find all unique segment IDs
        unique_segments = np.unique(mask_image)
        segment_colors = {}
        # For each segment ID, get all pixels in that region from the RGB image
        for segment in unique_segments:
            mask_indices = np.where(mask_image == segment)
            segment_rgb_values = rgb_image[mask_indices]  # Extract corresponding RGB values
            segment_colors[segment] = np.unique(segment_rgb_values, axis=0)  # Keep unique colors

        return segment_colors

    def find_optimal_clusters(self, seg, data):
        """
        Evaluates clustering quality for a given segment using silhouette score and inertia.
        Saves two plots: silhouette vs. K and elbow curve (inertia vs. K).
        Parameters:
            seg (int): Segment label for which clustering is evaluated.
            data (np.ndarray): RGB values to be clustered.
        """
        silhouette_scores = []
        K_range = [2,10,20,30,40,50,60,70,80,90,100]

        # Range of K values to test
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data)
            score = silhouette_score(data, kmeans.labels_)
            silhouette_scores.append(score)

        # Plot silhouette scores vs. K
        plt.figure(figsize=(8, 5))
        plt.plot(K_range, silhouette_scores, 'bo-', markersize=8)
        plt.xlabel('Number of clusters (K)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score Method for Optimal K')
        plt.grid(True)
        # plt.show()
        os.makedirs(os.path.join(self.main_output_path, 'patches_after_anno', 'metrics', str(seg)) , exist_ok=True)
        plt.savefig(os.path.join(self.main_output_path, 'patches_after_anno', 'metrics', str(seg) ,'Silhouette_number_of_k_graph.png'))


        # Evaluate clustering inertia (for elbow method)
        inertia = []
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data)
            inertia.append(kmeans.inertia_)

        # Plot the elbow curve
        plt.figure(figsize=(8, 5))
        plt.plot(K_range, inertia, 'bo-', markersize=8)
        plt.xlabel('Number of clusters (K)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method For Optimal K')
        plt.grid(True)
        plt.savefig(os.path.join(self.main_output_path, 'patches_after_anno', 'metrics', str(seg),'Inertia_number_of_k_graph.png'))

    def process_images(self, rgb_frames, mask_frames):
        """
        Processes multiple (RGB, mask) pairs to extract segment-wise colors
        and then evaluate KMeans clustering performance for each segment.
        Parameters:
            rgb_frames (list): List of file paths to RGB images.
            mask_frames (list): List of file paths to corresponding mask images.
        """

        segment_colors_list = []
        # Iterate over all image-mask pairs
        for i, (rgb_path, mask_path) in enumerate(zip(rgb_frames, mask_frames)):
            print(f"Processing image {i + 1}/{len(rgb_frames)}")
            rgb_image = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
            mask_image = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)  # Grayscale mask

            segment_colors = self.extract_segment_colors(rgb_image, mask_image)

            segment_colors_list.append(segment_colors)

        # Merge segment colors across all images
        merged_dict = defaultdict(list)

        for d in segment_colors_list:
            for key, value in d.items():
                merged_dict[key].append(value)  # Append values to the corresponding key

        # Final dict: segment -> unique RGB values from all images
        final_segment_colors_dict =  {key: np.unique(np.vstack(val), axis=0) for key , val in dict(merged_dict).items()}

        # Run clustering evaluation for each segment
        for segment, colors in final_segment_colors_dict.items():
            self.find_optimal_clusters(segment, colors)


    def check_for_k_cluster_number(self):
        """
        Wrapper function to run segment-wise color clustering analysis for all RGB/mask pairs.
        Outputs silhouette and elbow plots per segment label.
        """
        dg = DataGenerator()
        folder_path = dg.config['functions']['check_for_k_cluster_number']['folder_path']
        # Example usage
        rgb_frames = np.sort([os.path.join(folder_path , "images", file) for file in os.listdir(os.path.join(folder_path, 'images'))])
        mask_frames = np.sort([os.path.join(folder_path , "masks", file) for file in os.listdir(os.path.join(folder_path, 'masks'))])
        cluster_results = self.process_images(rgb_frames, mask_frames)

        print(cluster_results)



    def check_for_seg_classes(self):
        """
        Iterates through all mask images in the given folder and collects unique class values.
        Args:
            folder_path (str): Path to the folder containing mask images.
        Returns:
            List[int]: Sorted list of unique class values across all masks.
        """
        folder_path = self.config['functions']['check_for_seg_classes']['folder_path']

        unique_classes = set()

        # Iterate through all files in the folder
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            # Ensure it's an image file
            if file_name.lower().endswith(('.png')):
                # Read the mask image
                mask = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                if mask is None:
                    continue

                # Add unique pixel values to the set
                unique_classes.update(np.unique(mask))

        print(f'the list of unique classes is : {unique_classes}')





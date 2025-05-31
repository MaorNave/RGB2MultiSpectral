import os.path
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
import json
import pickle
import cv2
import yaml
from yaml.loader import SafeLoader
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import os
import spectral
from tqdm import tqdm
import random



class Materials_test():
    """
    A class for evaluating and visualizing hyperspectral material reconstruction results.
    Inputs:
    - Configuration file 'config_Materials_test.yaml' containing paths to input/output folders
    Process:
    - Loads config and provides tools for:
      - Folder and file I/O (YAML, JSON, Pickle)
      - Computing evaluation metrics (SSIM, MSE, RMSE, SAM)
      - Applying texture masking to predictions
      - Comparing spectral signatures and rendering analysis plots
    Output:
    - Used for evaluation and visualization tasks on predicted vs ground-truth hyperspectral cubes
    """

    def __init__(self):
        with open('config_Materials_test.yaml') as f:
            data = yaml.load(f, Loader=SafeLoader)
        f.close()
        self.config = data
        # Extract key directory paths from config
        self.path = {'main_pipeline_path': self.config['general_params']['main_pipeline_path'],
            'results': self.config['general_params']['results_path'],
                  'test': self.config['general_params']['test_path'],
                     'input': self.config['general_params']['input_path']}


    """ - -----------------------------  Static methods - ----------------------------------------- """


    @staticmethod
    def folder_checker(path):
        """
        Check if a folder exists at the given path. If not, create it.
        Inputs:
        - path: Folder path to validate or create
        Process:
        - Verifies folder existence
        - Creates folder if it does not exist
        Output:
        - Prints status message (exists or created)
        """
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Folders created for the following path : {path}" )
        else:
            print(f" Folder path are already exist for : {path}")

    @staticmethod
    def yaml_dumper(data, path):
        """
        Save a Python dictionary to a YAML file.
        Inputs:
        - data: Dictionary to save
        - path: Destination file path
        Process:
        - Uses PyYAML to write the data into a .yaml file
        Output:
        - YAML file saved at specified path
        """
        with open(path, 'w') as outfile:
            yaml.dump(data, outfile, sort_keys=False, default_flow_style=None)

    @staticmethod
    def json_dumper(path, data):
        """
        Write a dictionary to a JSON file.
        Inputs:
        - path: Destination file path
        - data: Dictionary to serialize
        Process:
        - Opens a file and dumps the JSON content
        Output:
        - JSON file written to disk
        """
        with open(path, 'w') as f:
            json.dump(data, f)
        f.close()

    @staticmethod
    def pickle_dumper(path, data):
        """
        Serialize and write a Python object using Pickle.
        Inputs:
        - path: File path to save the pickle
        - data: Python object to serialize
        Process:
        - Opens file in binary write mode and dumps object
        Output:
        - Pickle file saved
        """
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        f.close()

    @staticmethod
    def pickle_loader(path):
        """
        Load and deserialize a Python object from Pickle.
        Inputs:
        - path: Path to the .pkl file
        Process:
        - Opens file in binary read mode and loads the object
        Output:
        - Returns the unpickled Python object
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        f.close()
        return data

    @staticmethod
    def json_loader(path):
        """
        Load a JSON file and return the parsed dictionary.
        Inputs:
        - path: Path to the .json file
        Process:
        - Opens file and uses json.load to parse content
        Output:
        - Dictionary object loaded from file
        """
        with open(path, 'r') as f:
            data = json.load(f)
        f.close()
        return data


    @staticmethod
    def yaml_loader(path):
        """
        Load a YAML file and return the parsed dictionary.
        Inputs:
        - path: Path to the .yaml file
        Process:
        - Uses SafeLoader from PyYAML to parse YAML content
        Output:
        - Parsed dictionary from YAML file
        """
        with open(path) as f:
            data = yaml.load(f, Loader=SafeLoader)
        f.close()
        return data


    @staticmethod
    def compute_ssim_for_material_sig(img1, img2, clu_image, data_dict):
        """
        Compute SSIM per pixel across spectral bands and extract signatures for high-similarity pixels.
        Inputs:
        - img1: Ground-truth image, shape (224, 224, 9)
        - img2: Predicted image, shape (224, 224, 9)
        - clu_image: Clustered RGB image, shape (224, 224, 3)
        - data_dict: Dictionary containing material label mappings
        Process:
        - Compute SSIM for each spectral band (channel-wise)
        - Identify pixels with average SSIM > 0.7
        - For each such pixel, extract GT & predicted signature and its associated material info
        Output:
        - A list of tuples: (GT signature, predicted signature, material metadata)
        """
        ssim_maps = []
        full_materials_data_list = []
        # Compute SSIM map for each spectral channel
        for i in range(img1.shape[-1]):
            ssim_map = \
            ssim(img1[:, :, i], img2[:, :, i], data_range=img1[:, :, i].max() - img1[:, :, i].min(),
                 full=True)[1]
            ssim_maps.append(ssim_map)
        # Select pixels with high SSIM values
        mean_ssim_per_pixel = np.mean(np.dstack(ssim_maps), axis=-1)
        high_ssim_pixels = np.argwhere(mean_ssim_per_pixel > 0.7)

        for indices in high_ssim_pixels:
            pred_material_sig = img2[indices[0], indices[1], :]
            gt_material_sig = img1[indices[0], indices[1], :]
            clu_value = clu_image[indices[0], indices[1], :]
            clu_str_value = str(clu_value)
            for seg_key in data_dict:
                for clu_key in data_dict[seg_key]:
                    if clu_key == clu_str_value:
                        material_data = data_dict[seg_key][clu_key]
                        full_materials_data_list.append((gt_material_sig, pred_material_sig , material_data))

        return full_materials_data_list

    @staticmethod
    def compute_mse(img1, img2):
        """
        Compute Mean Squared Error (MSE) between two images.
        Inputs:
        - img1, img2: Ground truth and predicted images (same shape)
        Process:
        - Flattens both images and computes MSE using sklearn
        Output:
        - Scalar MSE value
        """
        mse = mean_squared_error(img1.flatten(), img2.flatten())
        return mse


    @staticmethod
    def compute_mse_rgb(img1, img2):
        """
        Compute normalized MSE between two RGB images.
        Inputs:
        - img1, img2: RGB images (same shape, 3 channels)
        Process:
        - Normalizes each image to [0, 1]
        - Computes MSE over flattened arrays
        Output:
        - Scalar MSE value
        """
        img1_norm = (img1-np.min(img1)) / (np.max(img1)-np.min(img1))
        img2_norm = (img2-np.min(img2)) / (np.max(img2)-np.min(img2))

        mse = mean_squared_error(img1_norm.flatten(), img2_norm.flatten())
        return mse

    @staticmethod
    def compute_ssim(img1, img2):
        """
        Compute average SSIM across all channels.
        Inputs:
        - img1, img2: 3D images of shape (H, W, C)
        Process:
        - For each channel, compute SSIM using skimage
        - Average the SSIM values over all channels
        Output:
        - Scalar average SSIM value
        """
        ssim_per_channel = [
            ssim(img1[:, :, i], img2[:, :, i], data_range=img1[:, :, i].max() - img1[:, :, i].min())
            for i in range(img1.shape[2])
        ]
        return np.mean(ssim_per_channel)

    @staticmethod
    def compute_rmse(img1, img2):
        """
        Compute normalized RMSE between two images.
        Inputs:
        - img1, img2: Ground truth and predicted images (same shape)
        Process:
        - Normalize images to [0, 1]
        - Compute MSE and take square root to get RMSE
        Output:
        - Scalar RMSE value
        """
        img1_norm = (img1-np.min(img1)) / (np.max(img1)-np.min(img1))
        img2_norm = (img2-np.min(img2)) / (np.max(img2)-np.min(img2))

        mse = mean_squared_error(img1_norm.flatten(), img2_norm.flatten())
        rmse = np.sqrt(mse)
        return rmse

    @staticmethod
    def compute_sam(img1, img2):
        """
        Compute Spectral Angle Mapper (SAM) between corresponding pixels.
        Inputs:
        - img1, img2: Hyperspectral images of shape (H, W, B)
        Process:
        - For each pixel, compute the angle (in radians) between the spectral vectors
        - Averages the angles over the entire image
        Output:
        - Scalar average SAM value in radians
        """
        eps = 1e-10  # Small value to prevent division by zero
        sam_values = np.zeros((img1.shape[0], img1.shape[1]))  # Per-pixel SAM
        # Compute SAM for each pixel
        for i in range(img1.shape[0]):
            for j in range(img1.shape[1]):
                vec1 = img1[i, j, :]
                vec2 = img2[i, j, :]
                # Normalize vectors
                norm1 = np.linalg.norm(vec1) + eps
                norm2 = np.linalg.norm(vec2) + eps

                # Compute spectral angle (cosine similarity)
                cosine_similarity = np.dot(vec1, vec2) / (norm1 * norm2)
                sam_values[i, j] = np.arccos(np.clip(cosine_similarity, -1, 1))  # Angle in radians

        return np.mean(sam_values)  # Average SAM over all pixels


    """ - -----------------------------  materials_tests- ----------------------------------------- """


    def apply_texture(self, rgb_img, cube):
        """
        Apply a grayscale texture from the RGB image onto the spectral cube.
        Inputs:
        - rgb_img: Input RGB image, shape (H, W, 3)
        - cube: Hyperspectral or multispectral cube, shape (H, W, C)
        Process:
        - Convert RGB image to grayscale and normalize it to [0, 1]
        - Ensure minimum texture value is not zero (replaced by 0.025)
        - Stack the grayscale image across all channels of the cube
        - Multiply original cube values by texture to emphasize image structure
        Output:
        - Cube with texture applied, same shape as input cube
        """

        rgb_gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        normalized_image = cv2.normalize(rgb_gray.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
        normalized_image[normalized_image == 0] = 0.025
        if cube.shape[-1] == 9:
            text_cube_data = np.dstack([normalized_image] * 9)
        else:
            text_cube_data = np.dstack([normalized_image] * 3)
        text_cube = cube * text_cube_data

        return text_cube



    def plot_multispectral_rgb_and_texture(self):
        """
        Generate and save comparison plots of GT cube, textured prediction, and segmentation layers.
        Inputs:
        - None explicitly. Uses configured input/output folders for:
          - RGB images
          - Ground truth cubes
          - Cluster images
          - Predicted RGB masks
        Process:
        - Iterates through predicted cubes
        - Loads all matching GT and RGB/cluster data
        - Applies texture to GT cube and material RGB mask
        - Saves textured cubes and RGB masks
        - Plots: GT, textured prediction, material mask, cluster image, original RGB
        - Saves visualization in a 'spectral_data_mask' folder
        Output:
        - Textured cube and mask files saved to disk
        - A plot image per cube saved as PNG
        """

        matplotlib.use('Agg')
        # Load all required paths from config
        input_cubes_folder = self.config['functions']['plot_multispectral_rgb']['input_cubes_path']
        input_gt_cubes_folder = self.config['functions']['plot_multispectral_rgb']['gt_cubes_path']
        rgb_images_folder_path = self.config['functions']['plot_multispectral_rgb']['rgb_folder_path']
        clu_images_folder_path = self.config['functions']['plot_multispectral_rgb']['clu_folder_path']
        materials_rgb_images_folder_path = self.config['functions']['plot_multispectral_rgb']['material_rgb_folder_path']
        # Extract implementation type (e.g., 'raffle' or 'max_score')
        impl_type = input_cubes_folder.split('\\')[-2]
        for cube_name in os.listdir(input_cubes_folder):
            full_cube_path = os.path.join(input_cubes_folder, cube_name)
            full_gt_cube_path = os.path.join(input_gt_cubes_folder, cube_name.replace('_cube', ''))
            cube = np.load(full_cube_path)
            cube_gt = np.load(full_gt_cube_path)
            # Select spectral indices for GT cube
            cube_gt_wl_type = cube[:,:, [147, 210, 315, 354, 390, 433, 515, 1264, 1852]]
            # Load cluster image and RGB-based segmentation
            clu_image = cv2.imread(os.path.join(clu_images_folder_path, cube_name.replace('_cube', '').replace('.npy', '.png')), cv2.IMREAD_UNCHANGED)
            material_rgb_image = cv2.cvtColor(cv2.imread(os.path.join(materials_rgb_images_folder_path, cube_name.replace('_cube', '_rgb_mask').replace('.npy', '.tiff')), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
            rgb_image = cv2.imread(os.path.join(rgb_images_folder_path, cube_name.replace('_cube', '').replace('.npy', '.png')), cv2.IMREAD_UNCHANGED)
            # Apply texture to predicted cube and material RGB
            text_cube = self.apply_texture(rgb_image, cube_gt_wl_type)
            text_material_rgb = self.apply_texture(rgb_image, material_rgb_image).astype('uint8')

            if not os.path.exists(materials_rgb_images_folder_path.replace('\\rgb_masks', '\\rgb_masks_text')):
                os.mkdir(materials_rgb_images_folder_path.replace('\\rgb_masks', '\\rgb_masks_text'))

            if not os.path.exists(input_cubes_folder.replace(f'{impl_type}\\cubes', f'{impl_type}\\cubes_text')):
                os.mkdir(input_cubes_folder.replace(f'{impl_type}\\cubes', f'{impl_type}\\cubes_text'))

            # Save processed data
            np.save(full_cube_path.replace('cubes\\', 'cubes_text\\'), text_cube)
            cv2.imwrite(os.path.join(materials_rgb_images_folder_path.replace('\\rgb_masks', '\\rgb_masks_text'), cube_name.replace('_cube', '_rgb_mask').replace('.npy', '.tiff')), text_material_rgb)
            # Create and display comparison figure
            fig, ax = plt.subplots(1, 5, figsize=(14, 5))

            ax[0].imshow((spectral.get_rgb(cube_gt)*255).astype('uint8'))
            ax[0].axis('off')  # Hide axis
            ax[0].set_title('GT cube')

            ax[1].imshow((spectral.get_rgb(text_cube)*255).astype('uint8'))
            ax[1].axis('off')
            ax[1].set_title('textured predicted cube')

            ax[2].imshow(material_rgb_image)
            ax[2].axis('off')
            ax[2].set_title('textured material rgb image')

            ax[3].imshow(clu_image)
            ax[3].axis('off')
            ax[3].set_title('clu image')

            ax[4].imshow(rgb_image)
            ax[4].axis('off')
            ax[4].set_title('rgb image')

            plt.tight_layout()
            plt.show()
            # Save and clear the plot
            if not os.path.exists(input_cubes_folder.replace(f'{impl_type}\\cubes', f'{impl_type}\\masks\\spectral_data_mask')):
                os.mkdir(input_cubes_folder.replace(f'{impl_type}\\cubes', f'{impl_type}\\masks\\spectral_data_mask'))
            plt.savefig(os.path.join(input_cubes_folder.replace(f'{impl_type}\\cubes', f'{impl_type}\\masks\\spectral_data_mask'), cube_name.replace('.npy', '.png')))
            plt.clf()

    def compare_metrics(self):
        """
        Compute and plot average MSE, SSIM, and SAM metrics between predicted and ground-truth hyperspectral cubes.
        Inputs:
        - Uses configuration to locate:
          - Ground truth folder path
          - Predicted cube folder path
        Process:
        - Iterate through predicted files and load their GT counterparts
        - Perform band selection if needed (based on source type)
        - Compute MSE, SSIM, and SAM for each pair
        - Aggregate metrics and plot their trends across samples
        Output:
        - Printed average scores
        - Visualization of metrics saved under a 'metrics' subfolder
        """
        matplotlib.use('Agg')
        gt_folder = self.config['functions']['compare_metrics']['gt_folder']
        predicted_folder = self.config['functions']['compare_metrics']['predicted_folder']
        mse_scores, ssim_scores, sam_scores = [], [], []

        pred_files = sorted(os.listdir(predicted_folder))

        for pred_file in tqdm(pred_files):
            if 'ref_rgb_cubes' in predicted_folder:
                gt_path = os.path.join(gt_folder, pred_file)
                pred_path = os.path.join(predicted_folder, pred_file)
            else:
                gt_path = os.path.join(gt_folder, pred_file.replace('_cube', ''))
                pred_path = os.path.join(predicted_folder, pred_file)

            gt_img = np.load(gt_path)
            pred_img = np.load(pred_path).astype('float64')
            # Apply WL band selection if applicable
            if 'ref_rgb_cubes' in predicted_folder:
                if 'Arad_python' in predicted_folder:
                    pred_img = pred_img[:, :, [9, 16, 26, 30]]/255
                    # Due to representation of data outputs on Arad paper in 0-255 scale 8 bit
                    gt_img =  gt_img[:,:,:4]
                else:
                    pred_img = pred_img[:, :, [7, 14, 24, 29]]
                    gt_img =  gt_img[:,:,:4]
            else:
                if pred_img.shape[-1]!=9:
                    pred_img = pred_img[:, :, [147, 210, 315, 354, 390, 433, 515, 1264, 1852]]

            # Ensure images have the same shape
            if gt_img.shape != pred_img.shape:
                print(f"Skipping {pred_file} - Shape mismatch")
                continue

            # Compute metrics
            mse_scores.append(self.compute_mse(gt_img, pred_img))
            ssim_scores.append(self.compute_ssim(gt_img, pred_img))
            sam_scores.append(self.compute_sam(gt_img, pred_img))

        # Compute overall averages
        avg_mse = np.mean(mse_scores) if mse_scores else None
        avg_ssim = np.mean(ssim_scores) if ssim_scores else None
        avg_sam = np.mean(sam_scores) if sam_scores else None

        print(f"Average MSE: {avg_mse:.4f} (Lower is better)")
        print(f"Average SSIM: {avg_ssim:.4f} (Higher is better, max 1.0)")
        print(f"Average SAM: {avg_sam:.4f} (Lower is better, angle in radians)")

        # Plot metrics
        plt.figure(figsize=(12, 4))
        # MSE Plot
        plt.subplot(1, 3, 1)
        plt.plot(mse_scores, marker='o')
        plt.title(f"MSE (Mean: {avg_mse:.4f})")
        plt.xlabel("Index")
        plt.ylabel("MSE")
        plt.grid()

        # SSIM Plot
        plt.subplot(1, 3, 2)
        plt.plot(ssim_scores, marker='o')
        plt.title(f"SSIM (Mean: {avg_ssim:.4f})")
        plt.xlabel("Index")
        plt.ylabel("SSIM")
        plt.grid()

        # SAM Plot
        plt.subplot(1, 3, 3)
        plt.plot(sam_scores, marker='o')
        plt.title(f"SAM (Mean: {avg_sam:.4f})")
        plt.xlabel("Index")
        plt.ylabel("SAM")
        plt.grid()

        plt.tight_layout()
        plt.show()

        # Save plots to disk
        if 'ref_rgb_cubes' in predicted_folder:
            if not os.path.exists(os.path.join(predicted_folder, 'metrics')):
                os.mkdir(os.path.join(predicted_folder, 'metrics'))

            plt.savefig(
                os.path.join(predicted_folder, 'metrics', "metrics_data_full_data_cubes.png"))
        else:
            if not os.path.exists(predicted_folder.replace('cubes_text', 'metrics')):
                os.mkdir(predicted_folder.replace('cubes_text', 'metrics'))

            plt.savefig(os.path.join(predicted_folder.replace('cubes_text', 'metrics'), "metrics_data_full_data_cubes.png"))





    def compare_materials_signatures(self):
        """
        Analyze and visualize the spectral similarity of materials based on SSIM-thresholded pixels.
        Inputs:
        - Configuration file specifying:
          - Ground truth cube folder
          - Predicted cube folder
          - Clustering image folder
          - Data dictionary folder with material info
        Process:
        - For each predicted cube, compute pixel-wise SSIM against GT cube
        - Identify pixels with SSIM > 0.7
        - Retrieve material info based on cluster ID
        - Sample up to 25 signature comparisons
        - Align predicted signatures to GT for better visual comparison
        - Plot GT, predicted, and aligned signatures with metadata
        Output:
        - Signature plots saved under the `sig_metrics` folder
        """

        matplotlib.use('Agg')
        gt_folder = self.config['functions']['compare_materials_signatures']['gt_folder']
        predicted_folder = self.config['functions']['compare_materials_signatures']['predicted_folder']
        data_dict_folder = self.config['functions']['compare_materials_signatures']['data_dict_folder_path']
        clu_images_folder = self.config['functions']['compare_materials_signatures']['clu_folder_path']
        wl_list = [497, 560, 665, 704, 740, 783, 865, 1614, 2202]


        pred_files = sorted(os.listdir(predicted_folder))  # Ensure matching order

        for pred_file in tqdm(pred_files):
            gt_path = os.path.join(gt_folder, pred_file.replace('_cube', ''))
            pred_path = os.path.join(predicted_folder, pred_file)
            clu_path = os.path.join(clu_images_folder, pred_file.replace('_cube', '').replace('.npy', '.png'))
            material_data_dict_path = os.path.join(data_dict_folder, pred_file.replace('.npy', '.pickle'))

            # Load images (Assuming .npy format; modify if needed)
            gt_img = np.load(gt_path)  # Shape (224, 224, 9)
            pred_img = np.load(pred_path).astype('float64')  # Shape (224, 224, 9)
            clu_image = cv2.imread(clu_path, cv2.IMREAD_UNCHANGED)
            material_data_dict = self.pickle_loader(material_data_dict_path)

            # Ensure images have the same shape
            if gt_img.shape != pred_img.shape:
                print(f"Skipping {pred_file} - Shape mismatch")
                continue

            # Get all high-SSIM material signature pairs
            full_materials_data_list = self.compute_ssim_for_material_sig(gt_img, pred_img, clu_image, material_data_dict)

            # Sample up to 25 signatures for visualization
            if len(full_materials_data_list) > 25:
                random_materials_data_samples = random.sample(full_materials_data_list, 25)
            else:
                random_materials_data_samples = full_materials_data_list
            
            # Align predicted signature to GT
            for indx, material_data_tup in enumerate(random_materials_data_samples):
                mean1, mean2 = np.mean(material_data_tup[0]), np.mean(material_data_tup[1])
                range1, range2 = np.max(material_data_tup[0]) - np.min(material_data_tup[0]), np.max(material_data_tup[1]) - np.min(material_data_tup[1])

                signature2_aligned = (material_data_tup[1] - mean2) * (range1 / range2) + mean1

                plt.figure(figsize=(12, 6))
                # Plot both signatures
                plt.plot(wl_list, material_data_tup[0], label="GT sig", marker='o')
                plt.plot(wl_list, material_data_tup[1],  label="Pred sig", marker='s')
                plt.plot(wl_list, signature2_aligned,  label="Pred sig aligned", marker='s')

                # Set title and labels
                plt.title(f"cube_name: {pred_file.replace('_cube', '')} material_Name&Class: {material_data_tup[2][-3]}")
                plt.xlabel("WL")
                plt.ylabel("Ref")

                # Show legend
                plt.legend()
                if not os.path.exists(predicted_folder.replace('cubes_text', 'sig_metrics')):
                    os.mkdir(predicted_folder.replace('cubes_text', 'sig_metrics'))
                plt.savefig(pred_path.replace('cubes_text', 'sig_metrics').replace('.npy', f'_{str(indx)}.png'))

                plt.clf()
                plt.close()




    def compare_metrics_rgb(self):
        """
        Compute and visualize SSIM, MSE, and RMSE between GT and predicted RGB images.
        Inputs:
        - Configuration file specifying:
          - Ground truth RGB folder
          - Predicted material RGB folder
        Process:
        - Iterate through predicted RGB files
        - Load and match GT images
        - Convert predicted BGR images to RGB
        - Compute SSIM, MSE, and RMSE per sample
        - Aggregate statistics and plot trends
        Output:
        - Printed averages
        - Line plots saved under 'metrics_rgb' directory
        """
        matplotlib.use('Agg')
        gt_folder = self.config['functions']['compare_metrics_rgb']['gt_folder']
        predicted_folder = self.config['functions']['compare_metrics_rgb']['predicted_folder']
        mse_scores, ssim_scores, rmse_scores = [], [], []

        pred_files = sorted(os.listdir(predicted_folder))

        for pred_file in tqdm(pred_files):
            if 'SRR_test_session' in predicted_folder or 'Arad_python' in predicted_folder:
                gt_path = os.path.join(gt_folder, pred_file)
                pred_path = os.path.join(predicted_folder, pred_file)
            else:
                gt_path = os.path.join(gt_folder, pred_file.replace('_rgb_mask.tiff', '.png'))
                pred_path = os.path.join(predicted_folder, pred_file)

            gt_img = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)  # Shape (224, 224, 3)
            pred_img = cv2.cvtColor(cv2.imread(pred_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)    # Shape (224, 224, 3)

            # Ensure images have the same shape
            if gt_img.shape != pred_img.shape:
                print(f"Skipping {pred_file} - Shape mismatch")
                continue

            # Compute metrics
            mse_scores.append(self.compute_mse_rgb(gt_img, pred_img))
            ssim_scores.append(self.compute_ssim(gt_img, pred_img))
            rmse_scores.append(self.compute_rmse(gt_img, pred_img))

        # Compute overall averages
        avg_mse = np.mean(mse_scores) if mse_scores else None
        avg_ssim = np.mean(ssim_scores) if ssim_scores else None
        avg_rmse = np.mean(rmse_scores) if rmse_scores else None

        print(f"Average MSE: {avg_mse:.4f} (Lower is better)")
        print(f"Average SSIM: {avg_ssim:.4f} (Higher is better, max 1.0)")
        print(f"Average RMSE: {avg_rmse:.4f} (Lower is better, closer to 0)")


        # Plot metrics
        plt.figure(figsize=(12, 4))

        # MSE Plot
        plt.subplot(1, 3, 1)
        plt.plot(mse_scores, marker='o')
        plt.title(f"MSE (Mean: {avg_mse:.4f})")
        plt.xlabel("Index")
        plt.ylabel("MSE")
        plt.grid()

        # SSIM Plot
        plt.subplot(1, 3, 2)
        plt.plot(ssim_scores, marker='o')
        plt.title(f"SSIM (Mean: {avg_ssim:.4f})")
        plt.xlabel("Index")
        plt.ylabel("SSIM")
        plt.grid()

        # SAM Plot
        plt.subplot(1, 3, 3)
        plt.plot(rmse_scores, marker='o')
        plt.title(f"RMSE (Mean: {avg_rmse:.4f})")
        plt.xlabel("Index")
        plt.ylabel("RMSE")
        plt.grid()

        plt.tight_layout()
        plt.show()

        if 'SRR_test_session' in predicted_folder or 'Arad_python' in predicted_folder :
            if not os.path.exists(os.path.join(predicted_folder, 'metrics_rgb')):
                os.mkdir(os.path.join(predicted_folder, 'metrics_rgb'))

            plt.savefig(os.path.join(os.path.join(predicted_folder, 'metrics_rgb'), "metrics_data_full_data_cubes.png"))

        else:
            if not os.path.exists(predicted_folder.replace('masks\\rgb_masks', 'metrics_rgb')):
                os.mkdir(predicted_folder.replace('masks\\rgb_masks', 'metrics_rgb'))

            plt.savefig(os.path.join(predicted_folder.replace('masks\\rgb_masks', 'metrics_rgb'),
                                     "metrics_data_full_data_cubes.png"))




import copy
import datetime
import multiprocessing as mp
from sklearn.metrics import mean_squared_error
import os.path
import ast
import json
from collections import defaultdict
import pickle
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import cv2
import yaml
from yaml.loader import SafeLoader
from matplotlib import pyplot as plt
from front.materials_heuristics_keys_sycronizore import keys_sycronizore
import matplotlib
import numpy as np
import os
import pandas as pd
import time
from sklearn.cluster import KMeans
from back.NN_dev import SegmentationModel

class Materials_prossesor():
    """
    A class to handle all processing related to material heuristics, clustering,
    RGB segmentation, YAML/JSON/CSV handling, and VGG-based classification.

    Inputs:
    - None directly; configuration is loaded from 'config_MATERIALS.yaml'

    Process:
    - Loads configuration paths and provides utilities for dumping/loading data
      and processing images, segments, and RGBs using multiprocessing and clustering.

    Output:
    - Object instance used for method calls to perform processing.
    """

    def __init__(self):
        with open('config_MATERIALS.yaml') as f:
            data = yaml.load(f, Loader=SafeLoader)
        f.close()
        self.config = data

        # Store important paths from config
        self.path = {'main_pipeline_path': self.config['functions']['general_params']['main_pipeline_path'],
            'results': self.config['functions']['general_params']['results_path'],
                  'test': self.config['functions']['general_params']['test_path'],
                     'input': self.config['functions']['general_params']['input_path']}

    """ - -----------------------------  Static methods - ----------------------------------------- """

    @staticmethod
    def folder_checker(path):
        """
        Ensures a folder exists; if not, creates it.
        Input:
        - path: String, directory path to check/create.
        Process:
        - If the directory doesn't exist, creates it.
        - Otherwise, logs that the directory already exists.
        Output:
        - None (prints status to console)
        """
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Folders created for the following path : {path}" )
        else:
            print(f" Folder path are already exist for : {path}")

    @staticmethod
    def yaml_dumper(data, path):
        """
        Dumps a Python dictionary into a YAML file.
        Inputs:
        - data: dict, the data to write.
        - path: str, path to the YAML file to create.
        Process:
        - Writes the data using PyYAML with sorted keys disabled.
        Output:
        - None
        """
        with open(path, 'w') as outfile:
            yaml.dump(data, outfile, sort_keys=False, default_flow_style=None)

    @staticmethod
    def json_dumper(path, data):
        """
        Saves a dictionary into a JSON file.
        Inputs:
        - path: str, path to write JSON to.
        - data: dict, the content to serialize.
        Process:
        - Opens a file and writes JSON content.
        Output:
        - None
        """
        with open(path, 'w') as f:
            json.dump(data, f)
        f.close()

    @staticmethod
    def pickle_dumper(path, data):
        """
        Serializes Python objects to a file using pickle.
        Inputs:
        - path: str, destination path
        - data: object to pickle
        Process:
        - Opens the file and writes using pickle.dump()
        Output:
        - None
        """
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        f.close()

    @staticmethod
    def pickle_loader(path):
        """
        Loads a pickled Python object from file.
        Input:
        - path: str, path to .pkl file
        Process:
        - Opens the file, reads content with pickle.load()
        Output:
        - Loaded Python object
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        f.close()
        return data

    @staticmethod
    def json_loader(path):
        """
        Loads a dictionary from a JSON file.
        Input:
        - path: str, path to the JSON file.
        Process:
        - Opens and reads the JSON file using json.load().
        Output:
        - data: dict, the parsed JSON content.
        """
        with open(path, 'r') as f:
            data = json.load(f)
        f.close()
        return data


    @staticmethod
    def yaml_loader(path):
        """
        Loads a dictionary from a YAML file.
        Input:
        - path: str, path to the YAML file.
        Process:
        - Opens the YAML file and parses it with SafeLoader.
        Output:
        - data: dict, parsed YAML content.
        """
        with open(path) as f:
            data = yaml.load(f, Loader=SafeLoader)
        f.close()
        return data

    @staticmethod
    def parallel_get_all_rgb(rgb_image_path):
        """
        Extracts all unique RGB colors per segment in a labeled image.
        Inputs:
        - rgb_image_path: str, path to the RGB image file.
        Process:
        - Loads the RGB image and corresponding segmentation mask.
        - For each segment, identifies all unique RGB values of pixels within the mask.
        Output:
        - segment_colors: dict, key is segment ID, value is unique RGB colors in that segment.
        """
        mask_image_path = rgb_image_path.replace('images', 'masks')

        rgb_image = cv2.imread(rgb_image_path, cv2.IMREAD_UNCHANGED)
        mask_image = cv2.imread(mask_image_path, cv2.IMREAD_UNCHANGED)
        # Identify unique segment labels
        unique_segments = np.unique(mask_image)
        segment_colors = {}

        for segment in unique_segments:
            # Get pixel indices belonging to this segment
            mask_indices = np.where(mask_image == segment)
            # Extract RGB values and find unique ones
            segment_rgb_values = rgb_image[mask_indices]
            segment_colors[segment] = np.unique(segment_rgb_values, axis=0)

        return segment_colors


    @staticmethod
    def parallel_clu_rgb(frame_tup):
        """
        Computes mean RGB values for clustered RGB segments and groups unique RGBs per segment.
        Input:
        - frame_tup: tuple (frame_num, frame_name)
        Process:
        - Reads clustered RGB, original RGB, and segmentation masks for the frame.
        - For each segment:
          - Extracts unique RGBs from clustered image.
          - Calculates the mean original RGB value for each cluster RGB.
        Output:
        - mean_dict: dict mapping cluster RGB to (mean RGB, count)
        - seg_unique_clu_dict: dict mapping segment name to its unique cluster RGBs.
        """
        frame_num = frame_tup[0]
        frame = frame_tup[1]
        mean_dict = dict()
        seg_unique_clu_dict = dict()
        # Load clustered RGB image and flatten it
        full_clu_path = os.path.join(main_clu_path_global, frame)
        frame_clu_rgb = cv2.imread(full_clu_path, cv2.IMREAD_UNCHANGED)

        flat_clu_rgb_img = frame_clu_rgb.reshape(-1, frame_clu_rgb.shape[2])

        # Load original RGB and segmentation images
        full_rgb_path = os.path.join(main_rgb_path_global, rgb_frame_list_global[frame_num])
        full_seg_path = os.path.join(main_seg_path_global, seg_rgb_list_global[frame_num])
        frame_seg = cv2.imread(full_seg_path, cv2.IMREAD_UNCHANGED)
        flat_seg_img = frame_seg.reshape(-1)
        frame_rgb = cv2.imread(full_rgb_path, cv2.IMREAD_UNCHANGED)
        flat_rgb_img = frame_rgb.reshape(-1, frame_rgb.shape[2])
        # For each segment, compute unique clustered RGBs
        unique_seg_list = np.unique(flat_seg_img, axis=0)
        unique_clu_rgb_list = []
        for seg in unique_seg_list:
            indices = np.where(flat_seg_img == seg)
            rgb_list = flat_clu_rgb_img[indices]
            unique_rgb_list = np.unique(rgb_list, axis=0)
            segment_name = objects_seg_rgb_data_global[
                Materials_prossesor.array_to_string_defulter_rgb_v(Materials_prossesor, str(seg))]
            seg_unique_clu_dict[segment_name] = unique_rgb_list
            unique_clu_rgb_list.extend(unique_rgb_list)

        # Compute average RGB for each unique clustered RGB
        full_clu_unique_rgb = np.unique(unique_clu_rgb_list, axis=0)
        for clu_rgb in full_clu_unique_rgb:
            str_clu_rgb = str(clu_rgb)
            indices = np.where(np.all(flat_clu_rgb_img == clu_rgb, axis=1))
            rgb_list = flat_rgb_img[indices]
            if len(rgb_list) == 0:
                continue
            mean_rgb = np.mean(np.array(rgb_list), axis=0)
            mean_dict[str_clu_rgb] = (mean_rgb, len(rgb_list))

        return mean_dict, seg_unique_clu_dict



    @staticmethod
    def bin_image(tup):
        """
        Applies k-means clustering to each segment of the image and assigns pixels with cluster center colors.
        Inputs:
        - tup: a tuple of three elements:
            0: str - path to the RGB image.
            1: str - path to save the binned image.
            2: dict - segment-to-unique-RGB mapping.
        Process:
        - Loads the RGB image and corresponding mask.
        - For each segment:
            - Runs k-means clustering on unique RGBs (or fewer if less than 30).
            - Applies the fitted model to segment pixels.
            - Replaces pixels with their cluster center.
        Output:
        - Writes the binned image to the specified location.
        """
        img_loc = tup[0]
        mask_loc = img_loc.replace('images', 'masks')
        save_loc = tup[1]
        segments_dict = tup[2]
        img = cv2.imread(img_loc, cv2.IMREAD_UNCHANGED)
        mask_img = cv2.imread(mask_loc, cv2.IMREAD_UNCHANGED)
        binned_img = np.zeros_like(img)

        k=30 # defined from clustering check on check_for_k_cluster_number from DataGenerator_dev
        kmeans_model = KMeans(n_clusters=k)


        for segment_label, unique_rgb_values in segments_dict.items():
            # Use fewer clusters if not enough data
            if len(unique_rgb_values)<30:
                k = len(unique_rgb_values)
                kmeans_model = KMeans(n_clusters=k)

            # Apply KMeans model to classify segment pixels
            K_means_labels_model = kmeans_model.fit(unique_rgb_values)

            # Find pixels belonging to the current segment
            segment_pixels = np.where(mask_img == segment_label)
            if len(segment_pixels[0]) == 0:
                continue  # Skip empty segments

            segment_rgb = img[segment_pixels]

            resulting_img_labels = K_means_labels_model.predict(segment_rgb)
            cluster_centers = K_means_labels_model.cluster_centers_


            # Replace pixel colors with their cluster center
            new_rgb_values = np.array([cluster_centers[label] for label in resulting_img_labels], dtype=np.uint8)
            binned_img[segment_pixels] = new_rgb_values

        cv2.imwrite(save_loc, binned_img)



    @staticmethod
    def array_to_string_defulter_rgb_v_parallel(arr_str):
        """
        Cleans and standardizes an RGB string into a list-format string.
        Input:
        - arr_str: str, a string representing RGB values like '[123, 45, 67]' or '123 45 67'.
        Process:
        - Strips brackets and splits by commas or spaces.
        - Converts digits to integers.
        Output:
        - str, formatted list-like string, e.g., '[123, 45, 67]'.
        """
        if ',' in arr_str:
            base_arr_str = arr_str.replace("[", "").replace("]", "").replace(" ", "")
            base_arr_list = base_arr_str.split(",")
        else:
            base_arr_str = arr_str.replace("[", "").replace("]", "")
            base_arr_list = base_arr_str.split(" ")

        final_arr_list = [int(x) for x in base_arr_list if x.isdigit()]
        return str(final_arr_list)


    @staticmethod
    def compute_object_scores_mat(args_list):
        """
        Computes a score matrix for a given object (segment class), evaluating how well
        candidate materials match the RGB values clustered from scene frames.

        Inputs:
        - args_list: tuple (obj, clu_rgb)
            obj: str - object/segment name.
            clu_rgb: list of RGB vectors (as lists) - base RGBs detected for that object.

        Process:
        - Initializes an empty score DataFrame indexed by material labels.
        - If no base RGBs exist for the object, all values are set to 0.
        - Otherwise, for each material with a valid RGB:
            - Computes MSE distance to each base RGB.
            - Normalizes these distances to [0, 1], assigns (1 - norm_dist) as score.
        - Applies heuristics using a hot vector matrix to zero out unsupported mappings.

        Output:
        - Tuple: (object_name, DataFrame of scores for each material and RGB cluster)
        """

        obj = args_list[0]
        clu_rgb = args_list[1]
        if len(clu_rgb) == 0:
            # No base RGBs found for this object — return zeroed 'no_base' column
            obj_scores_df = pd.DataFrame(columns=['no_base'], index=materials_labels_rgb_swir_dict_global_mat.keys())
            obj_scores_df.loc[:, 'no_base'] = 0
        else:
            # Initialize score matrix with cluster RGBs as columns
            obj_scores_df = pd.DataFrame(columns=[str(rgb) for rgb in clu_rgb],
                                         index=materials_labels_rgb_swir_dict_global_mat.keys())

            for column in hot_vector_df_global.columns:
                if column == 'Material_Label':
                    continue
                col_first_label = column.split(';')[0]
                # Skip if segment not relevant for this scene
                if col_first_label != obj:
                    continue
                if len(obj_scores_df) != len(hot_vector_df_global[column].values):
                    # Filter rows if misaligned with materials
                    common_indices = obj_scores_df.index.intersection(hot_vector_df_global.index)
                    hot_vector_df_filtered = hot_vector_df_global.loc[common_indices]
                    col_values = hot_vector_df_filtered[column].values
                    mask = col_values == 0
                    obj_scores_df[mask] = 0
                else:
                    # Assign zero to disallowed materials
                    col_values = hot_vector_df_global[column].values
                    mask = col_values == 0
                    obj_scores_df[mask] = 0

            # Extract reference RGBs for allowed materials
            mat_rgb_dict = {}
            for index in obj_scores_df.index:
                # Skip materials not allowed for this object
                if (obj_scores_df.loc[index] == 0).all():
                    continue
                else:
                    mat_rgb = materials_labels_rgb_swir_dict_global_mat[index][0]
                    mat_rgb_dict[index] = np.array(mat_rgb)

            # Compute score per RGB cluster per material
            if len(mat_rgb_dict) != 0:
                for column in obj_scores_df.columns:
                    new_column = Materials_prossesor.array_to_string_defulter_rgb_v_parallel(
                        column)
                    if new_column not in unique_clu_rgb_and_mean_rgb_dict_global_mat.keys():
                        continue

                    rgb = unique_clu_rgb_and_mean_rgb_dict_global_mat[new_column]
                    # Compute MSE for all materials to this base RGB
                    mse_list = [mean_squared_error(rgb, mat_rgb) for mat_rgb in mat_rgb_dict.values()]
                    # Apply only to rows where material is valid (non-zero)
                    obj_scores_df.loc[obj_scores_df[column] != 0, column] = mse_list
                    # Normalize distances to [0, 1] and convert to scores
                    max_dist = obj_scores_df[column].max()
                    obj_scores_df[column] = obj_scores_df[column] / (
                                max_dist + 1)
                    obj_scores_df.loc[obj_scores_df[column] != 0, column] = 1 - obj_scores_df.loc[
                        obj_scores_df[column] != 0, column]
            else:
                # No valid material candidates, ski
                pass
        # Replace all NaNs with 0 for safe export
        obj_scores_df.fillna(0, inplace=True)

        return (obj, obj_scores_df)



    """ - -----------------------------  heuristics vector - ----------------------------------------- """

    def heuristic_object_material_vector(self):
        """
        Builds a heuristic binary matrix mapping each material label to object/segment labels based on name matches.
        Inputs:
        - Reads object-segment mapping from a YAML file.
        - Loads the materials dataframe from CSV.
        Process:
        - Generates column names (segments) from the object keys.
        - Initializes a binary matrix with 1 where the material class name is part of the segment key.
        - Handles duplicate material labels by renaming them with incremental suffixes.
        - Drops rows with missing data and saves the matrix as CSV.
        Output:
        - Saves the final binary matrix CSV under `results/heuristic_object_material_vector/`
        """
        objects_list_path = self.path['input']
        input_folder = self.config['functions']['heuristic_object_material_vector']['input_folder']
        input_file = self.config['functions']['heuristic_object_material_vector']['input_file']
        output_folder = self.config['functions']['heuristic_object_material_vector']['output_folder']
        objects_list_full_path = os.path.join(objects_list_path, input_folder, input_file)
        output_path = os.path.join(self.path['results'], output_folder)
        self.folder_checker(output_path)
        output_file_name = self.config['functions']['heuristic_object_material_vector']['output_file']
        output_full_path = os.path.join(output_path, output_file_name)
        # Load segment label mapping and materials dataframe
        data = self.yaml_loader(objects_list_full_path)['labels_numbers_dict']
        materials_df = pd.read_csv(os.path.join(self.path['results'], 'generate_materials_df', 'materials_df.csv'))
        materials_df_copy = copy.deepcopy(materials_df)
        # Create hot vector DataFrame with objects as columns, materials as rows
        new_keys_list = keys_sycronizore(data)
        hot_vec_df = pd.DataFrame(columns=new_keys_list, index=materials_df_copy['Material_Label'])
        # Populate binary entries based on string matching
        for index in tqdm(materials_df_copy.index):
            for key in new_keys_list:
                if materials_df_copy['Material_Class'].loc[index].lower() in key.lower():
                    try:
                        hot_vec_df.at[materials_df_copy.loc[index]['Material_Label'] ,key] = 1
                    except:
                        continue
                else:
                    try:
                        hot_vec_df.at[materials_df_copy.loc[index]['Material_Label'] ,key] = 0
                    except:
                        continue

        hot_vec_df.dropna(inplace=True)
        # Handle duplicate material labels by appending numeric suffixes
        material_label_list = hot_vec_df.index
        count_dict = dict()
        material_label_final_list = []
        for label in material_label_list:
            if label not in material_label_final_list:
                count_dict[label]=0
                material_label_final_list.append(label)
            else:
                count_dict[label] += 1
                new_label = label+'_'+str(count_dict[label])
                material_label_final_list.append(new_label)

        # Finalize dataframe and Save
        hot_vec_df['Material_Label'] = material_label_final_list
        hot_vec_df.set_index('Material_Label', inplace=True)
        hot_vec_df.to_csv(output_full_path)


    """ - -----------------------------  scores metrics process - ----------------------------------------- """


    def worker_init_rgb_scores_dict(self, global_args_list):
        """
        Initializes global variables inside each worker process for parallel computation.
        Inputs:
        - global_args_list: a list containing:
            [0] materials_labels_rgb_swir_dict (dict)
            [1] unique_clu_rgb_and_mean_rgb_dict (dict)
            [2] hot_vector_df (DataFrame)
        This ensures each subprocess can access shared resources for computing RGB scores.
        """
        global materials_labels_rgb_swir_dict_global_mat
        global unique_clu_rgb_and_mean_rgb_dict_global_mat
        global hot_vector_df_global

        materials_labels_rgb_swir_dict_global_mat = global_args_list[0]
        unique_clu_rgb_and_mean_rgb_dict_global_mat = global_args_list[1]
        hot_vector_df_global = global_args_list[2]


    def rgb_scores_objects_dict(self, hot_vector_df, unique_clu_rgb_per_object_dict, materials_labels_rgb_swir_dict,
                                unique_clu_rgb_and_mean_rgb_dict, objects_seg_rgb_data):
        """
        Computes score matrices for each object by evaluating how well RGB cluster values match material RGBs.

        Inputs:
        - hot_vector_df: DataFrame of 0/1 indicating valid material-segment relationships.
        - unique_clu_rgb_per_object_dict: {segment_name: [clustered RGBs]}
        - materials_labels_rgb_swir_dict: {material_label: [RGB, SWIR, ...]}
        - unique_clu_rgb_and_mean_rgb_dict: {str(rgb): mean_rgb}
        - objects_seg_rgb_data: segment information per object
        Process:
        - Sets the index of hot_vector_df to 'Material_Label'
        - Prepares arguments and uses multiprocessing to compute score matrices in parallel.
        - For any object not present in the cluster RGB dict, a placeholder matrix is created.
        Outputs:
        - scores_mat_objects_dict: {object_name: DataFrame of scores (materials × RGB clusters)}
        """

        start = time.time()
        hot_vector_df.set_index(hot_vector_df.columns[0],
                                inplace=True)
        global_args_list = [copy.deepcopy(materials_labels_rgb_swir_dict), copy.deepcopy(unique_clu_rgb_and_mean_rgb_dict)
            , hot_vector_df]
        # Define the number of worker processes to use
        num_workers = self.config['functions']['general_params']['parallel_workers']
        # Create a pool of worker processes
        pool = mp.Pool(num_workers, initializer=self.worker_init_rgb_scores_dict, initargs=(global_args_list,))
        # Define a list of arguments to pass to the compute_object_scores function (object, [clustered RGBs])
        args_list = [(obj, clu_rgb) for
                     obj, clu_rgb in unique_clu_rgb_per_object_dict.items()]

        print("start polling")
        # Run parallel computation
        results = list(pool.map(self.compute_object_scores_mat, [arg_tup for arg_tup in args_list]))

        # Close the pool of worker processes
        pool.close()
        print("done pooling")
        end = time.time()
        time_in_sec = end - start
        time_in_min = time_in_sec / 60
        time_in_hours = time_in_min / 60
        print(f"time for all frames --> {time_in_hours}")

        # Merge results into a dictionary
        scores_mat_objects_dict = {obj: obj_scores_df for obj, obj_scores_df in results}
        # For missing segments, assign zero-filled 'no_base' matrices
        for seg in objects_seg_rgb_data.keys():
            if seg not in scores_mat_objects_dict:
                obj_scores_df = pd.DataFrame(columns=['no_base'],
                                             index=materials_labels_rgb_swir_dict.keys())
                obj_scores_df.loc[:, 'no_base'] = 0
                scores_mat_objects_dict[seg] = obj_scores_df
            else:
                continue

        return scores_mat_objects_dict


    def array_to_string_defulter_rgb_v(self, arr_str):
        """
        Converts a string representation of an RGB array into a standardized string format.
        Input:
        - arr_str: A string that may represent an RGB array, e.g., "[120, 200, 45]" or "120 200 45"
        Process:
        - Removes brackets and spaces.
        - Splits the string by commas (if present) or spaces.
        - Converts values to integers and back into a string list format.
        Output:
        - Standardized string format: "[120, 200, 45]"
        """
        if ',' in arr_str:
            clu_arr_str = arr_str.replace("[", "").replace("]", "").replace(" ", "")
            clu_arr_list = clu_arr_str.split(",")
        else:
            clu_arr_str = arr_str.replace("[", "").replace("]", "")
            clu_arr_list = clu_arr_str.split(" ")

        final_arr_list = [int(x) for x in clu_arr_list if x.isdigit()]
        return str(final_arr_list)


    def order_string_keys(self, unique_clu_rgb_and_mean_rgb_dict):
        """
        Reformats the keys of a dictionary from raw RGB string format to a standardized format.
        Input:
        - unique_clu_rgb_and_mean_rgb_dict: {raw_rgb_str: mean_rgb_values}
        Process:
        - Converts each key using `array_to_string_defulter_rgb_v`.
        Output:
        - A new dictionary with reformatted RGB string keys, ready to be used as keys in dataframes or scoring matrices.
        """
        new_dict = {}
        for key, value in unique_clu_rgb_and_mean_rgb_dict.items():
            new_key = self.array_to_string_defulter_rgb_v(key)
            new_dict[new_key] = value

        return new_dict



    def cluster_rgb(self, main_rgb_path, output_folder, frames_list):
        """
        Clusters RGB values of a folder of images using KMeans and bins them accordingly.
        Inputs:
        - main_rgb_path: path to original RGB images.
        - output_folder: path where the clustered (binned) images will be saved.
        - frames_list: list of image filenames to process.
        Process:
        - Uses multiprocessing to extract all unique RGB values per segment across all images.
        - Merges the segment-wise RGBs into a single dictionary keyed by segment ID.
        - Uses KMeans to bin each segment’s RGB values into clusters.
        - Writes new images where each pixel is replaced by its cluster centroid color.
        Output:
        - Saves the clustered/binned images into the output folder.
        """

        save_folder = output_folder  # Location to save binned images
        t = time.time()
        print('start polling on clustered rgb')
        # Parallel getting all unique RGB
        pool = mp.Pool(self.config['functions']['general_params']['parallel_workers'])

        with pool as p:
            all_frames_seg_data_dict = p.map(self.parallel_get_all_rgb,
                                       [os.path.join(main_rgb_path, frame) for frame in frames_list])

        p.close()

        # Merge the list of RGB dicts into a single dict: segment_id → all RGBs seen for that segment
        merged_dict = defaultdict(list)
        for d in all_frames_seg_data_dict:
            for key, value in d.items():
                merged_dict[key].append(value)  # Append values to the corresponding key

        # Stack and deduplicate RGBs per segment
        final_segment_colors_dict =  {key: np.unique(np.vstack(val), axis=0) for key , val in dict(merged_dict).items()}
        # Use another pool to bin images in parallel using the shared segment RGB clusters
        pool = mp.Pool(self.config['functions']['general_params']['parallel_workers'])
        results = list(pool.map(self.bin_image, [(os.path.join(main_rgb_path, frame), os.path.join(save_folder, frame), final_segment_colors_dict) for frame in frames_list]))
        pool.close()
        elapsed = (time.time() - t) / 60
        print('Binning Images: ' + str(np.round(elapsed, 2)) + ' minutes')


    def worker_init_clu_to_mean_rgb(self, global_args_list):
        """
        Initializes global variables for multiprocessing workers that will be used
        in the `parallel_clu_rgb` function.
        Input:
        - global_args_list: A list containing:
            0: path to clustered RGB images
            1: list of RGB frame filenames
            2: path to raw RGB images
            3: path to segmentation mask images
            4: list of segmentation mask filenames
            5: object-segment label mapping dictionary
        Output:
        - Sets these variables globally so all processes in the multiprocessing pool
          have access to them without reloading them.
        """
        global main_clu_path_global
        global main_rgb_path_global
        global main_seg_path_global
        global seg_rgb_list_global
        global rgb_frame_list_global
        global objects_seg_rgb_data_global

        main_clu_path_global = global_args_list[0]
        rgb_frame_list_global = global_args_list[1]
        main_rgb_path_global = global_args_list[2]
        main_seg_path_global = global_args_list[3]
        seg_rgb_list_global = global_args_list[4]
        objects_seg_rgb_data_global = global_args_list[5]


    def unique_clu_rgb_to_mean_rgb_and_seg_to_unique_rgb(self, clu_frame_list, rgb_frame_list, seg_rgb_list,
                                                          main_rgb_path, main_clu_path, main_seg_path,
                                                          objects_seg_rgb_data):
        """
        Computes two important mappings:
        1. For every clustered RGB value (base RGB), compute the mean true RGB value across all its instances.
        2. For every segment label, list all unique clustered RGBs that appear in it.
        Inputs:
        - clu_frame_list: filenames of clustered RGB images.
        - rgb_frame_list: filenames of raw RGB images.
        - seg_rgb_list: filenames of segmentation mask images.
        - main_rgb_path: path to raw RGB images.
        - main_clu_path: path to clustered RGB images.
        - main_seg_path: path to segmentation masks.
        - objects_seg_rgb_data: mapping from segment names to RGB label values.
        Process:
        - Uses multiprocessing to extract per-frame data (cluster-to-mean RGB mapping and segment-to-unique-clu-RGB).
        - Aggregates the data across all frames:
            a. Averages the RGB values per cluster across images.
            b. Unifies and deduplicates clustered RGBs per segment.
        Output:
        - full_list_mean_clu_rgb_to_rgb_dict: dict of cluster RGB → mean true RGB value.
        - final_seg_unique_clu_rgb_dict: dict of segment name → list of unique clustered RGBs.
        """
        # Prepare global values to be used in parallel worker processes
        global_args_list = [main_clu_path, rgb_frame_list, main_rgb_path, main_seg_path, seg_rgb_list,
                            {self.array_to_string_defulter_rgb_v(str(value)): key for key, value in
                             objects_seg_rgb_data.items()}
                            ]
        # init dict for moving avarage for all frames.
        full_seg_unique_clu_rgb_dict = dict()
        final_seg_unique_clu_rgb_dict = dict()
        full_list_mean_clu_rgb_to_rgb_dict = dict()

        print('start pooling')
        start = time.time()

        pool = mp.Pool(self.config['functions']['general_params']['parallel_workers'],
                       initializer=self.worker_init_clu_to_mean_rgb, initargs=(global_args_list,))
        with pool as p:
            results = p.map(self.parallel_clu_rgb, [(frame_num, frame) for frame_num, frame in enumerate(clu_frame_list)])

        print('done pooling')
        p.close()

        # Combine results from all frames
        for result_tup_dict in tqdm(results):
            seg_dict = result_tup_dict[1] # Segment → unique clustered RGBs
            mean_rgb_dict = result_tup_dict[0] # Cluster RGB → (mean RGB, count)

            # Unifying seg dicts
            for key, value in seg_dict.items():
                if key in full_seg_unique_clu_rgb_dict.keys():
                    full_seg_unique_clu_rgb_dict[key] = np.append(full_seg_unique_clu_rgb_dict[key], value, axis=0)
                else:  # If new segment, start with current dict clu rgbs.
                    full_seg_unique_clu_rgb_dict[key] = value

            # Merge cluster RGB to mean RGBs across frames (using weighted average)
            for str_rgb, data_tup in mean_rgb_dict.items():
                if str_rgb in full_list_mean_clu_rgb_to_rgb_dict.keys():  # If already in dict, update average rgb and number of pixels
                    cur_mean_rgb = data_tup[0]
                    cur_num_pixels = data_tup[1]
                    mean_rgb = full_list_mean_clu_rgb_to_rgb_dict[str_rgb][0]
                    num_pixels = full_list_mean_clu_rgb_to_rgb_dict[str_rgb][1]
                    new_mean_rgb = (num_pixels * mean_rgb + cur_num_pixels * cur_mean_rgb) / (
                                num_pixels + cur_num_pixels)
                    new_num_pixels = num_pixels + cur_num_pixels
                    new_tup = (new_mean_rgb, new_num_pixels)
                    full_list_mean_clu_rgb_to_rgb_dict[
                        str_rgb] = new_tup
                else:
                    full_list_mean_clu_rgb_to_rgb_dict[str_rgb] = data_tup

        full_list_mean_clu_rgb_to_rgb_dict = {key: val[0].tolist() for key, val in
                                               full_list_mean_clu_rgb_to_rgb_dict.items()}
        # Deduplicate RGB lists for each segment
        for seg, val in full_seg_unique_clu_rgb_dict.items():
            final_seg_unique_clu_rgb_dict[seg] = np.unique(val, axis=0)

        end = time.time()
        time_in_sec = end - start
        time_in_min = time_in_sec / 60
        time_in_hours = time_in_min / 60
        print(f"time for all frames --> {time_in_hours}")
        print('converting results to one dict --> full_list_mean_clu_rgb_to_rgb_dict')

        return full_list_mean_clu_rgb_to_rgb_dict, final_seg_unique_clu_rgb_dict



    def materials_rgb_dict_creator(self, materials_df):
        """
        Builds a dictionary mapping material labels to their associated properties,
        including RGB values, SWIR reflectance, label info, and segment mask path.
        Input:
        - materials_df: pandas DataFrame, typically loaded from 'materials_df.csv',
          containing metadata about all materials including paths to YAML descriptors.
        Process:
        - For each material:
            - Loads the corresponding YAML file.
            - Extracts relevant fields: RGB, SWIR ("Raw_ref"), label type, and RGB segment mask.
            - Stores this information in a dictionary keyed by material label.
            - Handles duplicate labels by appending a counter (_1, _2, etc.) to make keys unique.
        Output:
        - materials_label_rgb_dict: dictionary of format:
            {
              'Material_Label_1': [RGB, Raw_ref, Material_Label+Type, RGB_seg_mask],
              'Material_Label_1_1': [RGB, Raw_ref, ...],  # for duplicates
              ...
            }
        """

        materials_label_rgb_dict = dict()
        count_dict = dict()
        for index in tqdm(materials_df.index):
            path = materials_df['Material_Path'][index]
            if path == None or path == '-':
                print(f"material path is None for {materials_df['Material_Label'][index]}")
                continue

            data = self.yaml_loader(path)
            material_label = materials_df['Material_Label'][index]

            try:
                material_rgb = data['RGB']
                spectro_ref = data['Raw_ref']
                material_type = data['Label_Class'].capitalize()
                materials_type_rgb_seg = data['RGB_seg_mask']

                if material_rgb is None:
                    print(f'material rgb is None for {path}')
                    continue
            except:
                print(f'general problem for {path}')
            # Add to dictionary; ensure uniqueness for repeated material labels
            if material_label not in materials_label_rgb_dict.keys():
                materials_label_rgb_dict[material_label] = [material_rgb, spectro_ref, ''.join(
                    [material_label.capitalize(), material_type]), materials_type_rgb_seg]
                count_dict[material_label] = 0
            # Duplicate label: append counter suffix
            else:
                count_dict[material_label] += 1
                materials_label_rgb_dict[material_label + '_' + str(count_dict[material_label])] = [material_rgb,
                                                                                                    spectro_ref,
                                                                                                    ''.join(
                                                                                                        [material_label,
                                                                                                         material_type
                                                                                                         ]),
                                                                                                    materials_type_rgb_seg]

        return materials_label_rgb_dict

    """ - -----------------------------  Main materials vectors build process - ----------------------------------------- """

    def heuristic_rgb_material_vector_v1(self):
        """
        Main pipeline for generating heuristic RGB-to-material match matrices.
        Inputs:
        - Configuration is read from `config_MATERIALS.yaml`
        - Uses a list of folder names, each representing a simulation or scene
        Process:
        For each scene (folder_name):
        1. Load paths and segmentation mapping.
        2. Load or create the materials RGB+SWIR dictionary.
        3. Optionally generate clustered RGB images (binning).
        4. Load or compute the hot-vector (heuristic eligibility matrix).
        5. Create mappings of unique clustered RGBs to average RGBs.
        6. Compute the score matrix per object, where scores represent how well an RGB matches a material.
        Output:
        - Saves score matrices to pickle files
        - Saves material-to-RGB dict if required
        """

        folder_names_list = self.config['functions']['heuristic_rgb_material_vector_v1']['folder_names_list']

        # Load mapping of segments classes
        json_name = self.config['functions']['heuristic_rgb_material_vector_v1'][
            'object_data_file_name']

        objects_seg_rgb_data = self.yaml_loader(os.path.join(self.path['input'], 'seg_mapping', json_name))['labels_numbers_dict']


        for folder_name in tqdm(folder_names_list):
            # Set up paths for RGB images, segmentation, and clustering output
            print(folder_name)
            main_rgb_path = os.path.join(self.path['results'], 'vgg19_clasiffier', folder_name,
                                                   'images')
            main_seg_path = os.path.join(self.path['results'], 'vgg19_clasiffier', folder_name, 'masks')
            materials_df = pd.read_csv(os.path.join(self.path['results'], 'generate_materials_df', 'materials_df.csv'))
            clustered_rgb_path = os.path.join(self.path['results'], 'heuristic_rgb_material_vector_v1', folder_name, 'clustered_rgb')
            self.folder_checker(clustered_rgb_path)

            # Create or load the materials RGB+SWIR dict
            create_new_materials_rgb_swir_dict = self.config['functions']['heuristic_rgb_material_vector_v1'][
                'create_new_materials_rgb_swir_dict']
            if create_new_materials_rgb_swir_dict:
                materials_labels_rgb_swir_dict = self.materials_rgb_dict_creator(materials_df)
                print("new materials_labels_rgb_swir_dict was created")
                self.yaml_dumper(materials_labels_rgb_swir_dict,
                                 os.path.join(self.path['results'], 'heuristic_rgb_material_vector_v1',
                                              folder_name, 'material_rgb_dict.yml'))
            else:
                materials_labels_rgb_swir_dict = self.yaml_loader(
                    os.path.join(self.path['results'], 'heuristic_rgb_material_vector_v1', folder_name,
                                 'material_rgb_dict.yml'))
                print("materials_labels_rgb_swir_dict was lodaed")

            # Determine whether to generate clustered RGB images
            create_new_clustered_rgb_folder = self.config['functions']['heuristic_rgb_material_vector_v1'][
                'create_new_clustered_rgb_folder']
            use_clustered_rgb_folder = self.config['functions']['heuristic_rgb_material_vector_v1'][
                'use_clustered_base_rgb_folder']

            if create_new_clustered_rgb_folder:
                rgb_frames_list = np.sort(os.listdir(main_rgb_path))
                self.cluster_rgb(main_rgb_path, clustered_rgb_path, rgb_frames_list)

            # Load hot vector (heuristic eligibility matrix)
            hot_vector_csv_path = os.path.join(self.path['results'], 'seg_mapping',
                                               self.config['functions']['heuristic_rgb_material_vector_v1'][
                                                   'hot_vector_name'])
            hot_vector_df = pd.read_csv(hot_vector_csv_path)
            print("hot_vector_df was loaded")

            seg_frames_list = np.sort(os.listdir(main_seg_path))
            if create_new_clustered_rgb_folder:
                clu_frames_list = np.sort(os.listdir(clustered_rgb_path))
                main_clu_path = clustered_rgb_path
            elif use_clustered_rgb_folder:
                clu_frames_list = np.sort(os.listdir(clustered_rgb_path))
                main_clu_path = clustered_rgb_path
            else:
                clu_frames_list = np.sort(os.listdir(main_rgb_path))
                main_clu_path = main_rgb_path
            rgb_frame_list = np.sort(os.listdir(main_rgb_path))

            # Compute or load clustered RGB mean mappings
            create_new_clu_rgb_to_rgb_dict_and_unique_clu_rgb_per_obj = \
            self.config['functions']['heuristic_rgb_material_vector_v1'][
                'create_new_clu_rgb_to_rgb_dict_and_unique_clu_rgb_per_obj']

            if create_new_clu_rgb_to_rgb_dict_and_unique_clu_rgb_per_obj:
                unique_clu_rgb_and_mean_rgb_dict, unique_clu_rgb_per_object_dict = self.unique_clu_rgb_to_mean_rgb_and_seg_to_unique_rgb(
                    clu_frames_list, rgb_frame_list, seg_frames_list, main_rgb_path, main_clu_path,
                    main_seg_path, objects_seg_rgb_data)

            # Normalize clustered RGB dict keys to standard format
            unique_clu_rgb_and_mean_rgb_dict_with_ordered_keys = self.order_string_keys(
                unique_clu_rgb_and_mean_rgb_dict)

            # Compute the score matrix or load it from cache
            create_new_scores_per_objects_mat = self.config['functions']['heuristic_rgb_material_vector_v1'][
                'create_new_scores_mat_with_heuristics']


            if create_new_scores_per_objects_mat == False:
                scores_per_objects_mat = self.pickle_loader(
                    os.path.join(self.path['results'], 'heuristic_rgb_material_vector_v1', folder_name,
                                 'scores_mat_objects_dict_after_heur.pickle'))
                print("scores_mat_objects_dict_after_heur was loaded")
            else:
                scores_per_objects_mat = self.rgb_scores_objects_dict(hot_vector_df,
                                                                                    unique_clu_rgb_per_object_dict,
                                                                                    materials_labels_rgb_swir_dict,
                                                                                    unique_clu_rgb_and_mean_rgb_dict_with_ordered_keys,
                                                                                    objects_seg_rgb_data)
                self.pickle_dumper(
                    os.path.join(self.path['results'], 'heuristic_rgb_material_vector_v1', folder_name,
                                 'scores_mat_objects_dict_after_heur.pickle'), scores_per_objects_mat)
                print("scores_mat_objects_dict_after_heur was created")



    """ - -----------------------------  Frames segments Classifier - ----------------------------------------- """

    def vgg19_classifier(self):
        """
        Uses a pretrained finetuned VGG19-based segmentation model to classify frames into semantic masks.
        Inputs:
        - Model weights from path defined in config
        - Folder of test RGB frames (images)
        - Corresponding GT masks assumed to exist in parallel
        Process:
        1. Load model with VGG19 backbone and the trained weights
        2. Normalize and prepare each test image
        3. Predict a segmentation mask per image
        4. Save results: predicted mask, original image, side-by-side plot of input/GT/prediction
        Outputs:
        - Writes predicted masks to 'masks' folder
        - Writes plots to 'test' folder for visualization
        - Writes input images to 'images' folder
        """
        # Load parameters from config
        matplotlib.use('Agg')
        num_classes =  self.config['functions']['vgg19_classifier']['num_classes']
        path_to_trained_weights = self.config['functions']['vgg19_classifier']['models_weights_path']
        test_folder_name = self.config['functions']['vgg19_classifier']['test_folder_name']
        path_to_input_images_folder = self.config['functions']['vgg19_classifier']['path_to_input_images_folder']
        path_to_output_results = os.path.join(self.path['results'], 'vgg19_classifier', test_folder_name)


        folders_names_list = ['masks', 'images', 'test']
        for folder_name in folders_names_list:
            self.folder_checker(os.path.join(path_to_output_results, folder_name))

        # Initialize model with the correct number of classes and no pretrained ImageNet weights
        vgg19 = SegmentationModel(num_classes, False)
        # Preprocessing pipeline: resize, tensor conversion, and normalization
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # Load model weights
        vgg19.load_state_dict(torch.load(path_to_trained_weights, map_location=torch.device('cpu')))
        vgg19.eval()  # Set model to evaluation mode
        # Process each frame in the test folder
        for frame in os.listdir(path_to_input_images_folder):
            full_input_frame_path = os.path.join(path_to_input_images_folder, frame)
            # Load the input frame
            image_cv = cv2.imread(full_input_frame_path, cv2.IMREAD_UNCHANGED)
            image = Image.fromarray(image_cv)
            input_tensor = transform(image)
            mask_cv_gt = cv2.imread(full_input_frame_path.replace('images', 'masks'), cv2.IMREAD_UNCHANGED)
            # Pass the input frame through the model to obtain the predicted mask
            with torch.no_grad():
                predicted_output = vgg19(input_tensor)

            predicted_mask = torch.argmax(predicted_output, dim=0).numpy().astype('uint8')
            cv2.imwrite(os.path.join(path_to_output_results, 'masks', frame), predicted_mask)
            cv2.imwrite((os.path.join(path_to_output_results, 'images', frame)), image_cv)

            plt.figure(figsize=(15, 5))

            # Input frame
            plt.subplot(1, 3, 1)
            plt.imshow(image_cv)
            plt.title("Input Frame")
            plt.axis("off")

            # Ground truth mask
            plt.subplot(1, 3, 2)
            plt.imshow(mask_cv_gt, cmap='jet')
            plt.title("Ground Truth Mask")
            plt.axis("off")

            # Predicted mask
            plt.subplot(1, 3, 3)
            plt.imshow(predicted_mask, cmap='jet')
            plt.title("Predicted Mask")
            plt.axis("off")

            plt.tight_layout()
            plt.show()

            plt.savefig((os.path.join(path_to_output_results, 'test', frame)))




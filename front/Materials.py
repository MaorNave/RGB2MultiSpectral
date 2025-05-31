import os.path
import json
import pickle
from tqdm import tqdm
from scipy.spatial import cKDTree
import cv2
import yaml
from yaml.loader import SafeLoader
import numpy as np
import os
import spectral
import time
import multiprocessing as mp


class Materials():
    """
    A class to handle all processing related to material heuristics, spectral cube generation,
    RGB clustering, YAML/JSON/PKL file handling, and multiprocessing-based cube creation.
    Inputs:
    - Configuration file named 'config_PRTORAD.yaml' loaded on initialization
    Process:
    - Loads configuration paths and stores them for later use
    - Provides utility functions for serialization and folder checks
    - Supports material assignment, reflectance vector generation, and cube writing
    Output:
    - Materials instance used for generating hyperspectral cubes and masks
    """

    def __init__(self):

        with open('config_PRTORAD.yaml') as f:
            data = yaml.load(f, Loader=SafeLoader)
        f.close()
        # Save loaded configuration and set relevant path mappings
        self.config = data
        self.path = {'main_pipeline_path': self.config['general_params']['main_pipeline_path'],
            'results': self.config['general_params']['results_path'],
                  'test': self.config['general_params']['test_path'],
                     'input': self.config['general_params']['input_path']}


    """ - -----------------------------  Static methods - ----------------------------------------- """

    @staticmethod
    def folder_checker(path):
        """
        Check if a directory exists at the specified path; create it if it does not.
        Inputs:
        - path: Full path to the folder to check
        Process:
        - Uses os.path.exists to determine if path exists
        - Creates the folder using os.makedirs if needed
        Output:
        - Prints a message whether the folder was created or already exists
        """
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Folders created for the following path : {path}" )
        else:
            print(f" Folder path are already exist for : {path}")

    @staticmethod
    def yaml_dumper(data, path):
        """
        Dump a Python dictionary into a YAML file.
        Inputs:
        - data: Python dictionary to save
        - path: Output file path for the YAML
        Process:
        - Opens the file in write mode and writes YAML content using PyYAML
        Output:
        - Creates or overwrites the YAML file at the specified path
        """
        with open(path, 'w') as outfile:
            yaml.dump(data, outfile, sort_keys=False, default_flow_style=None)

    @staticmethod
    def json_dumper(path, data):
        """
        Dump a Python dictionary into a JSON file.
        Inputs:
        - path: Output file path for the JSON
        - data: Dictionary to serialize
        Process:
        - Opens the file in write mode and uses json.dump to write the data
        Output:
        - JSON file written at specified path
        """
        with open(path, 'w') as f:
            json.dump(data, f)
        f.close()

    @staticmethod
    def pickle_dumper(path, data):
        """
        Serialize a Python object to a binary Pickle file.
        Inputs:
        - path: File path to save the Pickle file
        - data: Python object to serialize
        Process:
        - Opens file in binary write mode and uses pickle.dump to save object
        Output:
        - Binary Pickle file saved at specified path
        """
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        f.close()

    @staticmethod
    def pickle_loader(path):
        """
        Load a Python object from a Pickle file.
        Inputs:
        - path: File path of the Pickle file
        Process:
        - Opens file in binary read mode and deserializes the object using pickle.load
        Output:
        - Returns the Python object loaded from Pickle
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        f.close()
        return data

    @staticmethod
    def json_loader(path):
        """
        Load a Python dictionary from a JSON file.
        Inputs:
        - path: File path to the JSON file
        Process:
        - Opens the file and uses json.load to parse its content
        Output:
        - Returns the dictionary loaded from JSON
        """
        with open(path, 'r') as f:
            data = json.load(f)
        f.close()
        return data


    @staticmethod
    def yaml_loader(path):
        """
        Load a Python dictionary from a YAML file.
        Inputs:
        - path: File path to the YAML file
        Process:
        - Opens the file and reads the content using PyYAML SafeLoader
        Output:
        - Dictionary object parsed from the YAML file
        """
        with open(path) as f:
            data = yaml.load(f, Loader=SafeLoader)
        f.close()
        return data


    """ - -----------------------------  cubes_Generator- ----------------------------------------- """


    def string_list_to_int(self, seg_rgb):
        """
        Convert a string-formatted list of integers to a list of integers.
        Inputs:
        - seg_rgb: A list containing one string, e.g., ['123 45 67']
        Process:
        - Splits the string into parts based on whitespace
        - Converts each numeric string to an integer
        - Handles both well-formed (length 3) and malformed input cases
        Output:
        - List of integers representing RGB values, e.g., [123, 45, 67]
        """
        seg_list = seg_rgb[0].split(' ')
        seg_list_copy = []
        if len(seg_list) != 3:
            for num in seg_list:
                if num.isdigit():
                    seg_list_copy.append(int(num))
        else:
            for num in seg_list:
                seg_list_copy.append(int(num))

        return seg_list_copy

    @staticmethod
    def convert_list(rgb):
        """
        Convert a string of RGB values with parentheses or brackets into a list of integers.
        Inputs:
        - rgb: String format of RGB list, e.g., '(12, 45, 78)' or '[12, 45, 78]'
        Process:
        - Strips the enclosing characters
        - Splits the string by commas
        - Converts each item to an integer
        Output:
        - A list of three integers representing RGB, e.g., [12, 45, 78]
        """
        striped_rgb = rgb.strip()
        new_rgb = striped_rgb[1:-1]
        new_rgb = new_rgb.split(',')
        return [int(x) for x in new_rgb]

    @staticmethod
    def clu_rgb_to_materials_labels_rgb_swir_dict_raffle(clu_rgb, obj):
        """
        Retrieve a reflectance vector for a clustered RGB by raffling one of the top-5 most similar materials.
        Inputs:
        - clu_rgb: Clustered RGB value (mean RGB of segment region)
        - obj: Object name to retrieve material scores from global score table
        Process:
        - Converts the RGB vector into a string to match score matrix column
        - Retrieves top 5 candidates with highest similarity scores
        - Performs weighted random selection (raffle) from the candidates
        - Handles possible exceptions in RGB formatting or lookup
        Output:
        - Tuple with (RGB value, 2150D reflectance vector, material name, class RGB, raffle score)
        - Returns np.zeros(2150) if no material is found
        """

        try:
            # Convert RGB to string format used in score matrix
            column_rgb_name = np.array2string(clu_rgb, separator=' ')
            # Get top-5 materials with highest scores for the RGB
            max_score_candidates = (objects_clu_rgb_mat[obj][column_rgb_name]).sort_values(ascending=False).head(5)
            probabilities = max_score_candidates / max_score_candidates.sum()
            raffle_score = np.random.choice(max_score_candidates, p=probabilities)
            # If the best score is 0, assign a null material
            if raffle_score == 0:
                return (np.zeros(2150))
            # Find the material name matching the selected score
            material_name = objects_clu_rgb_mat[obj][column_rgb_name].loc[
                objects_clu_rgb_mat[obj][column_rgb_name] == raffle_score].index
        except Exception as e:
            print(e)
            # If the RGB format fails, try converting it again more robustly
            try:
                column_rgb_name = Materials.convert_list(column_rgb_name)
            except:
                return (np.zeros(2150))

            # Use the re-formatted RGB string
            column_rgb_name = str(column_rgb_name)
            max_score_candidates = (objects_clu_rgb_mat[obj][column_rgb_name]).sort_values(ascending=False).head(5)
            probabilities = max_score_candidates / max_score_candidates.sum()
            raffle_score = np.random.choice(max_score_candidates, p=probabilities)


            if raffle_score == 0:
                return (np.zeros(2150))

            material_name = objects_clu_rgb_mat[obj][column_rgb_name].loc[
                objects_clu_rgb_mat[obj][column_rgb_name] == raffle_score].index

        # Return material components: RGB, reflectance, class name RGB, score, etc.
        return (materials_labels_rgb_swir_dict[material_name[0]][0],
                materials_labels_rgb_swir_dict[material_name[0]][1],
                materials_labels_rgb_swir_dict[material_name[0]][2],
                materials_labels_rgb_swir_dict[material_name[0]][3],
                raffle_score)

    @staticmethod
    def clu_rgb_to_materials_labels_rgb_swir_dict(clu_rgb, obj):
        """
        Retrieve a reflectance vector for a clustered RGB using the highest similarity score (max score).
        Inputs:
        - clu_rgb: Clustered RGB vector (e.g., from segmented region)
        - obj: Object name to match against the score matrix
        Process:
        - Converts the RGB into a string key for matrix lookup
        - Identifies the material with the highest similarity score
        - Uses exception handling to retry with formatted RGB if lookup fails
        Output:
        - Tuple with (RGB value, 2150D reflectance vector, material name, class RGB, max score)
        - Returns np.zeros(2150) if no valid material found
        """

        try:
            column_rgb_name = np.array2string(clu_rgb, separator=' ')
            max_score = (objects_clu_rgb_mat[obj][column_rgb_name]).max()
            # Return a zero vector if the best score is zero
            if max_score == 0:
                return (np.zeros(2150))
            # Get material name(s) with the highest score
            material_name = objects_clu_rgb_mat[obj][column_rgb_name].loc[
                objects_clu_rgb_mat[obj][column_rgb_name] == max_score].index
        except Exception as e:
            print(e)
            try:
                column_rgb_name = Materials.convert_list(column_rgb_name)
            except:
                return (np.zeros(2150))
            # Retry lookup with reformatted RGB
            column_rgb_name = str(column_rgb_name)
            max_score = objects_clu_rgb_mat[obj][column_rgb_name].max()

            if max_score == 0:
                return (np.zeros(2150))
            material_name = objects_clu_rgb_mat[obj][column_rgb_name].loc[
                objects_clu_rgb_mat[obj][column_rgb_name] == max_score].index

        # Return full material metadata tuple
        return (materials_labels_rgb_swir_dict[material_name[0]][0],
                materials_labels_rgb_swir_dict[material_name[0]][1],
                materials_labels_rgb_swir_dict[material_name[0]][2],
                materials_labels_rgb_swir_dict[material_name[0]][3],
                max_score)

    @staticmethod
    def indices_in_segment_object(flat_seg_img, seg):
        """
        Find all indices in a flat segmentation image that match a specific segment RGB.
        Inputs:
        - flat_seg_img: Flattened segmentation mask (1D array or flattened 2D mask)
        - seg: Segment RGB value (either list or malformed string)
        Process:
        - Uses np.where to identify pixel positions where RGB values match the segment
        - If direct match fails, tries to convert the segment from string to list of ints
        Output:
        - 1D array of indices where the segment appears in the flattened mask
        """
        try:
            obj_seg_indices = np.where(flat_seg_img == seg)[0]
        except:
            seg_rgb_int = Materials.string_list_to_int(seg)
            obj_seg_indices = np.where(flat_seg_img == seg_rgb_int)[0]
        return obj_seg_indices


    @staticmethod
    def cube_generator(item_tup):
        """
        Generate a hyperspectral cube and related masks using the deterministic max-score method.
        Inputs:
        - item_tup: Tuple containing (seg_frame_path, rgb_frame_path)
        Process:
        - Loads segmentation and RGB frames
        - Finds clustered RGB values for each segment object
        - Maps each RGB to its highest-scoring material (reflectance, RGB, etc.)
        - Fills spectral cube and masks for RGB, type, and heatmap
        - Completes missing regions using KDTree and nearest-neighbor material filling
        - Saves the generated data as .npy and image files
        Output:
        - Saves cube, masks, and material dictionaries to configured output folders
        """
        # Unpack input paths
        seg_frame_path=item_tup[0]
        rgb_frame_path=item_tup[1]
        frame_number = rgb_frame_path.split('\\')[-1].split('.')[0]
        # Initialize empty data containers
        swir_frame = np.zeros((224, 224, 2150))
        rgb_mask_frame = np.zeros((224, 224, 3)) # rgb presentation of materials mask
        materials_type_seg_mask = np.zeros((224, 224, 3)) # by RGB value of material type.
        materials_heat_map_mask = np.zeros((224, 224, 1)) # by max_score that has been implemented.
        frame_materials_dict = dict()
        frame_seg = cv2.imread(seg_frame_path,cv2.IMREAD_UNCHANGED)
        flat_seg_img = frame_seg.reshape(-1)
        frame_clu = cv2.imread(rgb_frame_path, cv2.IMREAD_UNCHANGED)
        flat_clu_frame = frame_clu.reshape(-1, frame_clu.shape[2])
        # Build a dictionary of object names to pixel indices in the segmentation image
        objects_seg_rgb_dict_indices = {obj: Materials.indices_in_segment_object(flat_seg_img, seg_rgb) \
                                        for obj, seg_rgb in objects_seg_rgb_dict_data.items()}
        # Filter out objects that have no presence in the current frame
        objects_seg_rgb_dict_relevant = {obj: obj_seg_indices for obj, obj_seg_indices in
                                         objects_seg_rgb_dict_indices.items()
                                         if len(obj_seg_indices) > 0}
        # Iterate through each object and its pixel indices
        for obj, obj_seg_indices in objects_seg_rgb_dict_relevant.items():
            clu_rgb_on_seg_list = flat_clu_frame[obj_seg_indices]
            unique_clu_rgb = np.unique(clu_rgb_on_seg_list, axis=0)
            # Map each unique RGB to its best scoring material
            material_swir_transplant_dict_unique = {np.array2string(clu_rgb, separator=' '): \
                                                        Materials.clu_rgb_to_materials_labels_rgb_swir_dict(clu_rgb, obj)
                                                    for clu_rgb in unique_clu_rgb}
            frame_materials_dict[obj] = material_swir_transplant_dict_unique
            material_swir_transplant_list = [
                material_swir_transplant_dict_unique[np.array2string(clu_rgb, separator=' ')]
                for clu_rgb in clu_rgb_on_seg_list]

            # Map all pixel indices to their 2D (row, col) location and fill cube/masks
            for j, index in enumerate(obj_seg_indices):
                row, col = np.unravel_index(index, frame_clu.shape[:2])

                swir_frame[row][col] = material_swir_transplant_list[j][1]
                if len(material_swir_transplant_list[j]) == 5:

                    rgb_mask_frame[row][col] = material_swir_transplant_list[j][0]
                    materials_type_seg_mask[row][col] = material_swir_transplant_list[j][3]
                    materials_heat_map_mask[row][col] = material_swir_transplant_list[j][4]

                else:  # Skip if max_score == 0 (unassigned pixel)
                    continue


        # Post-processing: fill zero-valued pixels using nearest-neighbor interpolation
        zero_coords = np.argwhere(np.all(swir_frame == np.zeros(2150), axis=-1))
        non_zero_coords = np.argwhere(np.all(swir_frame != np.zeros(2150), axis=-1))
        # Create a KDTree using the coordinates of non-zero pixels
        tree = cKDTree(non_zero_coords)
        # Num of closest pixel to consider.
        num_closest = 10 # Arbitrary – if 10 neighbors are all zero, the search will continue expanding until relevant neighbors are found.
        # Iterate over each zero-valued pixel
        for zero_coord in zero_coords:
            _, indices = tree.query(zero_coord, k=num_closest)
            rel_index = np.random.choice(indices)
            row_index, col_index = np.unravel_index(rel_index, swir_frame.shape[:-1])

            # Copy spectral + RGB + type + heatmap info from selected neighbor
            material_signature = swir_frame[row_index, col_index]
            material_color_rgb_val = rgb_mask_frame[row_index, col_index]
            material_type_seg = materials_type_seg_mask[row_index, col_index]
            material_heat_map_value = materials_heat_map_mask[row_index, col_index]

            # Replace the zero-valued pixel with the randomly selected non-zero pixel value
            try:
                swir_frame[tuple(zero_coord)] = material_signature
            except:
                print('exception in material transplanting')
            rgb_mask_frame[tuple(zero_coord)] = material_color_rgb_val
            materials_type_seg_mask[tuple(zero_coord)] = material_type_seg
            materials_heat_map_mask[tuple(zero_coord)] = material_heat_map_value


        new_name = frame_number + '_cube.npy'
        rgb_mask_frame_new_name = frame_number + '_rgb_mask.tiff'
        psedo_rgb_mask_frame_new_name = frame_number + '_psedo_rgb_mask.png'
        materials_type_seg_mask_new_name = frame_number + '_material_type_mask.tiff'
        materials_heat_map_mask_new_name = frame_number + '_heatmap_mask.tiff'
        
        # Save spectral cube and all related masks to disk
        np.save(os.path.join(output_folder, new_name), swir_frame.astype(np.float16))
        cv2.imwrite(os.path.join(rgb_mask_output_folder, rgb_mask_frame_new_name), cv2.cvtColor(rgb_mask_frame.astype('uint8'), cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(material_type_mask_folder, materials_type_seg_mask_new_name), materials_type_seg_mask.astype(int))
        cv2.imwrite(os.path.join(heatmap_mask_folder, materials_heat_map_mask_new_name), materials_heat_map_mask)
        spectral.save_rgb(os.path.join(psedo_rgb_mask_output_folder, psedo_rgb_mask_frame_new_name)
            , swir_frame)
        Materials.pickle_dumper(os.path.join(test_output_mat_dict_folder, new_name.replace('.npy', '.pickle')), frame_materials_dict)



    def worker_init_cubes_gen(self, global_args_list):
        """
        Initialize global variables for multiprocessing context during cube generation.
        Inputs:
        - global_args_list: A list of 9 elements:
            0: materials_labels_rgb_swir_dict
            1: objects_clu_rgb_mat
            2: objects_seg_rgb_dict_data
            3–8: various output folders
        Process:
        - Sets each element in the input list as a global variable used by subprocesses
        Output:
        - Global context set for use inside multiprocessing pool workers
        """
        global materials_labels_rgb_swir_dict
        global objects_clu_rgb_mat
        global objects_seg_rgb_dict_data
        global output_folder
        global test_output_mat_dict_folder
        global rgb_mask_output_folder
        global psedo_rgb_mask_output_folder
        global material_type_mask_folder
        global heatmap_mask_folder

        materials_labels_rgb_swir_dict = global_args_list[0]
        objects_clu_rgb_mat = global_args_list[1]
        objects_seg_rgb_dict_data = global_args_list[2]
        output_folder = global_args_list[3]
        test_output_mat_dict_folder = global_args_list[4]
        rgb_mask_output_folder = global_args_list[5]
        psedo_rgb_mask_output_folder = global_args_list[6]
        material_type_mask_folder = global_args_list[7]
        heatmap_mask_folder = global_args_list[8]

    def cubes_generator(self):
        """
        Launch parallel spectral cube generation for a configured list of test folders using max score method.
        Inputs:
        - None directly; uses configuration from `self.config['cubes_generator']['test_folders_list']`
        Process:
        - Loads segmentation and RGB cluster frames
        - Loads material score dictionaries and mappings from disk
        - Prepares output folders for results
        - Iterates over all frames and applies multiprocessing to run `cube_generator` in parallel
        Output:
        - Saves cubes and associated masks for each test folder
        """

        test_folders_list = self.config['cubes_generator']['test_folders_list']
        for test_folder in tqdm(test_folders_list):
            main_seg_path = os.path.join(self.path['results'], 'vgg19_classifier', test_folder, 'masks')
            main_clu_rgb_path = os.path.join(self.path['results'], 'heuristic_rgb_material_vector_v1', test_folder, 'clustered_rgb')
            materials_labels_rgb_swir_dict = self.yaml_loader(os.path.join(self.path['results'], 'heuristic_rgb_material_vector_v1' ,test_folder, self.config['cubes_generator']['materials_rgb_swir']))
            objects_clu_rgb_mat = self.pickle_loader(os.path.join(self.path['results'], 'heuristic_rgb_material_vector_v1', test_folder, self.config['cubes_generator']['objects_base_rgb_scores_mat']))

            objects_seg_rgb_dict_data = self.yaml_loader(os.path.join(self.path['input'], 'seg_mapping',
                                                          self.config['cubes_generator']['obj_seg_rgb']))['labels_numbers_dict']

            # Build relevant output paths for all result types
            test_output_mat_dict_folder = r"\\?\\" + os.path.join(self.path['test'], 'cubes_generator', test_folder, 'Max_score\\cubes_materials_dict')
            output_folder = r"\\?\\" + os.path.join(self.path['results'], 'cubes_generator',test_folder, 'Max_score\\cubes')
            rgb_mask_output_folder = r"\\?\\" + os.path.join(self.path['results'], 'cubes_generator', test_folder, 'Max_score\\masks\\rgb_masks')
            heatmap_mask_folder = r"\\?\\" + os.path.join(self.path['results'], 'cubes_generator', test_folder, 'Max_score\\masks\\heatmap_mask')
            psedo_rgb_mask_output_folder = r"\\?\\" + os.path.join(self.path['results'], 'cubes_generator', test_folder, 'Max_score\\masks\\psedo_rgb_masks')
            material_type_mask_folder = r"\\?\\" + os.path.join(self.path['results'], 'cubes_generator', test_folder, 'Max_score\\masks\\material_type_mask')


            output_path_list = [test_output_mat_dict_folder ,output_folder,  rgb_mask_output_folder\
                , material_type_mask_folder, heatmap_mask_folder, psedo_rgb_mask_output_folder]
            # Ensure all output folders exist
            for path in output_path_list:
                self.folder_checker(path)

            # Sort image files numerically to align segmentation and RGB
            seg_frames_list = np.sort(os.listdir(main_seg_path))
            clu_frames_list = np.sort(os.listdir(main_clu_rgb_path))

            # prepare to Parallel frames
            items = [(os.path.join(main_seg_path, seg), os.path.join(main_clu_rgb_path, rgb))  \
                     for seg, rgb in \
                     zip(seg_frames_list, clu_frames_list)]
            # Prepare arguments to be broadcast to each multiprocessing worker
            global_args_list = [materials_labels_rgb_swir_dict, objects_clu_rgb_mat, objects_seg_rgb_dict_data,
                                output_folder, test_output_mat_dict_folder, rgb_mask_output_folder, psedo_rgb_mask_output_folder,
                                material_type_mask_folder, heatmap_mask_folder
                                ]

            print('start pooling')
            p = mp.Pool(self.config['general_params']['parallel_workers'], initializer=self.worker_init_cubes_gen, initargs=(global_args_list,))

            results = p.map(Materials.cube_generator, [arg_tup for arg_tup in items])


    @staticmethod
    def parallel_raffle_cubes(item_tup):
        """
        Generate a hyperspectral cube using a raffled material selection from the top-5 candidates.
        Inputs:
        - item_tup: Tuple of (seg_frame_path, rgb_frame_path)
        Process:
        - Loads segmentation and clustered RGB image
        - For each segment object:
            - Extracts RGBs
            - Identifies unique RGBs
            - For each, selects a material by raffling one from the top-5 scores
        - Builds spectral cube and related masks
        - Fills missing pixels using nearest neighbor interpolation
        - Saves the generated data
        Output:
        - Saves cube, RGB/type/heatmap masks, and material dictionary for each frame
        """
        # Unpack segmentation and RGB image paths
        seg_frame_path=item_tup[0]
        rgb_frame_path=item_tup[1]
        frame_number = rgb_frame_path.split('\\')[-1].split('.')[0]
        # Initialize result containers
        swir_frame = np.zeros((224, 224, 2150))
        rgb_mask_frame = np.zeros((224, 224, 3)) # rgb presentation of materials mask
        materials_type_seg_mask = np.zeros((224, 224, 3)) # by RGB value of material type.
        materials_heat_map_mask = np.zeros((224, 224, 1)) # by raffle_score that has been implemented.
        frame_materials_dict = dict()
        # Load segmentation and RGB data
        frame_seg = cv2.imread(seg_frame_path,cv2.IMREAD_UNCHANGED)
        flat_seg_img = frame_seg.reshape(-1)
        frame_clu = cv2.imread(rgb_frame_path, cv2.IMREAD_UNCHANGED)
        flat_clu_frame = frame_clu.reshape(-1, frame_clu.shape[2])
        # Identify indices per segment object
        objects_seg_rgb_dict_indices = {obj: Materials.indices_in_segment_object(flat_seg_img, seg_rgb) \
                                        for obj, seg_rgb in objects_seg_rgb_dict_data.items()}
        # Filter only objects present in the current frame
        objects_seg_rgb_dict_relevant = {obj: obj_seg_indices for obj, obj_seg_indices in
                                         objects_seg_rgb_dict_indices.items()
                                         if len(obj_seg_indices) > 0}
        # Iterate over relevant segments
        for obj, obj_seg_indices in objects_seg_rgb_dict_relevant.items():
            clu_rgb_on_seg_list = flat_clu_frame[obj_seg_indices]
            unique_clu_rgb = np.unique(clu_rgb_on_seg_list, axis=0)

            # Raffle material for each unique RGB
            material_swir_transplant_dict_unique = {np.array2string(clu_rgb, separator=' '): \
                                                        Materials.clu_rgb_to_materials_labels_rgb_swir_dict_raffle(clu_rgb, obj)
                                                    for clu_rgb in unique_clu_rgb}

            frame_materials_dict[obj] = material_swir_transplant_dict_unique
            material_swir_transplant_list = [
                material_swir_transplant_dict_unique[np.array2string(clu_rgb, separator=' ')]
                for clu_rgb in clu_rgb_on_seg_list]

            # Fill cube and masks per pixel
            for j, index in enumerate(obj_seg_indices):
                row, col = np.unravel_index(index, frame_clu.shape[:2])

                swir_frame[row][col] = material_swir_transplant_list[j][1]
                if len(material_swir_transplant_list[j]) == 5:

                    rgb_mask_frame[row][col] = material_swir_transplant_list[j][0]
                    materials_type_seg_mask[row][col] = material_swir_transplant_list[j][3]
                    materials_heat_map_mask[row][col] = material_swir_transplant_list[j][4]

                else: # for raffle_score == 0 materials transplant
                    continue

        # Fill in missing values using KDTree-based interpolation
        zero_coords = np.argwhere(np.all(swir_frame == np.zeros(2150), axis=-1))
        non_zero_coords = np.argwhere(np.all(swir_frame != np.zeros(2150), axis=-1))
        tree = cKDTree(non_zero_coords)
        num_closest = 10

        for zero_coord in zero_coords:
            _, indices = tree.query(zero_coord, k=num_closest)
            rel_index = np.random.choice(indices)
            row_index, col_index = np.unravel_index(rel_index, swir_frame.shape[:-1])

            material_signature = swir_frame[row_index, col_index]
            material_color_rgb_val = rgb_mask_frame[row_index, col_index]
            material_type_seg = materials_type_seg_mask[row_index, col_index]
            material_heat_map_value = materials_heat_map_mask[row_index, col_index]

            try:
                swir_frame[tuple(zero_coord)] = material_signature
            except:
                print('exception in material transplanting')
            rgb_mask_frame[tuple(zero_coord)] = material_color_rgb_val
            materials_type_seg_mask[tuple(zero_coord)] = material_type_seg
            materials_heat_map_mask[tuple(zero_coord)] = material_heat_map_value

        # Save all generated data
        new_name = frame_number + '_cube.npy'
        rgb_mask_frame_new_name = frame_number + '_rgb_mask.tiff'
        psedo_rgb_mask_frame_new_name = frame_number + '_psedo_rgb_mask.png'
        materials_type_seg_mask_new_name = frame_number + '_material_type_mask.tiff'
        materials_heat_map_mask_new_name = frame_number + '_heatmap_mask.tiff'

        np.save(os.path.join(output_folder, new_name), swir_frame.astype(np.float16))
        cv2.imwrite(os.path.join(rgb_mask_output_folder, rgb_mask_frame_new_name), cv2.cvtColor(rgb_mask_frame.astype('uint8'), cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(material_type_mask_folder, materials_type_seg_mask_new_name), materials_type_seg_mask.astype(int))
        cv2.imwrite(os.path.join(heatmap_mask_folder, materials_heat_map_mask_new_name), materials_heat_map_mask)
        spectral.save_rgb(os.path.join(psedo_rgb_mask_output_folder, psedo_rgb_mask_frame_new_name)
            , swir_frame)
        Materials.pickle_dumper(os.path.join(test_output_mat_dict_folder, new_name.replace('.npy', '.pickle')), frame_materials_dict)


    def worker_init_cubes_gen_raffle(self, global_args_list):
        """
        Initialize global variables for multiprocessing in the raffle cube generation mode.
        Inputs:
        - global_args_list: List of required global structures and paths
        Process:
        - Sets each entry in the list as a global used by pool workers
        Output:
        - Globals initialized for raffle-based cube generation
        """
        global materials_labels_rgb_swir_dict
        global objects_clu_rgb_mat
        global objects_seg_rgb_dict_data
        global output_folder
        global test_output_mat_dict_folder
        global rgb_mask_output_folder
        global psedo_rgb_mask_output_folder
        global material_type_mask_folder
        global heatmap_mask_folder

        materials_labels_rgb_swir_dict = global_args_list[0]
        objects_clu_rgb_mat = global_args_list[1]
        objects_seg_rgb_dict_data = global_args_list[2]
        output_folder = global_args_list[3]
        test_output_mat_dict_folder = global_args_list[4]
        rgb_mask_output_folder = global_args_list[5]
        psedo_rgb_mask_output_folder = global_args_list[6]
        material_type_mask_folder = global_args_list[7]
        heatmap_mask_folder = global_args_list[8]

    def cubes_generator_raffle(self):
        """
        Generate spectral cubes using a raffled material assignment method over test folder frames.
        Inputs:
        - None directly; uses `self.config` to fetch folder and file paths self.config['cubes_generator']['test_folders_list']
        Process:
        - Loads segmentation & RGB data, material labels, and score dictionaries
        - Prepares output folder structure
        - Launches a multiprocessing pool that applies `parallel_raffle_cubes` across frame pairs
        Output:
        - Saves raffled cubes, masks, and materials metadata to disk
        """
        test_folders_list = self.config['cubes_generator']['test_folders_list']
        for test_folder in tqdm(test_folders_list):
            main_seg_path = os.path.join(self.path['results'], 'vgg19_classifier', test_folder, 'masks')
            main_clu_rgb_path = os.path.join(self.path['results'], 'heuristic_rgb_material_vector_v1', test_folder, 'clustered_rgb')
            materials_labels_rgb_swir_dict = self.yaml_loader(os.path.join(self.path['results'], 'heuristic_rgb_material_vector_v1' ,test_folder, self.config['cubes_generator']['materials_rgb_swir']))
            objects_clu_rgb_mat = self.pickle_loader(os.path.join(self.path['results'], 'heuristic_rgb_material_vector_v1', test_folder, self.config['cubes_generator']['objects_base_rgb_scores_mat']))

            objects_seg_rgb_dict_data = self.yaml_loader(os.path.join(self.path['input'], 'seg_mapping',
                                                                      self.config['cubes_generator'][
                                                                          'obj_seg_rgb']))['labels_numbers_dict']

            # Output folder structure
            test_output_mat_dict_folder = os.path.join(self.path['test'], 'cubes_generator', test_folder,
                                                       'raffle_score/cubes_materials_dict')
            output_folder = os.path.join(self.path['results'], 'cubes_generator', test_folder, 'raffle_score/cubes')
            rgb_mask_output_folder = os.path.join(self.path['results'], 'cubes_generator', test_folder,
                                                  'raffle_score/masks/rgb_masks')
            heatmap_mask_folder = os.path.join(self.path['results'], 'cubes_generator', test_folder,
                                               'raffle_score/masks/heatmap_mask')
            psedo_rgb_mask_output_folder = os.path.join(self.path['results'], 'cubes_generator', test_folder,
                                                        'raffle_score/masks/psedo_rgb_masks')
            material_type_mask_folder = os.path.join(self.path['results'], 'cubes_generator', test_folder,
                                                     'raffle_score/masks/material_type_mask')
            # Create output folders
            output_path_list = [test_output_mat_dict_folder, output_folder, rgb_mask_output_folder \
                , material_type_mask_folder, heatmap_mask_folder, psedo_rgb_mask_output_folder]

            for path in output_path_list:
                self.folder_checker(path)

            # Match and zip segmentation + clustered RGB frames
            seg_frames_list = np.sort(os.listdir(main_seg_path))
            clu_frames_list = np.sort(os.listdir(main_clu_rgb_path))
            items = [(os.path.join(main_seg_path, seg), os.path.join(main_clu_rgb_path, rgb)) \
                     for seg, rgb in \
                     zip(seg_frames_list, clu_frames_list)]
            # Setup global state for multiprocessing
            global_args_list = [materials_labels_rgb_swir_dict, objects_clu_rgb_mat, objects_seg_rgb_dict_data,
                                output_folder, test_output_mat_dict_folder, rgb_mask_output_folder,
                                psedo_rgb_mask_output_folder,
                                material_type_mask_folder, heatmap_mask_folder
                                ]

            # Run multiprocessing pool with workers
            print('start pooling')

            start = time.time()
            p = mp.Pool(self.config['general_params']['parallel_workers'], initializer=self.worker_init_cubes_gen_raffle, initargs=(global_args_list,))

            results = p.map(Materials.parallel_raffle_cubes, [arg_tup for arg_tup in items])

            end = time.time()
            time_in_sec = end - start
            time_in_min = time_in_sec / 60
            time_in_hours = time_in_min / 60
            print(f"time for all frames --> {time_in_hours}")
            print('done pooling')


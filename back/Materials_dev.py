import json
import pickle
import yaml
from yaml.loader import SafeLoader
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy import interpolate
from colormath.color_objects import sRGBColor, XYZColor
from colormath.color_conversions import convert_color
from tqdm import tqdm


class Materials_dev():
    """
    Class: Materials_dev
    Description: A class for managing material reflectance data, converting it to RGB representations,
    and serializing it to YAML or CSV formats. Includes preprocessing, interpolation, and CIE-based
    color conversion utilities.
    """
    def __init__(self):
        """
        Initializes wavelength ranges, loads CIE and config data, and prepares paths for further use.
        """
        # Load CIE standard observer color matching functions
        self.cie_data_df = pd.read_csv('CIE_2006_5nm_2deg.csv')
        # Load YAML configuration
        with open('config_Materials_dev.yaml') as f:
            data = yaml.load(f, Loader=SafeLoader)
        f.close()

        self.config = data
        self.inter_vis_wl = np.arange(400,700,1)
        self.raw_materials_wl = np.arange(350,2500)
        self.d65 = self.config['functions']['materials_to_rgb']['d65'][1:]
        self.tamiulus_raw_wl = np.arange(400, 700, 5)
        # Extract CIE XYZ labels (used for tristimulus calculations)
        self.x_label = self.cie_data_df['X'][:-1]
        self.y_label = self.cie_data_df['Y'][:-1]
        self.z_label = self.cie_data_df['Z'][:-1]
        # Define important file paths
        self.path = {'main_pipeline_path': self.config['functions']['general_params']['main_pipeline_path'],
            'results': self.config['functions']['general_params']['results_path'],
                  'test': self.config['functions']['general_params']['test_path']}

    """ - -----------------------------  Static methods - ----------------------------------------- """

    @staticmethod
    def json_dumper(path, data):
        """
        Dumps a Python dictionary to a JSON file.
        """
        with open(path, "w") as outfile:
            json.dump(data, outfile, indent= "")
        outfile.close()

    @staticmethod
    def pickle_dumper(path, data):
        """
        Saves a Python object to a file using pickle.
        """
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        f.close()

    @staticmethod
    def pickle_loader(path):
        """
        Loads a Python object from a pickle file.
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        f.close()
        return data

    @staticmethod
    def json_loader(path):
        """
        Loads data from a JSON file and returns a dictionary.
        """
        with open(path, 'r') as f:
            data = json.load(f)
        f.close()
        return data

    @staticmethod
    def yaml_dumper(data, path):
        """
        Saves a Python dictionary to a YAML file.
        """
        with open(path, 'w') as outfile:
            yaml.dump(data, outfile, sort_keys=False, default_flow_style=None)

    @staticmethod
    def yaml_loader(path):
        """
        Loads a YAML file and returns a Python dictionary.
        """
        with open(path) as f:
            data = yaml.load(f, Loader=SafeLoader)
        f.close()
        return data



    @staticmethod
    def linear_interpolation(wl, raw_wl, raw_ref):
        """
        Performs linear interpolation of reflectance values onto a new wavelength range.
        Parameters:
            wl (np.ndarray): Target wavelengths for interpolation.
            raw_wl (np.ndarray): Original wavelengths.
            raw_ref (np.ndarray): Reflectance values at original wavelengths.
        Returns:
            np.ndarray: Interpolated reflectance values at target wavelengths.
        """
        # Create interpolation function and apply to target wavelengths
        f = interpolate.interp1d(raw_wl, raw_ref, fill_value="extrapolate")
        inter_ref = f(wl)
        return inter_ref



    @staticmethod
    def minus_to_min_values_array_converter(arr):
        """
        Replaces all values <= 0 in the array with the smallest positive value in the array.
        Parameters:
            arr (np.ndarray): Input array.
        Returns:
            np.ndarray: Cleaned array with no non-positive values.
        """
        min_arr_value_gt_0 = min(number for number in arr if number > 0)
        arr_copy = arr.copy()
        for i in range(len(arr)):
            if arr[i]<=0:
                arr_copy[i] = min_arr_value_gt_0
        return arr_copy

    @staticmethod
    def over_one_to_max_values_array_converter(arr):
        """
        Replaces all values >= 1 with the largest value < 1 from the array.
        Parameters:
            arr (np.ndarray): Input array.
        Returns:
            np.ndarray: Cleaned array with no values ≥ 1.
        """
        max_arr_value_lt_1 = max(number for number in arr if number < 1)
        arr_copy = arr.copy()
        for i in range(len(arr)):
            if arr[i]>=1:
                arr_copy[i] = max_arr_value_lt_1
        return arr_copy



    @staticmethod
    def dict_wl_to_ref(data_dict):
        """
        Converts a dictionary with nested wavelength-reflectance mappings
        into a dictionary with lists of reflectance values only.
        Parameters:
            data_dict (dict): Original dictionary with wavelength keys.
        Returns:
            dict: Dictionary with keys mapping to lists of reflectance values.
        """
        data_dict_copy = data_dict.copy()
        for key_material in data_dict.keys():
            raw_ref_list = []
            for wl_key in data_dict[key_material].keys():
                raw_ref = data_dict[key_material][wl_key]
                raw_ref_list.append(raw_ref)

            # Replace wavelength-dict with flat list of reflectances
            data_dict_copy.pop(key_material)
            data_dict_copy[key_material] = raw_ref_list

        return data_dict_copy

    @staticmethod
    def folder_checker(path):
        """
        Ensures that a directory exists at the specified path.
        If it doesn't, the directory is created.
        Parameters:
            path (str): Target folder path.
        """
        # Create folder if it does not exist
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Folders created for the following path : {path}" )
        else:
            print(f" Folder path are already exist for : {path}")


    """ - -----------------------------  Materials_dev data manipulations inner functions - ----------------------------------------- """



    def max_rgb_value_checker(self, rgb_list):
        """
        Ensures RGB values are clipped within the valid 0–255 range.
        Parameters:
            rgb_list (list or np.ndarray): List of RGB values.
        Returns:
            list: Corrected RGB values clipped to [0, 255].
        """
        rgb_list_copy = rgb_list.copy()
        for i in range(len(rgb_list)):
            if rgb_list[i]>255:
                rgb_list_copy[i] = 255
            if rgb_list[i]<0:
                rgb_list_copy[i] = 0
        return rgb_list_copy

    def xyz_to_rgb(self, dict):
        """
        Converts tristimulus XYZ values to sRGB color space.
        Parameters:
            dict (dict): Dictionary with keys 'x', 'y', 'z'.
        Returns:
            np.ndarray: RGB values scaled to 0–255.
        """
        # Create XYZ vector and normalize
        xyz_vector = np.array([dict['x'],dict['y'],dict['z']])
        if xyz_vector.max() > 100:
            xyz_vector = xyz_vector / xyz_vector.max()
        else:
            xyz_vector = xyz_vector / 100
        # Convert from XYZ to sRGB using D65 illuminant
        xyz_vector_color = XYZColor(xyz_vector[0],xyz_vector[1],xyz_vector[2], illuminant='d65')
        simpale_rgb = convert_color(xyz_vector_color, sRGBColor)
        rgb_con = np.array(simpale_rgb.get_value_tuple())
        # Scale to 0–255 and round
        final_rgb_value = (rgb_con*255).round()

        return final_rgb_value

    def trimulus_values_converter(self, *args_trim):
        """
        Computes X, Y, Z tristimulus values by integrating the product of
        material spectral power and CIE color matching functions.
        Parameters:
            *args_trim: (wavelengths, power, X_label, Y_label, Z_label)
        Returns:
            tuple: Summed (X, Y, Z) values.
        """
        x_list, y_list, z_list = [], [], []
        vis_wl = args_trim[0]
        # Normalize power to range 0–1
        if max(args_trim[1])>100:
            material_power = np.array(args_trim[1]) / max(args_trim[1])
        else:
            material_power = np.array(args_trim[1]) / 100
        # Compute weighted sum for each channel
        x_trim = args_trim[2]
        y_trim = args_trim[3]
        z_trim = args_trim[4]
        for i in range(len(vis_wl)):
            x = material_power[i]*x_trim[i]
            x_list.append(x)
            y = material_power[i]*y_trim[i]
            y_list.append(y)
            z = material_power[i]*z_trim[i]
            z_list.append(z)
        # Sum up all values to produce final tristimulus values
        x_final = sum(x_list)
        y_final = sum(y_list)
        z_final = sum(z_list)
        return x_final, y_final, z_final

    def spectral_power_converter(self, *args):
        """
        Calculates the material spectral power by multiplying reflectance by D65 illuminant values.
        Parameters:
            *args: (D65_illuminant, reflectance, wavelengths)
        Returns:
            list: Spectral power values across the visible spectrum.
        """
        s_power_list = []
        vis_wl = args[-1]
        d_65 = args[0]
        ref = args[-2]
        # Multiply reflectance and D65 power at each wavelength
        for i in range(len(vis_wl)):
            s_power = (d_65[i] * ref[i])
            s_power_list.append(s_power)
        return s_power_list



    def array_to_string_defulter(self, arr_str):
        """
        Converts a string representation of a list (with brackets/commas/spaces)
        into a standardized list of integers as a string.
        Parameters:
            arr_str (str): Input string representation of an array.
        Returns:
            str: Cleaned string of integer list.
        """
        # Split the string by commas if they exist
        if ',' in arr_str:
            base_arr_str = arr_str.replace("[", "").replace("]", "").replace(" ", "")
            base_arr_list = base_arr_str.split(",")
        else:
            base_arr_str = arr_str.replace("[", "").replace("]", "")
            base_arr_list = base_arr_str.split(" ")

        # Keep only digits and return stringified list
        final_arr_list = [int(x) for x in base_arr_list if x.isdigit()]
        return str(final_arr_list)

    """ - -----------------------------  Materials_dev data manipulations - ----------------------------------------- """

    def unique_names_list(self, files_paths):
        """
        Extracts unique material base names from a list of file paths,
        removing patch numbers or suffixes at the end of the name.
        Parameters:
            files_paths (list of str): Full paths to files.
        Returns:
            list of str: Unique base names.
        """
        unique_names = []
        for path in files_paths:
            name = path.split('/')[-1].split('.')[-2]
            name_list = name.split('_')[:-1]
            new_name = '_'.join(name_list)
            if new_name not in unique_names:
                unique_names.append(new_name)

        return unique_names

    def full_yaml_creator_asd(self, ref_data_list, full_name, class_raw_wl = None):
        """
        Creates a complete YAML dictionary from raw reflectance data and metadata.
        Handles normalization, interpolation, color conversion, and populates all expected fields.
        Parameters:
            ref_data_list (list): Reflectance values (raw or interpolated).
            full_name (str): Material label name.
            class_raw_wl (np.ndarray or None): Optional raw wavelengths for interpolation.
        Returns:
            dict: Fully structured YAML material dictionary.
        """
        # Initialize with default keys
        yaml_dict = dict.fromkeys(self.config['functions']['general_params']['relevant_norm_keys'])
        yaml_dict['Measurement_Type'] = 'ASD FieldSpec4 Hi-Res'
        # Case: no custom wavelength input
        if class_raw_wl == None:
            yaml_dict['Raw_wave'] = self.raw_materials_wl.tolist()
            ref_data_list_after_min_check = self.minus_to_min_values_array_converter(ref_data_list)
            ref_data_list_after_max_check = self.over_one_to_max_values_array_converter(ref_data_list_after_min_check)
            yaml_dict['Raw_ref'] = ref_data_list_after_max_check.tolist()
        # Case: with custom wavelength input
        elif class_raw_wl != None:
            normed_raw_ref = self.linear_interpolation(self.raw_materials_wl, class_raw_wl, ref_data_list)
            normed_raw_ref_after_zero_check = self.minus_to_min_values_array_converter(normed_raw_ref)
            normed_raw_ref_after_max_check = self.over_one_to_max_values_array_converter(normed_raw_ref_after_zero_check)
            yaml_dict['Raw_ref'] = normed_raw_ref_after_max_check.tolist()
            yaml_dict['Raw_wave'] = self.raw_materials_wl.tolist()

        # Interpolate reflectance into visible wavelengths
        inter_vis_ref = self.linear_interpolation(self.inter_vis_wl, yaml_dict['Raw_wave'], yaml_dict['Raw_ref'])
        inter_vis_ref_after_zero_check = self.minus_to_min_values_array_converter(inter_vis_ref)
        inter_vis_ref_after_one_check = self.over_one_to_max_values_array_converter(inter_vis_ref_after_zero_check)
        yaml_dict['Vis_LinInter_wl'] = self.inter_vis_wl.tolist()
        yaml_dict['Vis_LinInter_ref'] = inter_vis_ref_after_one_check.tolist()
        # Compute spectral power
        args = [self.d65, inter_vis_ref_after_one_check, self.inter_vis_wl]
        material_spectral_power_list = self.spectral_power_converter(*args)
        yaml_dict['Material_spectral_power_list'] = np.array(material_spectral_power_list).tolist()
        # usually x and y ranges from 0-1 and z can be more then 1 (on D65 illumination)
        # Compute tristimulus components
        x_inter_label = self.over_one_to_max_values_array_converter(self.minus_to_min_values_array_converter(self.linear_interpolation(self.inter_vis_wl, self.tamiulus_raw_wl, self.x_label)))
        y_inter_label = self.over_one_to_max_values_array_converter(self.minus_to_min_values_array_converter(self.linear_interpolation(self.inter_vis_wl, self.tamiulus_raw_wl, self.y_label)))
        z_inter_label = self.minus_to_min_values_array_converter(self.linear_interpolation(self.inter_vis_wl, self.tamiulus_raw_wl, self.z_label))
        # Compute tristimulus values and convert to RGB
        args_trim = [self.inter_vis_wl, material_spectral_power_list, x_inter_label, y_inter_label, z_inter_label]
        x, y, z = self.trimulus_values_converter(*args_trim)
        yaml_dict['Material_trimulus_values_list'] = {'x': np.array(x).tolist(), 'y': np.array(y).tolist(),
                                                      'z': np.array(z).tolist()}
        rgb = self.xyz_to_rgb(yaml_dict['Material_trimulus_values_list'])
        yaml_dict['RGB'] = np.array(self.max_rgb_value_checker(rgb)).tolist()
        yaml_dict['Label_Name'] = full_name

        return yaml_dict

    def process_raw_material(self, file_path):
        """
        Reads and parses a raw ASD material `.txt` file, extracting reflectance data and cleaning it.
        Parameters:
            file_path (str): Full path to the raw material text file.
        Returns:
            tuple: (cleaned_material_name, cleaned_reflectance_array)
        """
        with open(file_path, 'r') as file:
            lines = file.readlines()
        # Extract and clean the name from the first line
        first_line = lines[0]
        name_part = first_line.split(':')[1].strip().split(' ')[0]
        name = ''.join(filter(str.isalpha, name_part))
        # Extract numerical data from the rest of the lines
        data = []
        for line in lines[2:]:
            try:
                value = float(line.strip())
                data.append(value)
            except ValueError:
                pass   # Skip non-numeric lines

        # Interpolate invalid values (e.g., negative or >1) with surrounding valid values
        data_array = np.array(data)
        interpolate_mask = np.logical_or(data_array < 0, data_array > 1)
        interpolated_values = np.interp(np.arange(len(data_array)),
                                        np.arange(len(data_array))[~interpolate_mask],
                                        data_array[~interpolate_mask])

        data_array[interpolate_mask] = interpolated_values[interpolate_mask]

        return name, data_array


    def process_raw_material_spec_lib(self, file_path):
        """
        Loads a YAML file of material reflectance and corrects invalid values.
        Parameters:
            file_path (str): Path to YAML file.
        Returns:
            tuple: (label_name, cleaned_ref_array, raw_wavelengths)
        """
        data = self.yaml_loader(file_path)
        name = data['Label_Name']
        # Clean up reflectance values (e.g., remove <0 or >1)
        data_array = np.array(data['Raw_ref'])
        interpolate_mask = np.logical_or(data_array < 0, data_array > 1)
        interpolated_values = np.interp(np.arange(len(data_array)),
                                        np.arange(len(data_array))[~interpolate_mask],
                                        data_array[~interpolate_mask])

        data_array[interpolate_mask] = interpolated_values[interpolate_mask]

        return name, data_array, data['Raw_wave']

    def raw_asd_to_yaml(self):
        """
        Converts raw ASD reflectance files into standardized YAML material files.
        For each class folder, it:
            - Processes `.txt` raw files
            - Converts to normalized and interpolated reflectance values
            - Generates visual plots
            - Saves results in YAML and PNG formats
        """
        # Resolve input/output paths from config
        materials_main_folder = os.path.join(self.path['main_pipeline_path'], self.config['functions']['raw_asd_to_yaml']['materials_main_folder'])
        class_folder_list = self.config['functions']['raw_asd_to_yaml']['class_folders_list']

        for class_material in tqdm(class_folder_list):
            class_input_path = os.path.join(materials_main_folder, class_material)
            class_output_path = os.path.join(self.path['results'], 'raw_asd_to_se_yaml', class_material)
            class_test_path = os.path.join(self.path['test'], 'raw_asd_to_se_yaml', class_material)
            out_paths_list = [class_output_path, class_test_path]
            for path in out_paths_list:
                self.folder_checker(path)
                
            for file in os.listdir(class_input_path):
                full_material_path = os.path.join(class_input_path, file)
                full_name, ref_data_list  = self.process_raw_material(full_material_path)
                material_dict = self.full_yaml_creator_asd(ref_data_list, full_name)

                # Save as YAML and plot reflectance as PNG
                full_output_path = os.path.join(class_output_path, file.replace('.txt', '.yml'))
                full_test_path = os.path.join(class_test_path, file.replace('.txt', '.png'))
                plt.plot(material_dict['Raw_wave'], material_dict['Raw_ref'] , color= 'yellow')
                plt.ylim(0,1)

                plt.savefig(full_test_path)
                plt.close()

                self.yaml_dumper(material_dict, full_output_path)

    def raw_spectral_lib_to_yaml(self):
        """
        Converts a curated spectral library in YAML format to standardized materials format.
        Handles interpolation, trimming, RGB conversion, and visualization.
        """
        materials_main_folder = os.path.join(self.path['main_pipeline_path'],
                                             self.config['functions']['raw_spectral_lib_to_yaml']['materials_main_folder'])
        class_folder_list = self.config['functions']['raw_spectral_lib_to_yaml']['class_folders_list']

        for class_material in tqdm(class_folder_list):
            class_input_path = os.path.join(materials_main_folder, class_material)
            class_output_path = os.path.join(self.path['results'], 'raw_spectral_lib_to_yaml', class_material)
            class_test_path = os.path.join(self.path['test'], 'raw_spectral_lib_to_yaml', class_material)
            out_paths_list = [class_output_path, class_test_path]
            for path in out_paths_list:
                self.folder_checker(path)

            for file in os.listdir(class_input_path):
                full_material_path = os.path.join(class_input_path, file)
                full_name, ref_data_list, raw_wl = self.process_raw_material_spec_lib(full_material_path)
                material_dict = self.full_yaml_creator_asd(ref_data_list, full_name, raw_wl)

                full_output_path = os.path.join(class_output_path, file.replace('.yml', '.yml'))
                full_test_path = os.path.join(class_test_path, file.replace('.yml', '.png'))
                plt.plot(material_dict['Raw_wave'], material_dict['Raw_ref'], color='yellow')
                plt.ylim(0, 1)

                plt.savefig(full_test_path)
                plt.close()

                self.yaml_dumper(material_dict, full_output_path)


    def classes_adder(self):
        """
        Appends semantic segmentation labels and RGB mask values to each material YAML file.
        Useful for creating class-aware labeled material entries.
        """
        # Input configuration
        materials_main_folder = os.path.join(self.path['main_pipeline_path'],
                                             self.config['functions']['classes_adder']['materials_main_folder'])
        class_folder_list = self.config['functions']['classes_adder']['class_folders_list']
        seg_colors_dict = self.config['functions']['classes_adder']['segs_dict']

        for class_material in tqdm(class_folder_list):
            class_input_path = os.path.join(materials_main_folder, class_material)
            class_output_path = os.path.join(self.path['results'], 'classes_adder', class_material)
            class_test_path = os.path.join(self.path['test'], 'classes_adder', class_material)
            out_paths_list = [class_output_path, class_test_path]
            for path in out_paths_list:
                self.folder_checker(path)

            for file in os.listdir(class_input_path):
                full_material_path = os.path.join(class_input_path, file)
                # Load YAML and update with class-specific info
                mat_data = self.yaml_loader(full_material_path)
                mat_data['Label_Type'] = class_material
                mat_data['Label_Class'] = class_material
                mat_data['RGB_seg_mask'] = seg_colors_dict[class_material]

                full_output_path = os.path.join(class_output_path, file)

                self.yaml_dumper(mat_data, full_output_path)

    def generate_materials_df(self):
        """
        Generates a pandas DataFrame with all available materials, their classes,
        and file paths. Useful for dataset indexing and filtering.
        Outputs the result as a CSV file.
        """
        materials_json = {'Material_Label':[],  'Material_Class':[], 'Material_Path':[]}
        materials_main_folder = os.path.join(self.path['main_pipeline_path'],
                                             self.config['functions']['generate_materials_df']['materials_main_folder'])
        materials_df_out_folder = os.path.join(self.path['results'], 'generate_materials_df')
        self.folder_checker(materials_df_out_folder)
        materials_df_out_full_path = os.path.join(materials_df_out_folder, 'materials_df.csv')
        class_folder_list = self.config['functions']['generate_materials_df']['class_folders_list']
        # Collect metadata from each YAML file
        for class_material in tqdm(class_folder_list):
            class_input_path = os.path.join(materials_main_folder, class_material)
            for file in os.listdir(class_input_path):
                full_material_path = os.path.join(class_input_path, file)
                mat_data = self.yaml_loader(full_material_path)
                materials_json['Material_Label'].append(mat_data['Label_Name'])
                materials_json['Material_Class'].append(mat_data['Label_Class'])
                materials_json['Material_Path'].append(full_material_path)


        materials_df = pd.DataFrame(materials_json)
        materials_df.to_csv(materials_df_out_full_path)





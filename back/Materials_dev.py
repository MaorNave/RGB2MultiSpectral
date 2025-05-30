import ast
import copy
# import ray
import json
import random
import shutil

import specdal
import re
from scipy.ndimage import zoom
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, make_scorer
from sklearn.model_selection import GridSearchCV
# from osgeo import gdal
import multiprocessing
import pickle
import string
from scipy.interpolate import CubicSpline
import os
import cv2
import yaml
from yaml.loader import SafeLoader
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import os
import pandas as pd
from scipy import interpolate
# import seaborn as sns
import pprint
import uuid
import spectral
from colormath.color_objects import sRGBColor, XYZColor
from colormath.color_conversions import convert_color
from skimage import io
from collections import Counter
import multiprocessing as mp
from tqdm import tqdm
import time


class Materials_dev():

    def __init__(self):
        self.cie_data_df = pd.read_csv('CIE_2006_5nm_2deg.csv')
        with open('config_Materials_dev.yaml') as f:
            data = yaml.load(f, Loader=SafeLoader)
        f.close()
        self.config = data

        self.inter_vis_wl = np.arange(400,700,1)
        self.raw_materials_wl = np.arange(350,2500)
        self.d65 = self.config['functions']['materials_to_rgb']['d65'][1:]
        self.tamiulus_raw_wl = np.arange(400, 700, 5)
        self.x_label = self.cie_data_df['X'][:-1]
        self.y_label = self.cie_data_df['Y'][:-1]
        self.z_label = self.cie_data_df['Z'][:-1]

        self.path = {'main_pipeline_path': r"C:\Maor Nanikashvili\thesis_pipeline\Pipeline",
            'results': r"C:\Maor Nanikashvili\thesis_pipeline\Pipeline\results",
                  'test': r"C:\Maor Nanikashvili\thesis_pipeline\Pipeline\test"}

    """ - -----------------------------  Static methods - ----------------------------------------- """

    @staticmethod
    def json_dumper(path, data):
        with open(path, "w") as outfile:
            json.dump(data, outfile, indent= "")
        outfile.close()

    @staticmethod
    def pickle_dumper(path, data):
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        f.close()

    @staticmethod
    def pickle_loader(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        f.close()
        return data

    @staticmethod
    def json_loader(path):
        with open(path, 'r') as f:
            data = json.load(f)
        f.close()
        return data

    @staticmethod
    def yaml_dumper(data, path):
        with open(path, 'w') as outfile:
            yaml.dump(data, outfile, sort_keys=False, default_flow_style=None)

    @staticmethod
    def yaml_loader(path):
        with open(path) as f:
            data = yaml.load(f, Loader=SafeLoader)
        f.close()
        return data



    @staticmethod
    ##lin interpolation function##
    def linear_interpolation(wl, raw_wl, raw_ref):
        f = interpolate.interp1d(raw_wl, raw_ref, fill_value="extrapolate")
        inter_ref = f(wl)
        return inter_ref



    @staticmethod
    def minus_to_min_values_array_converter(arr):
        min_arr_value_gt_0 = min(number for number in arr if number > 0)
        arr_copy = arr.copy()
        for i in range(len(arr)):
            if arr[i]<=0:
                arr_copy[i] = min_arr_value_gt_0
        return arr_copy

    @staticmethod
    def over_one_to_max_values_array_converter(arr):
        max_arr_value_lt_1 = max(number for number in arr if number < 1)
        arr_copy = arr.copy()
        for i in range(len(arr)):
            if arr[i]>=1:
                arr_copy[i] = max_arr_value_lt_1
        return arr_copy



    @staticmethod
    def dict_wl_to_ref(data_dict):
        data_dict_copy = data_dict.copy()
        for key_material in data_dict.keys():
            raw_ref_list = []
            for wl_key in data_dict[key_material].keys():
                raw_ref = data_dict[key_material][wl_key]
                raw_ref_list.append(raw_ref)
            data_dict_copy.pop(key_material)
            data_dict_copy[key_material] = raw_ref_list

        return data_dict_copy

    @staticmethod
    def folder_checker(path):
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Folders created for the following path : {path}" )
        else:
            print(f" Folder path are already exist for : {path}")



    """ - -----------------------------  Materials_dev data manipulations inner functions - ----------------------------------------- """



    def max_rgb_value_checker(self, rgb_list):
        rgb_list_copy = rgb_list.copy()
        for i in range(len(rgb_list)):
            if rgb_list[i]>255:
                rgb_list_copy[i] = 255
            if rgb_list[i]<0:
                rgb_list_copy[i] = 0
        return rgb_list_copy

    def xyz_to_rgb(self, dict):
        xyz_vector = np.array([dict['x'],dict['y'],dict['z']])
        if xyz_vector.max() > 100:
            xyz_vector = xyz_vector / xyz_vector.max()
        else:
            xyz_vector = xyz_vector / 100

        xyz_vector_color = XYZColor(xyz_vector[0],xyz_vector[1],xyz_vector[2], illuminant='d65')
        simpale_rgb = convert_color(xyz_vector_color, sRGBColor)
        rgb_con = np.array(simpale_rgb.get_value_tuple())
        final_rgb_value = (rgb_con*255).round()

        return final_rgb_value

    def trimulus_values_converter(self, *args_trim):
        x_list, y_list, z_list = [], [], []
        vis_wl = args_trim[0]
        if max(args_trim[1])>100:
            material_power = np.array(args_trim[1]) / max(args_trim[1])
        else:
            material_power = np.array(args_trim[1]) / 100
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
        x_final = sum(x_list)
        y_final = sum(y_list)
        z_final = sum(z_list)
        return x_final, y_final, z_final

    def spectral_power_converter(self, *args):
        s_power_list = []
        vis_wl = args[-1]
        d_65 = args[0]
        ref = args[-2]
        for i in range(len(vis_wl)):
            s_power = (d_65[i] * ref[i])
            s_power_list.append(s_power)
        return s_power_list



    def array_to_string_defulter(self, arr_str):
        if ',' in arr_str:
            base_arr_str = arr_str.replace("[", "").replace("]", "").replace(" ", "")
            base_arr_list = base_arr_str.split(",")
        else:
            base_arr_str = arr_str.replace("[", "").replace("]", "")
            base_arr_list = base_arr_str.split(" ")

        final_arr_list = [int(x) for x in base_arr_list if x.isdigit()]
        return str(final_arr_list)




    """ - -----------------------------  Materials_dev data manipulations - ----------------------------------------- """

    def unique_names_list(self, files_paths):
        unique_names = []
        for path in files_paths:
            name = path.split('/')[-1].split('.')[-2]
            name_list = name.split('_')[:-1]
            new_name = '_'.join(name_list)
            if new_name not in unique_names:
                unique_names.append(new_name)

        return unique_names

    def full_yaml_creator_asd(self, ref_data_list, full_name, class_raw_wl = None):
        yaml_dict = dict.fromkeys(self.config['functions']['general_params']['relevant_norm_keys'])
        yaml_dict['Measurement_Type'] = 'ASD FieldSpec4 Hi-Res'

        if class_raw_wl == None:
            yaml_dict['Raw_wave'] = self.raw_materials_wl.tolist()
            ref_data_list_after_min_check = self.minus_to_min_values_array_converter(ref_data_list)
            ref_data_list_after_max_check = self.over_one_to_max_values_array_converter(ref_data_list_after_min_check)
            yaml_dict['Raw_ref'] = ref_data_list_after_max_check.tolist()
        elif class_raw_wl != None:
            normed_raw_ref = self.linear_interpolation(self.raw_materials_wl, class_raw_wl, ref_data_list)
            normed_raw_ref_after_zero_check = self.minus_to_min_values_array_converter(normed_raw_ref)
            normed_raw_ref_after_max_check = self.over_one_to_max_values_array_converter(normed_raw_ref_after_zero_check)
            yaml_dict['Raw_ref'] = normed_raw_ref_after_max_check.tolist()
            yaml_dict['Raw_wave'] = self.raw_materials_wl.tolist()



        inter_vis_ref = self.linear_interpolation(self.inter_vis_wl, yaml_dict['Raw_wave'], yaml_dict['Raw_ref'])
        inter_vis_ref_after_zero_check = self.minus_to_min_values_array_converter(inter_vis_ref)
        inter_vis_ref_after_one_check = self.over_one_to_max_values_array_converter(inter_vis_ref_after_zero_check)
        yaml_dict['Vis_LinInter_wl'] = self.inter_vis_wl.tolist()
        yaml_dict['Vis_LinInter_ref'] = inter_vis_ref_after_one_check.tolist()
        args = [self.d65, inter_vis_ref_after_one_check, self.inter_vis_wl]
        material_spectral_power_list = self.spectral_power_converter(*args)
        yaml_dict['Material_spectral_power_list'] = np.array(material_spectral_power_list).tolist()
        # usually x and y ranges from 0-1 and z can be more then 1 (on D65 illumination)
        x_inter_label = self.over_one_to_max_values_array_converter(self.minus_to_min_values_array_converter(self.linear_interpolation(self.inter_vis_wl, self.tamiulus_raw_wl, self.x_label)))
        y_inter_label = self.over_one_to_max_values_array_converter(self.minus_to_min_values_array_converter(self.linear_interpolation(self.inter_vis_wl, self.tamiulus_raw_wl, self.y_label)))
        z_inter_label = self.minus_to_min_values_array_converter(self.linear_interpolation(self.inter_vis_wl, self.tamiulus_raw_wl, self.z_label))

        args_trim = [self.inter_vis_wl, material_spectral_power_list, x_inter_label, y_inter_label, z_inter_label]

        x, y, z = self.trimulus_values_converter(*args_trim)
        yaml_dict['Material_trimulus_values_list'] = {'x': np.array(x).tolist(), 'y': np.array(y).tolist(),
                                                      'z': np.array(z).tolist()}
        rgb = self.xyz_to_rgb(yaml_dict['Material_trimulus_values_list'])
        yaml_dict['RGB'] = np.array(self.max_rgb_value_checker(rgb)).tolist()
        yaml_dict['Label_Name'] = full_name

        return yaml_dict

    def process_raw_material(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        # Extract and clean the name from the first line
        first_line = lines[0]
        name_part = first_line.split(':')[1].strip().split(' ')[0]  # Get the part after the colon and strip whitespace
        name = ''.join(filter(str.isalpha, name_part))  # Keep only alphabetical characters
        # Extract numerical data from the rest of the lines
        data = []
        for line in lines[2:]:
            try:
                value = float(line.strip())
                data.append(value)
            except ValueError:
                pass  # Ignore lines that cannot be converted to float

        #linear interpolation of numpy to extract problomatic values and change them
        data_array = np.array(data)
        interpolate_mask = np.logical_or(data_array < 0, data_array > 1)
        interpolated_values = np.interp(np.arange(len(data_array)),
                                        np.arange(len(data_array))[~interpolate_mask],
                                        data_array[~interpolate_mask])

        data_array[interpolate_mask] = interpolated_values[interpolate_mask]

        return name, data_array


    def process_raw_material_spec_lib(self, file_path):
        data = self.yaml_loader(file_path)
        name = data['Label_Name']

        #linear interpolation of numpy to extract problomatic values and change them
        data_array = np.array(data['Raw_ref'])
        interpolate_mask = np.logical_or(data_array < 0, data_array > 1)
        interpolated_values = np.interp(np.arange(len(data_array)),
                                        np.arange(len(data_array))[~interpolate_mask],
                                        data_array[~interpolate_mask])

        data_array[interpolate_mask] = interpolated_values[interpolate_mask]

        return name, data_array, data['Raw_wave']

    def raw_asd_to_yaml(self):
        """function that takes raw asd data and turn it to yaml material"""
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


                full_output_path = os.path.join(class_output_path, file.replace('.txt', '.yml'))
                full_test_path = os.path.join(class_test_path, file.replace('.txt', '.png'))
                plt.plot(material_dict['Raw_wave'], material_dict['Raw_ref'] , color= 'yellow')
                plt.ylim(0,1)

                plt.savefig(full_test_path)
                plt.close()

                self.yaml_dumper(material_dict, full_output_path)

    def raw_spectral_lib_to_yaml(self):
        """function that takes raw spectral lib data and turn it to yaml material"""

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
        """function that adds each yaml the seg values and classes names"""
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
                mat_data = self.yaml_loader(full_material_path)
                mat_data['Label_Type'] = class_material
                mat_data['Label_Class'] = class_material
                mat_data['RGB_seg_mask'] = seg_colors_dict[class_material]

                full_output_path = os.path.join(class_output_path, file)

                self.yaml_dumper(mat_data, full_output_path)

    def generate_materials_df(self):
        """"function that takes all materials data and generate a materials df that have as columns the lbel name of material and its class"""
        materials_json = {'Material_Label':[],  'Material_Class':[], 'Material_Path':[]}
        materials_main_folder = os.path.join(self.path['main_pipeline_path'],
                                             self.config['functions']['generate_materials_df']['materials_main_folder'])
        materials_df_out_folder = os.path.join(self.path['results'], 'generate_materials_df')
        self.folder_checker(materials_df_out_folder)
        materials_df_out_full_path = os.path.join(materials_df_out_folder, 'materials_df.csv')
        class_folder_list = self.config['functions']['generate_materials_df']['class_folders_list']

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





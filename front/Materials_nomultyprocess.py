import copy
import multiprocessing
import os.path
import ast
import json
import pickle
import random
import shutil
from tqdm import tqdm
from scipy.spatial import cKDTree
import skimage
import string
from scipy.interpolate import CubicSpline
import os
import cv2
# import ray
import yaml
from yaml.loader import SafeLoader
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import os
import pandas as pd
#from Frontend.utils.Interpolate_NonNeg import interpolate_non_neg
from scipy import interpolate
import seaborn as sns
import pprint
import uuid
import spectral
# from colormath.color_objects import sRGBColor, XYZColor
# from colormath.color_conversions import convert_color
# from skimage import io
from collections import Counter
import time
import multiprocessing as mp



class Materials():

    def __init__(self):
        with open('config_PRTORAD.yaml') as f:
            data = yaml.load(f, Loader=SafeLoader)
        f.close()
        self.config = data

        # self.path = {'main_pipeline_path': r"C:\Maor Nanikashvili\thesis_pipeline\Pipeline",
        #     'results': r"C:\Maor Nanikashvili\thesis_pipeline\Pipeline\results",
        #           'test': r"C:\Maor Nanikashvili\thesis_pipeline\Pipeline\test",
        #              'input': r"C:\Maor Nanikashvili\thesis_pipeline\Pipeline\input"}


        self.path = {'main_pipeline_path': r"D:\Maor_Computer_Backup\Maor Nanikashvili\thesis_pipeline_march_last_ver\Pipeline",
            'results': r"D:\Maor_Computer_Backup\Maor Nanikashvili\thesis_pipeline_march_last_ver\Pipeline\results",
                  'test': r"D:\Maor_Computer_Backup\Maor Nanikashvili\thesis_pipeline_march_last_ver\Pipeline\test",
                     'input': r"D:\Maor_Computer_Backup\Maor Nanikashvili\thesis_pipeline_march_last_ver\Pipeline\input"}


    """ - -----------------------------  Static methods - ----------------------------------------- """

    @staticmethod
    def folder_checker(path):
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Folders created for the following path : {path}" )
        else:
            print(f" Folder path are already exist for : {path}")

    @staticmethod
    def yaml_dumper(data, path):
        with open(path, 'w') as outfile:
            yaml.dump(data, outfile, sort_keys=False, default_flow_style=None)

    @staticmethod
    def json_dumper(path, data):
        with open(path, 'w') as f:
            json.dump(data, f)
        f.close()

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


    """ - -----------------------------  cubes_jenerator- ----------------------------------------- """

    ##patches for strings lists of rgb values that came from simulator in bad forms##
    def string_list_to_int(self, seg_rgb):
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
        striped_rgb = rgb.strip()
        new_rgb = striped_rgb[1:-1]
        new_rgb = new_rgb.split(',')
        return [int(x) for x in new_rgb]

    @staticmethod
    def clu_rgb_to_materials_labels_rgb_swir_dict_raffle(clu_rgb, obj):
        """
        By base rgb in a specific seg, return the material vector reflectance in materials_labels_rgb_swir_dict
        :param clu_rgb: mean of rgb in clu rgb
        :param obj: obj name to search his scores in materials_labels_rgb_swir_dict
        : raffle score method - the material pick will be by a random chice of top 5 candidates of similarity to the pixel
        :return: 2150D vector reflectance by the chosen material to this clu_rgb
        """

        try:
            column_rgb_name = np.array2string(clu_rgb, separator=' ')
            max_score_candidates = (objects_clu_rgb_mat[obj][column_rgb_name]).sort_values(ascending=False).head(5)
            probabilities = max_score_candidates / max_score_candidates.sum()
            raffle_score = np.random.choice(max_score_candidates, p=probabilities)
            # for a state of no material to transplant
            if raffle_score == 0:
                return (np.zeros(2150))
            material_name = objects_clu_rgb_mat[obj][column_rgb_name].loc[
                objects_clu_rgb_mat[obj][column_rgb_name] == raffle_score].index
        except Exception as e:
            print(e)
            ##if cannot make throght this function that means that its already had to be found in the first try and except from above with nparray to string method##
            try:
                column_rgb_name = Materials.convert_list(column_rgb_name)
            except:
                return (np.zeros(2150))

            column_rgb_name = str(column_rgb_name)
            max_score_candidates = (objects_clu_rgb_mat[obj][column_rgb_name]).sort_values(ascending=False).head(5)
            probabilities = max_score_candidates / max_score_candidates.sum()
            raffle_score = np.random.choice(max_score_candidates, p=probabilities)
            ##if cannot find a value of a column in this pint that means it doesent show on the scores_mat data##
            # for a state of no material to transplant
            if raffle_score == 0:
                return (np.zeros(2150))
            material_name = objects_clu_rgb_mat[obj][column_rgb_name].loc[
                objects_clu_rgb_mat[obj][column_rgb_name] == raffle_score].index

        # return 0 - RGB value of material, 1- ref sig of 2150wls, 2- material name and class in capital letters, 3- materials class RGB value, 4 - max_score
        # if zero then return np.zeros in size 2150 and nothing else for masks becose masks will have a zero value which dieaince the no material signature
        return (materials_labels_rgb_swir_dict[material_name[0]][0],
                materials_labels_rgb_swir_dict[material_name[0]][1],
                materials_labels_rgb_swir_dict[material_name[0]][2],
                materials_labels_rgb_swir_dict[material_name[0]][3],
                raffle_score)

    @staticmethod
    def clu_rgb_to_materials_labels_rgb_swir_dict(clu_rgb, obj):
        """
        By base rgb in a specific seg, return the material vector reflectance in materials_labels_rgb_swir_dict
        :param clu_rgb: mean of rgb in clu rgb
        :param obj: obj name to search his scores in materials_labels_rgb_swir_dict
        :return: 2150 vector reflectance by the chosen material to this clu_rgb
        """

        try:
            column_rgb_name = np.array2string(clu_rgb, separator=' ')
            max_score = (objects_clu_rgb_mat[obj][column_rgb_name]).max()
            # for a state of no material to transplant
            if max_score == 0:
                return (np.zeros(2150))
            material_name = objects_clu_rgb_mat[obj][column_rgb_name].loc[
                objects_clu_rgb_mat[obj][column_rgb_name] == max_score].index
        except Exception as e:
            print(e)
            ##if cannot make throght this function that means that its already had to be found in the first try and except from above with nparray to string method##
            try:
                column_rgb_name = Materials.convert_list(column_rgb_name)
            except:
                return (np.zeros(2150))
            column_rgb_name = str(column_rgb_name)
            ##if cannot find a value of a column in this pint that means it doesent show on the scores_mat data (only on 10 first frames)##
            max_score = objects_clu_rgb_mat[obj][column_rgb_name].max()
            # for a state of no material to transplant
            if max_score == 0:
                return (np.zeros(2150))
            material_name = objects_clu_rgb_mat[obj][column_rgb_name].loc[
                objects_clu_rgb_mat[obj][column_rgb_name] == max_score].index

        # return 0 - RGB value of material, 1- ref sig of 2150wls, 2- material name and class in capital letters, 3- materials class RGB value, 4 - max_score
        # if zero then return np.zeros in size 2150 and nothing else for masks becose masks will have a zero value which dieaince the no material signature
        return (materials_labels_rgb_swir_dict[material_name[0]][0],
                materials_labels_rgb_swir_dict[material_name[0]][1],
                materials_labels_rgb_swir_dict[material_name[0]][2],
                materials_labels_rgb_swir_dict[material_name[0]][3],
                max_score)

    @staticmethod
    def indices_in_segment_object(flat_seg_img, seg):
        """
        Get flat_seg_img and seg_rgb and return all indices that seg_rgb in flat_seg_img
        :param flat_seg_img: mask segment image in flatten format.
        :param seg: segment rgb to search
        :return obj_seg_indices: all indices that seg_rgb in flat_seg_img
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
        Generate one cube with globals: materials_labels_rgb_swir_dict, objects_clu_rgb_mat, objects_seg_rgb_dict,
        pick_score_method, output_folder and with frame segment and rgb image
        :param seg_frame_path: path to mask segment frame
        :param rgb_frame_path: path to unreal rgn image
        :param ind: cube index name
        Save npy generated Cube by Maor transplanting materials algorithm
        """

        # print(override_dict_rev)
        seg_frame_path=item_tup[0]
        rgb_frame_path=item_tup[1]
        frame_number = rgb_frame_path.split('\\')[-1].split('.')[0]
        swir_frame = np.zeros((224, 224, 2150))
        rgb_mask_frame = np.zeros((224, 224, 3)) # rgb presentation of materials mask
        materials_type_seg_mask = np.zeros((224, 224, 3)) # by RGB value of material type from ES.
        materials_heat_map_mask = np.zeros((224, 224, 1)) # by max_score that has been implemented
        frame_materials_dict = dict()
        frame_seg = cv2.imread(seg_frame_path,cv2.IMREAD_UNCHANGED)
        flat_seg_img = frame_seg.reshape(-1)
        frame_clu = cv2.imread(rgb_frame_path, cv2.IMREAD_UNCHANGED)
        flat_clu_frame = frame_clu.reshape(-1, frame_clu.shape[2])
        # iterating on all relevant obj for the frame seg##
        # remove segments not in frame and return segment indices
        objects_seg_rgb_dict_indices = {obj: Materials.indices_in_segment_object(flat_seg_img, seg_rgb) \
                                        for obj, seg_rgb in objects_seg_rgb_dict_data.items()}
        objects_seg_rgb_dict_relevant = {obj: obj_seg_indices for obj, obj_seg_indices in
                                         objects_seg_rgb_dict_indices.items()
                                         if len(obj_seg_indices) > 0}

        for obj, obj_seg_indices in objects_seg_rgb_dict_relevant.items():
            clu_rgb_on_seg_list = flat_clu_frame[obj_seg_indices]
            # iterating on every unique base rgb in a specific seg
            unique_clu_rgb = np.unique(clu_rgb_on_seg_list, axis=0)


            material_swir_transplant_dict_unique = {np.array2string(clu_rgb, separator=' '): \
                                                        Materials.clu_rgb_to_materials_labels_rgb_swir_dict(clu_rgb, obj)
                                                    for clu_rgb in unique_clu_rgb}



            frame_materials_dict[obj] = material_swir_transplant_dict_unique
            material_swir_transplant_list = [
                material_swir_transplant_dict_unique[np.array2string(clu_rgb, separator=' ')]
                for clu_rgb in clu_rgb_on_seg_list]
            ##converting the 1 d indexes to 2 d for frame transplant##
            for j, index in enumerate(obj_seg_indices):
                row, col = np.unravel_index(index, frame_clu.shape[:2])

                swir_frame[row][col] = material_swir_transplant_list[j][1]
                if len(material_swir_transplant_list[j]) == 5:

                    rgb_mask_frame[row][col] = material_swir_transplant_list[j][0]
                    materials_type_seg_mask[row][col] = material_swir_transplant_list[j][3]
                    materials_heat_map_mask[row][col] = material_swir_transplant_list[j][4]

                else: # for max_score == 0 materials transplant
                    continue


        # finig and completeing zero values in cube (if neccecery)
        zero_coords = np.argwhere(np.all(swir_frame == np.zeros(2150), axis=-1))
        non_zero_coords = np.argwhere(np.all(swir_frame != np.zeros(2150), axis=-1))
        # Create a KDTree using the coordinates of non-zero pixels
        tree = cKDTree(non_zero_coords)
        # Num of closest pixel to consider.
        num_closest = 10
        # Iterate over each zero-valued pixel
        for zero_coord in zero_coords:
            # Find the indices and distances of the num_closest non-zero pixels closest to the zero-valued pixel
            _, indices = tree.query(zero_coord, k=num_closest)

            # random choice of the relevant new material signature index in the cube
            rel_index = np.random.choice(indices)
            row_index, col_index = np.unravel_index(rel_index, swir_frame.shape[:-1])

            # get the value of the spectral signature in the cube
            # notice to complete all data for all masks by the same indices
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

        np.save(os.path.join(output_folder, new_name), swir_frame.astype(np.float16))
        cv2.imwrite(os.path.join(rgb_mask_output_folder, rgb_mask_frame_new_name), cv2.cvtColor(rgb_mask_frame.astype('uint8'), cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(material_type_mask_folder, materials_type_seg_mask_new_name), materials_type_seg_mask.astype(int))
        cv2.imwrite(os.path.join(heatmap_mask_folder, materials_heat_map_mask_new_name), materials_heat_map_mask)
        spectral.save_rgb(os.path.join(psedo_rgb_mask_output_folder, psedo_rgb_mask_frame_new_name)
            , swir_frame)
        Materials.pickle_dumper(os.path.join(test_output_mat_dict_folder, new_name.replace('.npy', '.pickle')), frame_materials_dict)



    def worker_init_cubes_gen(self, global_args_list):
        # set global tables score from config
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
        Run cube_generator in parallel over relevant frames and score materials tabels
        """

        test_folders_list = self.config['cubes_generator']['test_folders_list']

        for test_folder in tqdm(test_folders_list):
            ##loading relevent files and data from config file (files locations --> deploy)##
            # main_seg_path = os.path.join(self.path['results'], 'vgg19_clasiffier', test_folder, 'masks')
            main_seg_path = "C:\Maor Nanikashvili\\thesis_pipeline\Pipeline\\results\\vgg19_clasiffier\\feb25_test_session\masks"
            # main_clu_rgb_path = os.path.join(self.path['results'], 'heuristic_rgb_material_vector_v1', test_folder, 'clustered_rgb')
            main_clu_rgb_path = "D:\Maor_Computer_Backup\Maor Nanikashvili\\thesis_pipeline_march_last_ver\Pipeline\\results\heuristic_rgb_material_vector_v1\\feb25_test_session_hvi_net_enhanced_full_materials_data_final_ver\clustered_rgb"

            # materials_labels_rgb_swir_dict = self.yaml_loader(os.path.join(self.path['results'], 'heuristic_rgb_material_vector_v1' ,test_folder, self.config['cubes_generator']['materials_rgb_swir']))
            materials_labels_rgb_swir_dict = self.yaml_loader("D:\Maor_Computer_Backup\Maor Nanikashvili\\thesis_pipeline_march_last_ver\Pipeline\\results\heuristic_rgb_material_vector_v1\\feb25_test_session_hvi_net_enhanced_full_materials_data_final_ver\material_rgb_dict.yml")
            # objects_clu_rgb_mat = self.pickle_loader(os.path.join(self.path['results'], 'heuristic_rgb_material_vector_v1', test_folder, self.config['cubes_generator']['objects_base_rgb_scores_mat']))
            objects_clu_rgb_mat = self.pickle_loader("D:\Maor_Computer_Backup\Maor Nanikashvili\\thesis_pipeline_march_last_ver\Pipeline\\results\heuristic_rgb_material_vector_v1\\feb25_test_session_hvi_net_enhanced_full_materials_data_final_ver\scores_mat_objects_dict_after_heur.pickle")

            objects_seg_rgb_dict_data = self.yaml_loader(os.path.join(self.path['input'], 'seg_mapping',
                                                          self.config['cubes_generator']['obj_seg_rgb']))['labels_numbers_dict']




            ##relevant output and test paths
            test_output_mat_dict_folder = r"\\?\\" + os.path.join(self.path['test'], 'cubes_generator', test_folder, 'Max_score\\cubes_materials_dict')
            output_folder = r"\\?\\" + os.path.join(self.path['results'], 'cubes_generator',test_folder, 'Max_score\\cubes')
            rgb_mask_output_folder = r"\\?\\" + os.path.join(self.path['results'], 'cubes_generator',test_folder, 'Max_score\\masks\\rgb_masks')
            heatmap_mask_folder = r"\\?\\" + os.path.join(self.path['results'], 'cubes_generator',test_folder, 'Max_score\\masks\\heatmap_mask')
            psedo_rgb_mask_output_folder = r"\\?\\" + os.path.join(self.path['results'], 'cubes_generator',test_folder, 'Max_score\\masks\\psedo_rgb_masks')
            material_type_mask_folder = r"\\?\\" + os.path.join(self.path['results'], 'cubes_generator',test_folder, 'Max_score\\masks\\material_type_mask')


            output_path_list = [test_output_mat_dict_folder ,output_folder,  rgb_mask_output_folder\
                , material_type_mask_folder, heatmap_mask_folder, psedo_rgb_mask_output_folder]

            for path in output_path_list:
                self.folder_checker(path)


            ##sorting scene data from 0-max##
            seg_frames_list = np.sort(os.listdir(main_seg_path))
            clu_frames_list = np.sort(os.listdir(main_clu_rgb_path))

            # prepare to Parallel frames
            items = [(os.path.join(main_seg_path, seg), os.path.join(main_clu_rgb_path, rgb))  \
                     for seg, rgb in \
                     zip(seg_frames_list, clu_frames_list)]

            global_args_list = [materials_labels_rgb_swir_dict, objects_clu_rgb_mat, objects_seg_rgb_dict_data,
                                output_folder, test_output_mat_dict_folder, rgb_mask_output_folder, psedo_rgb_mask_output_folder,
                                material_type_mask_folder, heatmap_mask_folder
                                ]

            print('start pooling')
            # with mp.Pool() as p:
                # p.starmap(Materials.cube_generator, items)
            p = mp.Pool(2, initializer=self.worker_init_cubes_gen, initargs=(global_args_list,))
            #
            results = p.map(Materials.cube_generator, [arg_tup for arg_tup in items])

            # Materials.cube_generator(items[0], materials_labels_rgb_swir_dict, objects_clu_rgb_mat, objects_seg_rgb_dict_data,
            #                     output_folder, test_output_mat_dict_folder, rgb_mask_output_folder, psedo_rgb_mask_output_folder,
            #                     material_type_mask_folder, heatmap_mask_folder)


    @staticmethod
    def parallel_raffle_cubes(item_tup):
        """
        Generate one cube with globals: materials_labels_rgb_swir_dict, objects_clu_rgb_mat, objects_seg_rgb_dict,
        pick_score_method, output_folder and with frame segment and rgb image
        :param seg_frame_path: path to mask segment frame
        :param rgb_frame_path: path to unreal rgn image
        :param ind: cube index name
        Save npy generated Cube by Maor transplanting materials algorithm
        """

        # print(override_dict_rev)
        seg_frame_path=item_tup[0]
        rgb_frame_path=item_tup[1]
        frame_number = rgb_frame_path.split('\\')[-1].split('.')[0]
        swir_frame = np.zeros((224, 224, 2150))
        rgb_mask_frame = np.zeros((224, 224, 3)) # rgb presentation of materials mask
        materials_type_seg_mask = np.zeros((224, 224, 3)) # by RGB value of material type from ES.
        materials_heat_map_mask = np.zeros((224, 224, 1)) # by max_score that has been implemented
        frame_materials_dict = dict()
        frame_seg = cv2.imread(seg_frame_path,cv2.IMREAD_UNCHANGED)
        flat_seg_img = frame_seg.reshape(-1)
        frame_clu = cv2.imread(rgb_frame_path, cv2.IMREAD_UNCHANGED)
        flat_clu_frame = frame_clu.reshape(-1, frame_clu.shape[2])
        # iterating on all relevant obj for the frame seg##
        # remove segments not in frame and return segment indices
        objects_seg_rgb_dict_indices = {obj: Materials.indices_in_segment_object(flat_seg_img, seg_rgb) \
                                        for obj, seg_rgb in objects_seg_rgb_dict_data.items()}
        objects_seg_rgb_dict_relevant = {obj: obj_seg_indices for obj, obj_seg_indices in
                                         objects_seg_rgb_dict_indices.items()
                                         if len(obj_seg_indices) > 0}

        for obj, obj_seg_indices in objects_seg_rgb_dict_relevant.items():
            clu_rgb_on_seg_list = flat_clu_frame[obj_seg_indices]
            # iterating on every unique base rgb in a specific seg
            unique_clu_rgb = np.unique(clu_rgb_on_seg_list, axis=0)


            material_swir_transplant_dict_unique = {np.array2string(clu_rgb, separator=' '): \
                                                        Materials.clu_rgb_to_materials_labels_rgb_swir_dict_raffle(clu_rgb, obj)
                                                    for clu_rgb in unique_clu_rgb}



            frame_materials_dict[obj] = material_swir_transplant_dict_unique
            material_swir_transplant_list = [
                material_swir_transplant_dict_unique[np.array2string(clu_rgb, separator=' ')]
                for clu_rgb in clu_rgb_on_seg_list]
            ##converting the 1 d indexes to 2 d for frame transplant##
            for j, index in enumerate(obj_seg_indices):
                row, col = np.unravel_index(index, frame_clu.shape[:2])

                swir_frame[row][col] = material_swir_transplant_list[j][1]
                if len(material_swir_transplant_list[j]) == 5:

                    rgb_mask_frame[row][col] = material_swir_transplant_list[j][0]
                    materials_type_seg_mask[row][col] = material_swir_transplant_list[j][3]
                    materials_heat_map_mask[row][col] = material_swir_transplant_list[j][4]

                else: # for max_score == 0 materials transplant
                    continue


        # finig and completeing zero values in cube (if neccecery)
        zero_coords = np.argwhere(np.all(swir_frame == np.zeros(2150), axis=-1))
        non_zero_coords = np.argwhere(np.all(swir_frame != np.zeros(2150), axis=-1))
        # Create a KDTree using the coordinates of non-zero pixels
        tree = cKDTree(non_zero_coords)
        # Num of closest pixel to consider.
        num_closest = 10
        # Iterate over each zero-valued pixel
        for zero_coord in zero_coords:
            # Find the indices and distances of the num_closest non-zero pixels closest to the zero-valued pixel
            _, indices = tree.query(zero_coord, k=num_closest)

            # random choice of the relevant new material signature index in the cube
            rel_index = np.random.choice(indices)
            row_index, col_index = np.unravel_index(rel_index, swir_frame.shape[:-1])

            # get the value of the spectral signature in the cube
            # notice to complete all data for all masks by the same indices
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

        np.save(os.path.join(output_folder, new_name), swir_frame.astype(np.float16))
        cv2.imwrite(os.path.join(rgb_mask_output_folder, rgb_mask_frame_new_name), cv2.cvtColor(rgb_mask_frame.astype('uint8'), cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(material_type_mask_folder, materials_type_seg_mask_new_name), materials_type_seg_mask.astype(int))
        cv2.imwrite(os.path.join(heatmap_mask_folder, materials_heat_map_mask_new_name), materials_heat_map_mask)
        spectral.save_rgb(os.path.join(psedo_rgb_mask_output_folder, psedo_rgb_mask_frame_new_name)
            , swir_frame)
        Materials.pickle_dumper(os.path.join(test_output_mat_dict_folder, new_name.replace('.npy', '.pickle')), frame_materials_dict)

    # @staticmethod
    # def c(data_tup):
    #     # C.R. 7/May/23: split to procedural static functions
    #     i = data_tup[0]
    #     frame = data_tup[1]
    #     frame_number = frame.split('_')[0]
    #     # swir_frame = np.zeros((1024, 1280, 12)) # C.R. 7/May/23: params to config
    #     label_seg_frame = np.zeros((1024, 1280, 1)) # with segments of UNREAL
    #     rgb_mask_frame = np.zeros((1024, 1280, 3)) # rgb presentation of materials mask
    #     instance_seg_frame = np.zeros((1024, 1280, 1)) # by instance materials from ES
    #     materials_seg_mask = np.zeros((1024, 1280, 1)) # by instances of materials detection_instance from ES --> same material with diffrenet instances gets the same label
    #     materials_type_seg_mask = np.zeros((1024, 1280, 3)) # by RGB value of material type from ES.
    #     materials_heat_map_mask = np.zeros((1024, 1280, 1)) # by max_score that has been implemented
    #     materials_distance_heat_map_mask = np.zeros((1024, 1280, 1)) # by distance_rgb_matrix distance from RGB on pixel
    #     class_balance_vector = np.zeros((4096)) # by detection_instance material
    #     seg_frame_path = os.path.join(main_seg_path, frame)
    #     seg_frame = cv2.imread(seg_frame_path)
    #     flat_seg_frame = seg_frame.reshape(-1, seg_frame.shape[2])
    #     max_cube_path = os.path.join(main_max_cubes_path, max_cubes_list[i])
    #     max_cube = np.load(max_cube_path)
    #     raffle_cube = np.zeros((1024, 1280, 12))
    #     # scores_mask = np.zeros((1080, 1920, 1)) # C.R. 7/May/23: params to config
    #     # materials_score_mask_dict = dict()
    #     objects_seg_rgb_dict_indices = {obj: Materials.indices_in_segment_object(flat_seg_frame, seg_rgb) \
    #                                     for obj, seg_rgb in objects_seg_rgb_dict.items()}
    #     objects_seg_rgb_dict_relevant = {obj: obj_seg_indices for obj, obj_seg_indices in
    #                                      objects_seg_rgb_dict_indices.items()
    #                                      if len(obj_seg_indices) > 0}
    #
    #     frame_materials_dict = dict()
    #     ##raffle new material##
    #     # find the unique material swir values that are implemented on a specific segment
    #     # put all data in a dict that key is seg and val is unique material swir values on seg
    #     for seg, indices in tqdm(objects_seg_rgb_dict_relevant.items()):
    #         # values of ovveride table will stay the same while raffle
    #         if seg in override_dict_relavent.keys():
    #             continue
    #         # elif seg == 'Soldier_Gloves_Fabric_03' or seg == 'M1Abrams_M240Guided_Metal' or seg == 'M1Abrams_M2MachineGun_Metal':
    #         #     print(seg)
    #         # continue
    #
    #         reshaped_cube = np.reshape(max_cube, (-1, 12)) # C.R. 7/May/23: get channels from shape rather than hard coded
    #         unique_materials_swir_list = np.unique(reshaped_cube[indices], axis=0)
    #         unique_materials_swir_list_no_zeros = unique_materials_swir_list[
    #             np.any(unique_materials_swir_list != 0, axis=1)]
    #
    #         if len(unique_materials_swir_list_no_zeros) == 0:
    #             continue
    #         # search names of unique materials that were transplanted to seg.
    #         # put all data on a dict with seg name as key and material name as values
    #         material_name_swir_dict = dict()
    #         for material_name, material_data in materials_labels_rgb_swir_dict.items():
    #             if np.array(material_data[1]) in np.array(unique_materials_swir_list_no_zeros):
    #                 material_seg_indices = np.where(np.all(reshaped_cube == np.array(material_data[1]), axis=1))[0]
    #                 material_seg_indices_relevant = list(set(indices).intersection(set(material_seg_indices)))
    #                 material_name_swir_dict[material_name] = (material_data[1], material_seg_indices_relevant)
    #
    #
    #             else:
    #                 continue
    #
    #         # upload scores matrix df for each segment
    #         seg_data_df = objects_clu_rgb_mat[seg]
    #         seg_data_df.fillna(0, inplace=True)
    #         distance_data_df = objects_distance_base_rgb_mat[seg]
    #         distance_data_df.fillna(0, inplace=True)
    #
    #         # drop from scores df the materials that were allready implemented by max score method
    #         for name, swir_tup in tqdm(material_name_swir_dict.items()):
    #             # swir_val = swir_tup[0]
    #             material_swir_ind = swir_tup[1]
    #             # raffle_seg_data_df = copy.deepcopy(seg_data_df)
    #             # raffle_seg_data_distance_df = copy.deepcopy(distance_data_df)
    #             # raffle_seg_data_df = raffle_seg_data_df.drop(raffle_seg_data_df[raffle_seg_data_df.index == name].index)
    #             # raffle_seg_data_distance_df = raffle_seg_data_distance_df.drop(raffle_seg_data_distance_df[raffle_seg_data_distance_df.index == name].index)
    #             # take all df values and raffle with treshold of > 0 and > mean
    #             raffle_seg_data_df = seg_data_df.loc[seg_data_df.index != name, :]
    #             raffle_values = np.array(raffle_seg_data_df.values).flatten()
    #             raffle_values = raffle_values[raffle_values > 0]
    #             raffle_mean = np.mean(raffle_values)
    #             relevent_raffle_values = raffle_values[raffle_values > raffle_mean]
    #             try:
    #                 picked_score = random.choices(relevent_raffle_values, weights=relevent_raffle_values, k=1)[0]
    #             except:
    #                 print('score_problem')
    #             # Finding where the picked score is in the df, and retrieving its material
    #             # gets index , col values as list --> takes the first pair in list
    #             # the row, column are outputed as ints and not the values of index and col names
    #             new_material_val, new_material_column_val = np.where(raffle_seg_data_df.values == picked_score)
    #             new_material_index_val = new_material_val[0]
    #             new_material_column_index_val = new_material_column_val[0]
    #             # new material name from index
    #             new_material = raffle_seg_data_df.index[new_material_index_val]
    #             new_material_column = raffle_seg_data_df.columns[new_material_column_index_val]
    #             # material_raffled_mask = (raffle_seg_data_df == picked_score)
    #             # true_cells = material_raffled_mask.where(material_raffled_mask == True)
    #             #capture name and column of material - index + column
    #             # new_material_data = true_cells.stack().index[0]
    #             # new_material = new_material_data[0]
    #             # new_material_column = new_material_data[1]
    #             material_rgb_distance_val = distance_data_df.loc[raffle_seg_data_df.index[new_material_index_val], raffle_seg_data_df.columns[new_material_column_index_val]]
    #             material_raffle_swir = materials_labels_rgb_swir_dict[new_material][1]
    #
    #
    #             # for index in material_swir_ind:
    #             row, col = np.unravel_index(material_swir_ind, seg_frame.shape[:2])
    #             # if max_cube[row][col].tolist() == swir_val:
    #             materials_distance_heat_map_mask[row, col] = material_rgb_distance_val
    #             raffle_cube[row, col, :] = material_raffle_swir
    #             materials_heat_map_mask[row, col] = picked_score
    #             label_seg_frame[row, col] = seg_dict_relevant[seg][0]
    #             materials_seg_mask[row, col] = materials_labels_rgb_swir_dict[new_material][4]
    #             instance_seg_frame[row, col]= materials_labels_rgb_swir_dict[new_material][3]
    #             rgb_mask_frame[row, col, :] = materials_labels_rgb_swir_dict[new_material][0]
    #             class_balance_vector[materials_labels_rgb_swir_dict[new_material][4]] += 1
    #             materials_type_seg_mask[row, col, :] = materials_labels_rgb_swir_dict[new_material][5]
    #
    #
    #             if new_material_column not in frame_materials_dict.keys():
    #                 frame_materials_dict[new_material_column] = [materials_labels_rgb_swir_dict[material_name]]
    #             else:
    #                 frame_materials_dict[new_material_column].append(materials_labels_rgb_swir_dict[material_name])
    #
    #     class_balance_vector = class_balance_vector / 1310720
    #
    #
    #     raffle_new_name = frame_number+ 'raffle_cube.npy'
    #     label_seg_new_name = frame_number + '_label_seg_mask.npy'
    #     rgb_mask_frame_new_name = frame_number + '_rgb_mask.npy'
    #     instance_seg_frame_new_name = frame_number + '_instance_seg_frame.npy'
    #     class_balance_vector_new_name = frame_number + '_class_vector.npy'
    #     materials_type_seg_mask_new_name = frame_number + '_material_type_mask.npy'
    #     materials_seg_mask_new_name = frame_number + '_material_mask.npy'
    #     materials_heat_map_mask_new_name = frame_number + '_heatmap_mask.npy'
    #     materials_rgb_distance_heat_map_mask_new_name = frame_number + '_rgb_distance_heatmap_mask.npy'
    #     np.save(os.path.join(output_folder, raffle_new_name), raffle_cube)
    #     np.save(os.path.join(seg_mask_output_folder, label_seg_new_name), label_seg_frame)
    #     np.save(os.path.join(rgb_mask_output_folder, rgb_mask_frame_new_name), rgb_mask_frame.astype(int))
    #     np.save(os.path.join(class_balance_vector_folder, class_balance_vector_new_name), class_balance_vector)
    #     np.save(os.path.join(instance_seg_frame_output_folder, instance_seg_frame_new_name), instance_seg_frame)
    #     np.save(os.path.join(material_type_mask_folder, materials_type_seg_mask_new_name), materials_type_seg_mask)
    #     np.save(os.path.join(materials_mask_folder, materials_seg_mask_new_name), materials_seg_mask)
    #     np.save(os.path.join(heatmap_mask_folder, materials_heat_map_mask_new_name), materials_heat_map_mask)
    #     np.save(os.path.join(distance_heatmap_map_folder, materials_rgb_distance_heat_map_mask_new_name), materials_distance_heat_map_mask)

    def worker_init_cubes_gen_raffle(self, global_args_list):
        # set global tables score from config
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
            cubes generator function to create cubes on with raffle of materials on base of max.score cubes data ,by segments and unique implemented materials
            operation is raffled new materials on every frame
            outputted data:
            1. raffled cubes
            2. scores masks for every cube
            3. score json -> a dict with all materials names that were implamentaed as keys and row, col indices for every time
            that the materials was picked and with which score as value
            4. heat score map for every cube
            """
            test_folders_list = self.config['cubes_generator']['test_folders_list']

            for test_folder in tqdm(test_folders_list):
                ##loading relevent files and data from config file (files locations --> deploy)##
                main_seg_path = os.path.join("C:\Maor Nanikashvili\\thesis_pipeline\Pipeline\\results", 'vgg19_clasiffier', test_folder, 'masks')
                main_clu_rgb_path = os.path.join("C:\Maor Nanikashvili\\thesis_pipeline\Pipeline\\results", 'heuristic_rgb_material_vector_v1', test_folder, 'clustered_rgb')
                materials_labels_rgb_swir_dict = self.yaml_loader(os.path.join("C:\Maor Nanikashvili\\thesis_pipeline\Pipeline\\results", 'heuristic_rgb_material_vector_v1' ,test_folder, self.config['cubes_generator']['materials_rgb_swir']))
                objects_clu_rgb_mat = self.pickle_loader(os.path.join("C:\Maor Nanikashvili\\thesis_pipeline\Pipeline\\results", 'heuristic_rgb_material_vector_v1', test_folder, self.config['cubes_generator']['objects_base_rgb_scores_mat']))
                objects_seg_rgb_dict_data = self.yaml_loader(os.path.join("C:\Maor Nanikashvili\\thesis_pipeline\Pipeline\\input", 'seg_mapping',
                                                                          self.config['cubes_generator'][
                                                                              'obj_seg_rgb']))['labels_numbers_dict']

                ##relevant output and test paths
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

                output_path_list = [test_output_mat_dict_folder, output_folder, rgb_mask_output_folder \
                    , material_type_mask_folder, heatmap_mask_folder, psedo_rgb_mask_output_folder]

                for path in output_path_list:
                    self.folder_checker(path)

                ##sorting scene data from 0-max##
                seg_frames_list = np.sort(os.listdir(main_seg_path))
                clu_frames_list = np.sort(os.listdir(main_clu_rgb_path))

                # prepare to Parallel frames
                items = [(os.path.join(main_seg_path, seg), os.path.join(main_clu_rgb_path, rgb)) \
                         for seg, rgb in \
                         zip(seg_frames_list, clu_frames_list)]

                global_args_list = [materials_labels_rgb_swir_dict, objects_clu_rgb_mat, objects_seg_rgb_dict_data,
                                    output_folder, test_output_mat_dict_folder, rgb_mask_output_folder,
                                    psedo_rgb_mask_output_folder,
                                    material_type_mask_folder, heatmap_mask_folder
                                    ]

                print('start pooling')
                # with mp.Pool() as p:
                #     p.starmap(Materials.cube_generator, items)
                p = mp.Pool(2, initializer=self.worker_init_cubes_gen_raffle, initargs=(global_args_list,))
                #
                results = p.map(Materials.parallel_raffle_cubes, [arg_tup for arg_tup in items])

                print('start pooling')
                start = time.time()

                # self.parallel_raffle_cubes((0, seg_frames_list[0]))

                end = time.time()
                time_in_sec = end - start
                time_in_min = time_in_sec / 60
                time_in_hours = time_in_min / 60
                print(f"time for all frames --> {time_in_hours}")
                print('done pooling')


    """ -------------------------------------------------  PR_TO_RAD CUBES TEST DATA -------------------------------------------------------"""

    def transplanted_materials_counter_test(self):
        """function that gets all test pickles files from Materials frontend generator
        and check the amount of unique materials that are transplanted to the Max_score method"""

        test_folders_list = self.config['cubes_generator']['test_folders_list']
        materials_folder_version = self.config['cubes_generator']['materials_version_folder']

        for test_folder in tqdm(test_folders_list):
            #copy frames from unreal to results folder for test case

            if self.config['transplanted_materials_counter_test']['test_set']:
                shutil.copytree(os.path.join(self.config['transplanted_materials_counter_test']['main_frames_path'], test_folder, 'Frames'),
                                os.path.join(self.path['results'], materials_folder_version, test_folder, 'Frames'), dirs_exist_ok=True)

            input_test_folder = os.path.join(self.path['test'], materials_folder_version, test_folder, 'Max_score', 'cubes_materials_dict')
            output_test_folder = os.path.join(self.path['test'], materials_folder_version, test_folder, 'Max_score', 'materials_scene_counter')
            self.folder_checker(output_test_folder)



            input_test_files_list = os.listdir(input_test_folder)
            full_data_dict = dict()
            for file in tqdm(input_test_files_list):
                full_path = os.path.join(input_test_folder, file)
                file_data = self.pickle_loader(full_path)
                for key, value in file_data.items():
                    if key not in full_data_dict:
                        full_data_dict[key] = []
                    else:
                        pass
                    for base_rgb, data_tup in value.items():
                        if data_tup[1] != 25 and key.lower() != 'sky':
                            if data_tup[-4] in full_data_dict[key]:
                                continue
                            else:
                                full_data_dict[key].append(data_tup[-4])
                        else:
                            continue

            materials_detection_labels_list = [val for val in full_data_dict.values()]
            materials_detection_labels_list_flat = [val for vals in materials_detection_labels_list for val in vals]

            self.pickle_dumper(os.path.join(output_test_folder, 'seg_materials_data_dict.pickle'), full_data_dict)

            matplotlib.use('Agg')
            ## two figures + check the y and x axis limits
            plt.figure(figsize=(18,8))
            plt.bar(full_data_dict.keys(), [len(full_data_dict[key]) for key in full_data_dict.keys()])
            plt.tick_params(axis='x', labelrotation=90)
            plt.subplots_adjust(top=0.95, bottom=0.55)
            plt.title('Count of different materials transplanted in seg')
            plt.savefig(os.path.join(output_test_folder, 'seg_materials_data_counter.png'))
            plt.close()

            plt.hist(materials_detection_labels_list_flat, bins = 4000)
            plt.axvline(x=100, color='red')
            # plt.tick_params(axis='x', labelrotation=90)
            # plt.subplots_adjust(top=0.95, bottom=0.55)
            plt.title('Count of materials used in the scene')
            plt.xlim(0, 5000)
            plt.savefig(os.path.join(output_test_folder, 'materials_unique_counter.png'))
            plt.close()


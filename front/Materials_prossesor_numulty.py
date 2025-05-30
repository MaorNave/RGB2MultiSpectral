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
import torch.nn as nn
from torchvision.models import vgg19  # Assuming you're using VGG19 for segmentation
from torchvision import transforms
from PIL import Image  # Assuming you're using PIL for image loading
import matplotlib.pyplot as plt
import random
import shutil
from tqdm import tqdm
import string
from scipy.interpolate import CubicSpline
import os
import cv2
import yaml
from yaml.loader import SafeLoader
from matplotlib import pyplot as plt
from front.materials_heuristics_keys_sycronizore import keys_sycronizore
import matplotlib
import numpy as np
import os
import pandas as pd
from scipy import interpolate
import seaborn as sns
import pprint
import uuid
import spectral
from skimage import io
from collections import Counter
import time
from sklearn.cluster import KMeans
from back.NN_dev import SegmentationModel

class Materials_prossesor():



    def __init__(self):
        with open('config_MATERIALS.yaml') as f:
            data = yaml.load(f, Loader=SafeLoader)
        f.close()
        self.config = data

        self.path = {'main_pipeline_path': r"C:\Maor Nanikashvili\thesis_pipeline\Pipeline",
            'results': r"C:\Maor Nanikashvili\thesis_pipeline\Pipeline\results",
                  'test': r"C:\Maor Nanikashvili\thesis_pipeline\Pipeline\test",
                     'input': r"C:\Maor Nanikashvili\thesis_pipeline\Pipeline\input"}

    """ - -----------------------------  Static methods - ----------------------------------------- """

    @staticmethod
    def folder_checker(path):
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Folders created for the following path : {path}" )
        else:
            print(f" Folder path are already exist for : {path}")



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



    @staticmethod
    def parallel_get_all_rgb(rgb_image_path):
        """
        The function calculates the unique base RGBs of
        images and allows for parallel calculation.
        :param img_path: Path to image.
        :return unique_brgbs: List of unique base RGBs in the frame.
        """

        mask_image_path = rgb_image_path.replace('images', 'masks').replace('.jpg', '.png')

        rgb_image = cv2.imread(rgb_image_path, cv2.IMREAD_UNCHANGED)
        mask_image = cv2.imread(mask_image_path, cv2.IMREAD_UNCHANGED)

        unique_segments = np.unique(mask_image)
        segment_colors = {}

        for segment in unique_segments:
            mask_indices = np.where(mask_image == segment)
            segment_rgb_values = rgb_image[mask_indices]  # Extract corresponding RGB values
            segment_colors[segment] = np.unique(segment_rgb_values, axis=0)  # Keep unique colors

        return segment_colors



    @staticmethod
    def parallel_clu_rgb(frame_tup):
        frame_num = frame_tup[0]
        frame = frame_tup[1]
        mean_dict = dict()
        seg_unique_clu_dict = dict()
        full_clu_path = os.path.join(main_clu_path_global, frame)
        frame_clu_rgb = cv2.imread(full_clu_path, cv2.IMREAD_UNCHANGED)
        try:
            flat_clu_rgb_img = frame_clu_rgb.reshape(-1, frame_clu_rgb.shape[2])
        except:
            print(frame_num)
            print(full_clu_path)
        full_rgb_path = os.path.join(main_rgb_path_global, rgb_frame_list_global[frame_num])
        full_seg_path = os.path.join(main_seg_path_global, seg_rgb_list_global[frame_num])
        frame_seg = cv2.imread(full_seg_path, cv2.IMREAD_UNCHANGED)
        flat_seg_img = frame_seg.reshape(-1)
        frame_rgb = cv2.imread(full_rgb_path, cv2.IMREAD_UNCHANGED)
        flat_rgb_img = frame_rgb.reshape(-1, frame_rgb.shape[2])
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
        The function bins images using the kmeans model.
        :param tup: A tuple containing three fields:
        0: Image path
        1: kmeans model
        2: Location for saving
        """
        img_loc = tup[0]
        mask_loc = img_loc.replace('images', 'masks').replace('.jpg', '.png')
        save_loc = tup[1]
        segments_dict = tup[2]
        img = cv2.imread(img_loc, cv2.IMREAD_UNCHANGED)
        mask_img = cv2.imread(mask_loc, cv2.IMREAD_UNCHANGED)
        binned_img = np.zeros_like(img)

        k=30
        kmeans_model = KMeans(n_clusters=k)


        for segment_label, unique_rgb_values in segments_dict.items():
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


            # Assign each pixel the centroid color of its cluster
            new_rgb_values = np.array([cluster_centers[label] for label in resulting_img_labels], dtype=np.uint8)
            binned_img[segment_pixels] = new_rgb_values

        cv2.imwrite(save_loc, binned_img)

        # flat_img = np.reshape(img, (-1, 3))
        # resulting_img_labels = kmeans_model.predict(flat_img)
        # cluster_centers = kmeans_model.cluster_centers_
        #
        # # Translating from cluster id to centroid
        # resulting_rgb_list = [cluster_centers[i] for i in resulting_img_labels]
        # binned_img = np.reshape(resulting_rgb_list, img.shape)





    @staticmethod
    def array_to_string_defulter_rgb_v_parallel(arr_str):
        if ',' in arr_str:
            base_arr_str = arr_str.replace("[", "").replace("]", "").replace(" ", "")
            base_arr_list = base_arr_str.split(",")
        else:
            base_arr_str = arr_str.replace("[", "").replace("]", "")
            base_arr_list = base_arr_str.split(" ")

        final_arr_list = [int(x) for x in base_arr_list if x.isdigit()]
        return str(final_arr_list)


    @staticmethod
    # Define a function to compute the scores for a single object and distances of RGB values - as 2 dfs
    def compute_object_scores_mat(args_list):
        obj = args_list[0]
        clu_rgb = args_list[1]
        if len(clu_rgb) == 0:
            obj_scores_df = pd.DataFrame(columns=['no_base'], index=materials_labels_rgb_swir_dict_global_mat.keys())
            obj_scores_df.loc[:, 'no_base'] = 0
        else:
            obj_scores_df = pd.DataFrame(columns=[str(rgb) for rgb in clu_rgb],
                                         index=materials_labels_rgb_swir_dict_global_mat.keys())

            for column in hot_vector_df_global.columns:
                if column == 'Material_Label':
                    continue
                col_first_label = column.split(';')[0]
                if col_first_label != obj:  # for a case that there are no relevent segmentes on the frames that have made the scores_mat_objects_dict .
                    # means that there are less segments on the specific scene or folder that we got on segments dict that we inputed from UNREAL engine,
                    continue
                if len(obj_scores_df) != len(hot_vector_df_global[column].values):
                    # for a case that there are some materials that dont have
                    # any vis RGB values (on scores df, but was assigned on hot vactor df)
                    common_indices = obj_scores_df.index.intersection(hot_vector_df_global.index)
                    hot_vector_df_filtered = hot_vector_df_global.loc[common_indices]
                    col_values = hot_vector_df_filtered[column].values
                    mask = col_values == 0
                    obj_scores_df[mask] = 0
                else:
                    col_values = hot_vector_df_global[column].values
                    mask = col_values == 0
                    obj_scores_df[mask] = 0

            mat_rgb_dict = {}
            for index in obj_scores_df.index:
                # only on nulls if a row is zero then the materials is exluded for this seg
                if (obj_scores_df.loc[index] == 0).all():
                    continue
                else:
                    mat_rgb = materials_labels_rgb_swir_dict_global_mat[index][0]
                    mat_rgb_dict[index] = np.array(mat_rgb)

            if len(mat_rgb_dict) != 0:  # checks if there are any materials candidets that can be transplanted to any base rgb --> if all materials cannot be implemented to specific
                # segment so the dict will be empty, if its not emty then will be candidets for implenetation and max_dist will not be 0
                for column in obj_scores_df.columns:
                    new_column = Materials_prossesor.array_to_string_defulter_rgb_v_parallel(
                        column)  # get new column (base_rgb) value as accaptable string
                    if new_column not in unique_clu_rgb_and_mean_rgb_dict_global_mat.keys():
                        continue

                    rgb = unique_clu_rgb_and_mean_rgb_dict_global_mat[new_column]
                    # rgb = bgr[::-1]
                    # takes only values that are relevant for this specific seg (from mat_rgb_dict,  its keeps on the convention and places of the values and materials values on the df)
                    mse_list = [mean_squared_error(rgb, mat_rgb) for mat_rgb in mat_rgb_dict.values()]
                    # assigning the relevant distance values only on null cells (0 cells are heuristiccly not supported for the relevent obj/seg)
                    obj_scores_df.loc[obj_scores_df[column] != 0, column] = mse_list
                    # getting column max value and compute score for all base_rgb material candidates, normilazie the data on 0-1 scale
                    max_dist = obj_scores_df[column].max()
                    obj_scores_df[column] = obj_scores_df[column] / (
                                max_dist + 1)  # calc max_dist +1 so the value of max_dist that is now a candidate will not fall out as zero
                    # assigning the 1- normilized value to each cell on column (only for data which is not 0 , means can be huristilclly assigned)
                    # obj_scores_df.loc[obj_scores_df[column]!=0, column] = 1 - obj_scores_df.loc[(obj_scores_df!=0).all(axis=1), column]
                    obj_scores_df.loc[obj_scores_df[column] != 0, column] = 1 - obj_scores_df.loc[
                        obj_scores_df[column] != 0, column]
            else:
                pass

        obj_scores_df.fillna(0, inplace=True)


        return (obj, obj_scores_df)



    """ - -----------------------------  heuristics vector - ----------------------------------------- """

    def heuristic_object_material_vector(self): #continue from here
        """function that creates heuristics table for implementations"""
        objects_list_path = self.path['input']
        input_folder = self.config['functions']['heuristic_object_material_vector']['input_folder']
        input_file = self.config['functions']['heuristic_object_material_vector']['input_file']
        output_folder = self.config['functions']['heuristic_object_material_vector']['output_folder']
        objects_list_full_path = os.path.join(objects_list_path, input_folder, input_file)
        output_path = os.path.join(self.path['results'], output_folder)
        self.folder_checker(output_path)
        output_file_name = self.config['functions']['heuristic_object_material_vector']['output_file']
        output_full_path = os.path.join(output_path, output_file_name)
        data = self.yaml_loader(objects_list_full_path)['labels_numbers_dict']
        materials_df = pd.read_csv(os.path.join(self.path['results'], 'generate_materials_df' , 'materials_df.csv'))
        materials_df_copy = copy.deepcopy(materials_df)


        new_keys_list = keys_sycronizore(data)
        hot_vec_df = pd.DataFrame(columns=new_keys_list, index=materials_df_copy['Material_Label'])
        for index in tqdm(materials_df_copy.index):
            for key in new_keys_list:
                ##add to if that is considering the string rgb color of each material
                if materials_df_copy['Material_Class'].loc[index].lower() in key.lower(): # and material_dict['RGB_Name'].lower() in key.lower(): --> not relevant for new heuristics
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
        hot_vec_df['Material_Label'] = material_label_final_list
        hot_vec_df.set_index('Material_Label', inplace=True)
        hot_vec_df.to_csv(output_full_path)




    """ - -----------------------------  scores matrices process - ----------------------------------------- """



    def worker_init_rgb_scores_dict(self, global_args_list):
        global materials_labels_rgb_swir_dict_global_mat
        global unique_clu_rgb_and_mean_rgb_dict_global_mat
        global hot_vector_df_global


        materials_labels_rgb_swir_dict_global_mat = global_args_list[0]
        unique_clu_rgb_and_mean_rgb_dict_global_mat = global_args_list[1]
        hot_vector_df_global = global_args_list[2]


    ##create scores matrix before and after heuristics by toggles from config file##
    def rgb_scores_objects_dict(self, hot_vector_df, unique_clu_rgb_per_object_dict, materials_labels_rgb_swir_dict,
                                unique_clu_rgb_and_mean_rgb_dict, objects_seg_rgb_data):


        start = time.time()
        hot_vector_df.set_index(hot_vector_df.columns[0],
                                inplace=True)
        global_args_list = [copy.deepcopy(materials_labels_rgb_swir_dict), copy.deepcopy(unique_clu_rgb_and_mean_rgb_dict) , hot_vector_df]
        # # Define the number of worker processes to use
        num_workers = 6
        # # Create a pool of worker processes
        pool = mp.Pool(num_workers, initializer=self.worker_init_rgb_scores_dict, initargs=(global_args_list,))
        # # Define a list of arguments to pass to the compute_object_scores function
        args_list = [(obj, clu_rgb) for
                     obj, clu_rgb in unique_clu_rgb_per_object_dict.items()]

        print("start polling")
        # self.compute_object_scores_mat(args_list[0], materials_labels_rgb_swir_dict, unique_clu_rgb_and_mean_rgb_dict, hot_vector_df)

        results = list(pool.map(self.compute_object_scores_mat, [arg_tup for arg_tup in args_list]))

        # Close the pool of worker processes
        # pool.close()
        print("done pooling")
        end = time.time()
        time_in_sec = end - start
        time_in_min = time_in_sec / 60
        time_in_hours = time_in_min / 60
        print(f"time for all frames --> {time_in_hours}")

        # Combine the results into a dictionary
        scores_mat_objects_dict = {obj: obj_scores_df for obj, obj_scores_df in results}        ## addinf no base for segments that are not represented on input frames on specific scene
        for seg in objects_seg_rgb_data.keys():
            if seg not in scores_mat_objects_dict:
                obj_scores_df = pd.DataFrame(columns=['no_base'],
                                             index=materials_labels_rgb_swir_dict.keys())
                obj_scores_df.loc[:, 'no_base'] = 0
                scores_mat_objects_dict[seg] = obj_scores_df
            else:
                continue

        return scores_mat_objects_dict


    ##function that change base rgb values format to defult format##
    def array_to_string_defulter_rgb_v(self, arr_str):
        if ',' in arr_str:
            clu_arr_str = arr_str.replace("[", "").replace("]", "").replace(" ", "")
            clu_arr_list = clu_arr_str.split(",")
        else:
            clu_arr_str = arr_str.replace("[", "").replace("]", "")
            clu_arr_list = clu_arr_str.split(" ")

        final_arr_list = [int(x) for x in clu_arr_list if x.isdigit()]
        return str(final_arr_list)



    ##function that change clu rgb values format to defult format##
    def order_string_keys(self, unique_clu_rgb_and_mean_rgb_dict):
        new_dict = {}
        for key, value in unique_clu_rgb_and_mean_rgb_dict.items():
            new_key = self.array_to_string_defulter_rgb_v(key)
            new_dict[new_key] = value

        return new_dict



    def cluster_rgb(self, main_clu_rgb_path, output_folder, frames_list):
        """
        The function clusters a folder of  RGB images to 100 clusters using
        the kmeans algorithm.
        :param folder_loc: Full path to folder.
        """

        save_folder = output_folder  # Location to save binned images

        t = time.time()
        print('start polling on clustered rgb')
        # Parallel getting all unique RGB
        pool = mp.Pool(6)

        # self.parallel_get_all_rgb(os.path.join(main_base_rgb_path, frames_list[0]))
        with pool as p:
            all_frames_seg_data_dict = p.map(self.parallel_get_all_rgb,
                                       [os.path.join(main_clu_rgb_path, frame) for frame in frames_list])

        p.close()


        merged_dict = defaultdict(list)

        for d in all_frames_seg_data_dict:
            for key, value in d.items():
                merged_dict[key].append(value)  # Append values to the corresponding key


        final_segment_colors_dict =  {key: np.unique(np.vstack(val), axis=0) for key , val in dict(merged_dict).items()}

        # all_frames_array = np.concatenate(all_frames_unique, axis=0)
        # final_unique = np.unique(all_frames_array, axis=0)

        # # Fitting data k clusters
        # kmeans_model = KMeans(n_clusters=k)
        # cluster_labels = kmeans_model.fit(final_unique)

        # Parallel binning and saving images
        # self.bin_image(
        #     (os.path.join(main_base_bgr_path, base_frames_list[0]), cluster_labels, os.path.join(save_folder, base_frames_list[0])))
        pool = mp.Pool(6)
        results = list(pool.map(self.bin_image, [(os.path.join(main_clu_rgb_path, frame), os.path.join(save_folder, frame), final_segment_colors_dict) for frame in frames_list]))
        pool.close()
        elapsed = (time.time() - t) / 60
        print('Binning Images: ' + str(np.round(elapsed, 2)) + ' minutes')


    def worker_init_clu_to_mean_rgb(self, global_args_list):
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
        create mean RGB value per clu rgb in all frames as dict --> key clu_rgb, value--> Mean rgb on key
        create a unique CLU_rgb per seg dict--> key --. seg , value --> list of unique clu rgb on seg
        """

        # init global vars foe parallel computing
        # color_names_to_rgb = {objects_seg_rgb_data['color_names_dict'][key]: objects_seg_rgb_data['rgb_values'][key] for key in objects_seg_rgb_data['color_names_dict']}
        global_args_list = [main_clu_path, rgb_frame_list, main_rgb_path, main_seg_path, seg_rgb_list,
                            {self.array_to_string_defulter_rgb_v(str(value)): key for key, value in
                             objects_seg_rgb_data.items()}
                            ]
        # init dict for moving avarage for all frames.
        full_seg_unique_clu_bgr_dict = dict()
        final_seg_unique_clu_bgr_dict = dict()
        full_list_mean_clu_rgb_to_rgb_dict = dict()

        print('start pooling')
        start = time.time()

        # result = self.parallel_clu_rgb((0, clu_frame_list[0]))
        pool = mp.Pool(6, initializer=self.worker_init_clu_to_mean_rgb, initargs=(global_args_list,))
        with pool as p:
            results = p.map(self.parallel_clu_rgb, [(frame_num, frame) for frame_num, frame in enumerate(clu_frame_list)])

        print('done pooling')
        p.close()
        # Unifying all frames dicts
        for result_tup_dict in tqdm(results):
            seg_dict = result_tup_dict[1]
            mean_bgr_dict = result_tup_dict[0]

            # Unifying seg dicts
            for key, value in seg_dict.items():
                if key in full_seg_unique_clu_bgr_dict.keys():  # If already in dict, extend with new clu rgbs
                    full_seg_unique_clu_bgr_dict[key] = np.append(full_seg_unique_clu_bgr_dict[key], value, axis=0)
                else:  # If new segment, start with current dict clu rgbs.
                    full_seg_unique_clu_bgr_dict[key] = value

            # Unifying mean rgb per clu rgb dicts
            for str_bgr, data_tup in mean_bgr_dict.items():
                if str_bgr in full_list_mean_clu_rgb_to_rgb_dict.keys():  # If already in dict, update average rgb and number of pixels
                    cur_mean_bgr = data_tup[0]
                    cur_num_pixels = data_tup[1]
                    mean_bgr = full_list_mean_clu_rgb_to_rgb_dict[str_bgr][0]
                    num_pixels = full_list_mean_clu_rgb_to_rgb_dict[str_bgr][1]
                    new_mean_bgr = (num_pixels * mean_bgr + cur_num_pixels * cur_mean_bgr) / (
                                num_pixels + cur_num_pixels)
                    new_num_pixels = num_pixels + cur_num_pixels
                    new_tup = (new_mean_bgr, new_num_pixels)
                    full_list_mean_clu_rgb_to_rgb_dict[
                        str_bgr] = new_tup  # Tuples are immutable, so must create new tuple
                else:  # If new base rgb, this is the current average and number of pixels
                    full_list_mean_clu_rgb_to_rgb_dict[str_bgr] = data_tup

        full_list_mean_clu_rgb_to_rgb_dict = {key: val[0].tolist() for key, val in
                                               full_list_mean_clu_rgb_to_rgb_dict.items()}
        # Running on all segments and getting true unique clu rgb list
        for seg, val in full_seg_unique_clu_bgr_dict.items():
            final_seg_unique_clu_bgr_dict[seg] = np.unique(val, axis=0)

        end = time.time()
        time_in_sec = end - start
        time_in_min = time_in_sec / 60
        time_in_hours = time_in_min / 60
        print(f"time for all frames --> {time_in_hours}")
        print('converting results to one dict --> full_list_mean_clu_rgb_to_rgb_dict')

        return full_list_mean_clu_rgb_to_rgb_dict, final_seg_unique_clu_bgr_dict



    def materials_rgb_dict_creator(self, materials_df):
        """
        creates a dict with material labels names as keys and 2 d array with place 0 --> rgb values from fit to material + place 1 --> swir 12 channels data##
        new version creates :
        array[2]--> Material_label+Material_type+Material_experiment
        array[3]--> Material instance
        array[4]-->detection_instance
        array[5]-->detection_instance
        """

        materials_label_rgb_dict = dict()
        count_dict = dict()
        for index in tqdm(materials_df.index):
            path = materials_df['Material_Path'][index]
            # there are several materials that amit have uploaded to the index and
            # they wont have any path to yaml file - for stattistics.
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
            if material_label not in materials_label_rgb_dict.keys():
                materials_label_rgb_dict[material_label] = [material_rgb, spectro_ref, ''.join(
                    [material_label.capitalize(), material_type]), materials_type_rgb_seg]
                count_dict[material_label] = 0
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

    """
    MAIN
    main heuristics scores matrix by rgb values in simulation frames
    materials scores vector with values between 0-1 --> ver 1 -->  rgb image no shadow comparison
    types of outputs:
    materials_labels_rgb_swir_dict
    objects_seg_rgb_data --> dict of seg rgb for every scene
    unique_clu_bgr_list --> unique base rgb list for all sences
    unique_clu_rgb_and_mean_rgb_dict --> dict of base rgb as key and mean of rgb per every base rgb
    unique_clu_rgb_per_object_dict --> unique base rgb to segments on the frame
    scores_per_objects_mat before and after heuristics --> by toggels
    """

    def heuristic_rgb_material_vector_v1(self):

        folder_names_list = self.config['functions']['heuristic_rgb_material_vector_v1']['folder_names_list']
        # load mapping of segments classes
        json_name = self.config['functions']['heuristic_rgb_material_vector_v1'][
            'object_data_file_name']

        objects_seg_rgb_data = self.yaml_loader(os.path.join(self.path['input'], 'seg_mapping', json_name))['labels_numbers_dict']


        for folder_name in tqdm(folder_names_list):
            print(folder_name)

            main_rgb_path = os.path.join(self.path['results'], 'vgg19_clasiffier', folder_name,
                                                   'images')
            main_seg_path = os.path.join(self.path['results'], 'vgg19_clasiffier', folder_name, 'masks')

            materials_df = pd.read_csv(os.path.join(self.path['results'], 'generate_materials_df', 'materials_df.csv'))

            clustered_rgb_path = os.path.join(self.path['results'], 'heuristic_rgb_material_vector_v1', folder_name, 'clustered_rgb')

            self.folder_checker(clustered_rgb_path)


            ##makes a dict that get every material-rgb value and swir value -- only for new data version##
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

            #continue from here
            create_new_clustered_rgb_folder = self.config['functions']['heuristic_rgb_material_vector_v1'][
                'create_new_clustered_rgb_folder']


            use_clustered_rgb_folder = self.config['functions']['heuristic_rgb_material_vector_v1'][
                'use_clustered_base_rgb_folder']
            if create_new_clustered_rgb_folder: # compute case for  main_base_bgr_path
                rgb_frames_list = np.sort(os.listdir(main_rgb_path))
                self.cluster_rgb(main_rgb_path, clustered_rgb_path, rgb_frames_list)


            ##loading the heuiristics 0\1 matrix##
            hot_vector_csv_path = os.path.join(self.path['results'], 'seg_mapping',
                                               self.config['functions']['heuristic_rgb_material_vector_v1'][
                                                   'hot_vector_name'])
            hot_vector_df = pd.read_csv(hot_vector_csv_path)
            print("hot_vector_df was loaded")
            #
            # ##sorting scene data from 0-max##
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

            # ##making a dict of base rgb as key and mean of rgb per every base rgb##
            create_new_clu_rgb_to_rgb_dict_and_unique_clu_rgb_per_obj = \
            self.config['functions']['heuristic_rgb_material_vector_v1'][
                'create_new_clu_rgb_to_rgb_dict_and_unique_clu_rgb_per_obj']
            #
            if create_new_clu_rgb_to_rgb_dict_and_unique_clu_rgb_per_obj:
                unique_clu_rgb_and_mean_rgb_dict, unique_clu_rgb_per_object_dict = self.unique_clu_rgb_to_mean_rgb_and_seg_to_unique_rgb(
                    clu_frames_list, rgb_frame_list, seg_frames_list, main_rgb_path, main_clu_path,
                    main_seg_path, objects_seg_rgb_data)

            #
            ##ordering the dict keys to be configured as df with defulted template --> [x, y, z]##
            unique_clu_rgb_and_mean_rgb_dict_with_ordered_keys = self.order_string_keys(
                unique_clu_rgb_and_mean_rgb_dict)
            #
            # ##creating full scores matrix for the scene##
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



    """ - -----------------------------  Frames Classifier - ----------------------------------------- """
    def vgg19_clasiffier(self):

        matplotlib.use('Agg')

        num_classes =  self.config['functions']['vgg19_clasiffier']['num_classes']
        path_to_trained_weights = self.config['functions']['vgg19_clasiffier']['models_weights_path']
        test_folder_name = self.config['functions']['vgg19_clasiffier']['test_folder_name']
        path_to_input_images_folder = self.config['functions']['vgg19_clasiffier']['path_to_input_images_folder']
        path_to_output_results = os.path.join(self.path['results'], 'vgg19_clasiffier', test_folder_name)


        folders_names_list = ['masks', 'images']
        for folder_name in folders_names_list:
            self.folder_checker(os.path.join(path_to_output_results, folder_name))


        num_classes = num_classes

        vgg19 = SegmentationModel(num_classes, False)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        vgg19.load_state_dict(torch.load(path_to_trained_weights, map_location=torch.device('cpu')))
        vgg19.eval()  # Set model to evaluation mode



        for frame in os.listdir(path_to_input_images_folder):
            full_input_frame_path = os.path.join(path_to_input_images_folder, frame)

            # Load the input frame
            image_cv = cv2.imread(full_input_frame_path, cv2.IMREAD_UNCHANGED)  # Image.open(self.images_list_paths[idx])
            image = Image.fromarray(image_cv)
            input_tensor = transform(image)
            mask_cv_gt = cv2.imread(full_input_frame_path.replace('images', 'masks').replace('.jpg', '.png'), cv2.IMREAD_UNCHANGED)

            # Pass the input frame through the model to obtain the predicted mask
            with torch.no_grad():
                predicted_output = vgg19(input_tensor)

            predicted_mask = torch.argmax(predicted_output, dim=0).numpy().astype('uint8')

            cv2.imwrite(os.path.join(path_to_output_results, 'masks', frame.replace('jpg', 'png')), predicted_mask)
            cv2.imwrite((os.path.join(path_to_output_results, 'images', frame)), image_cv)

            plt.figure(figsize=(15, 5))

            # Input frame
            plt.subplot(1, 3, 1)
            plt.imshow(image_cv)  # Convert BGR to RGB
            plt.title("Input Frame")
            plt.axis("off")

            # Ground truth mask
            plt.subplot(1, 3, 2)
            plt.imshow(mask_cv_gt, cmap='jet')  # Adjust colormap if necessary
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














# from Backend.Framework_dev.BaseModelDev import BaseModelDev
from Frontend.Framework.BaseModel import BaseModel
from Backend.Data.src.ElasticsearchHandler import ElasticsearchHandler
from spectral import *
import numpy as np
import cv2
from scipy.interpolate import interp1d
import torch
import torch.nn.functional as TF
import torchvision.transforms as transforms
import os
from skimage.util import random_noise
import matplotlib.pyplot as plt

class PR_to_Rad(BaseModel):

    def __init__(self, version):
        BaseModel.__init__(self, version)
        self.num_wl = len(self.config['ref_to_rad']['wavelengths'])

    def __call__(self, *args, **kwargs):
        pass

    def pr_to_rad(self, path_cube, path_rgb, path_base_rgb, path_d_map, radial_dist_map, psf_dict, reg=True, psf=True, tm=None, shot=True):
        """
        The function converts a materials cube into a Spectro RAD cube.
        :param path_cube: Path to transplanted materials cube (Ref Cube)
        :param path_rgb: Path to final RGB image from Unreal
        :param path_base_rgb: Path to basecolor (base RGB) image from Unreal
        :param path_d_map: Path to depth map
        :param radial_dist_map: Radial distance map (normalized)
        :param psf_dict: Dictionary containing PSF information
        :return final_cube: Converted Rad Cube
        """


        cube_stage_list = []
        distance_thresh = self.config['ref_to_rad']['units']['dist_thresh']
        cur_cube = np.load(path_cube)
        cur_d_map = np.load(path_d_map)
        if np.median(cur_d_map) > distance_thresh:   # Checking the units of the depth map
            cur_d_map = cur_d_map / self.config['ref_to_rad']['units']['km']  # Normalizing to km units (For Modtran)
        sky_ind = np.argwhere(cur_d_map > distance_thresh)

        # Applying texture
        textured_cube = self.apply_texture(path_base_rgb, path_rgb, cur_cube, sky=sky_ind)
        cube_stage_list.append(textured_cube)

        # Applying atmosphere
        A, B, S, La, Range, modtran_data = self.get_modtran_coeffs(depth_map=cur_d_map.copy())
        atm_cube = self.transform_ref_to_rad(textured_cube.copy(), cur_d_map, A, B, S, La, Range)
        cube_stage_list.append(atm_cube)

        # Tools Matching
        if tm is not None:
            matched_cube = self.tools_match(cube_stage_list[-1].copy(), tm, mode='Slopes')
            cube_stage_list.append(matched_cube)
        else:
            matched_cube = self.tools_match(cube_stage_list[-1].copy(), tm, mode='Standard')
            cube_stage_list.append(matched_cube)
        # Registration warp
        if reg:
            reg_cube = self.registration_warp(cube_stage_list[-1].copy())
            cube_stage_list.append(reg_cube)

        # PSF blurring
        if psf:
            psf_cube = self.psf_blurrer(cube_stage_list[-1].copy(), psf_dict, radial_dist_map)
            cube_stage_list.append(psf_cube)
            numpy_cube = self.tens_to_numpy(cube_stage_list[-1])
            cube_stage_list.append(numpy_cube)

        # Adding Shot Noise
        if shot:
            shot_cube = self.shot_noise(cube_stage_list[-1].copy())
            cube_stage_list.append(shot_cube)

        res_cube = cube_stage_list[-1]
        final_cube = res_cube

        # Returning the sky locations to -1
        if len(sky_ind) != 0:
            final_cube[sky_ind[:, 0], sky_ind[:, 1], :] = -1

        return final_cube, modtran_data

    def tools_match(self, cube, tm, mode):
        """
        The function accepts a hyperspectral cube and matches it to the other domain using slopes and offsets
        :param cube: Hyperspectral cube to be matched
        :param tm: Tools matching vector containing slope and offset
        :param mode: Mode to tools match E{'Standard', 'Linear'}
        :return: matched_cube
        """
        if mode == 'Standard':
            matched_cube = cube
            # cube_obj = CubeHyperspectral()
            # modtran_factor = cube_obj.config['Modtran_Params']['factor_Modtran2Real']
            # matched_cube = modtran_factor*cube
        else:
            m = tm[0]
            b = tm[1]
            matched_cube = cube*m + b
        return matched_cube




    def apply_texture(self, base_rgb_path, rgb_img_path, cube, sky=None, save=None):
        """
        The function calculates and applies the "texture" map to a simulated materials cube.
        :param base_rgb_path: Path to the base rgb of the Unreal image
        :param rgb_img_path:  Path to the rgb Unreal image
        :param cube: Simulated materials cube
        :param sky: Indices for the cube that contain sky pixels. Set to empty.
        :param save: (Optional) Dictionary containing info for saving image.
        :return textured_cube: Cube after applying texture
        """

        # Reading images and converting to grayscale.
        base_rgb = cv2.imread(base_rgb_path)
        rgb_img = cv2.imread(rgb_img_path)
        gray_base_rgb = cv2.cvtColor(base_rgb, cv2.COLOR_BGR2GRAY)
        gray_rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        unique_base_rgb = np.unique(gray_base_rgb)
        text_map = np.zeros(gray_rgb_img.shape)
        for gl in unique_base_rgb:
            a = np.zeros(text_map.shape)
            rel_ind = np.argwhere(gray_base_rgb == gl)
            a[rel_ind[:, 0], rel_ind[:, 1]] = gray_rgb_img[rel_ind[:, 0], rel_ind[:, 1]]
            max_val = a.max()
            if max_val == 0:
                max_val = 1
            a = a/max_val
            text_map += a

        norm_unreal_map = text_map
        text_cube = np.dstack([norm_unreal_map] * self.num_wl) # Making texture map into texture cube.
        textured_cube = text_cube * cube

        # Resetting sky pixels values
        if sky is not None:
            textured_cube[sky[:, 0], sky[:, 1], :] = -1

        return textured_cube

    def get_modtran_coeffs(self, Atm_Modl=None, Sun_Zenith=None, Alt=None, Sun_Azimuth=None, RH=None, Aer_Modl=None, Vis=None, Range=None, depth_map=None):
        """
        The function retrieves Modtran coefficients (A, B, S, La)
        from the ElasticSearch database based on the arguments. The arguments are all set
        to None by default. If the user does not input an argument, a random variable (from the list of possible values
        for that variable in the config file) is chosen.
        :param Atm_Modl: Atmospheric Model [1-6]
        :param Sun_Zenith: Sun Zenith Angle [deg]
        :param Alt: Altitude [km]
        :param Sun_Azimuth: Sun Azimuth Angle [deg]
        :param RH: Relative Humidity [%]
        :param Aer_Modl: Aerosol Model [1-5]
        :param Vis: Visibility [km]
        :param Range: Distance to sensor [km]
        :param depth_map: Image Depth Map [km]
        :return A, B, S, La, modtran_range, modtran_rand:
        A, B, S, La - Parameters from Modtran in numpy arrays
        modtran_range - Ranges that were used in numpy array
        modtran_rand - Dictionary containing the atmospheric data
        """


        # Pulling all possible Modtran values from config
        modtran_params = self.config['ref_to_rad']['default_modtran_params']
        min_modtran_range = min( self.config['ref_to_rad']['default_modtran_params']['Range'])
        max_modtran_range = max( self.config['ref_to_rad']['default_modtran_params']['Range'])


        # Adding user inputs as lists. Must be list for random number generation in while loop.
        if Atm_Modl is not None:
            modtran_params['Atm_Modl'] = [Atm_Modl]
        if Sun_Zenith is not None:
            modtran_params['Sun_Zenith'] = [Sun_Zenith]
        if Sun_Azimuth is not None:
            modtran_params['Sun_Azimuth'] = [Sun_Azimuth]
        if Alt is not None:
            modtran_params['Alt'] = [Alt]
        if Vis is not None:
            modtran_params['Vis'] = [Vis]
        if RH is not None:
            modtran_params['RH'] = [RH]
        if Aer_Modl is not None:
            modtran_params['Aer_Modl'] = [Aer_Modl]

        # Finding range from depth map
        min_range = 100 # Arbitrary high number
        if depth_map is not None:
            min_range = np.floor(np.min(depth_map))
            max_range = np.ceil(np.max(depth_map))
            if min_range==-1:
                a = depth_map
                a[a==-1] = 1000 # Setting to arbitrary high number
                min_range = np.floor(np.min(a))
            if min_range < min_modtran_range:
                min_range = min_modtran_range
            if max_range > max_modtran_range:
                max_range = max_modtran_range


        # Setting up ElasticSearch query
        es = ElasticsearchHandler()
        es.config['functions']['get_data_without_version']['index'] = 'modtran'
        es.config['functions']['get_data_without_version']['fields'] = ['A', 'B', 'La', 'S', 'Range']
        flag = 0
        while flag == 0:
            modtran_rand = []
            for idx in modtran_params.keys():  # Creating query for each parameter
                r2 = np.random.randint(0, len(modtran_params[idx]))
                if idx=='Range':
                    continue
                if idx == 'Atm_Modl' or idx == 'Aer_Modl':  # Must add .keyword to strings
                    modtran_rand.append({'term': {f'{idx}' + '.keyword': modtran_params[idx][r2]}})
                else:
                    modtran_rand.append({'term': {f'{idx}': modtran_params[idx][r2]}})
                if idx == 'Alt':  # Saving altitude to later remove impossible results
                    alt = modtran_params[idx][r2]
            if min_range < alt:  # Not physically possible (assuming relatively flat ground
                continue
            # Adding Range query in range of depth map
            if (Range is None) and (depth_map is not None):
                modtran_rand.append({'range': {'Range': {'gte': min_range, 'lte': max_range + min_modtran_range}}})

            # Running query in elastic search
            es.config['functions']['get_data_without_version']['queries']['bool']['must'] = modtran_rand
            results = es.get_data_without_version()

            # Dealing with empty results (shouldn't happen)
            if type(results) == list:
                if results:
                    flag = 1
            else:
                if not results.empty:
                    flag = 1

        # Converting resulting dataframe to numpy array
        A = np.array(results.loc[:, 'A'].to_list())
        B = np.array(results.loc[:, 'B'].to_list())
        S = np.array(results.loc[:, 'S'].to_list())
        La = np.array(results.loc[:, 'La'].to_list())
        Range = np.array(results.loc[:, 'Range'].to_list())


        return A, B, S, La, Range, modtran_rand

    def transform_ref_to_rad(self, old_cube, depth_map, A, B, S, La, modtran_range):
        """
        :param old_cube: Reflectance hyperspectral cube
        :param depth_map: Depth map in units of [km]
        :param A, B, S, La: Modtran atmospheric parameters
        :param modtran_range: Range that was used to get A, B, S, La - for interpolation
        :param rad_to_ref: (Optional) In case of calibrating cubes from rad to ref, rad_to_ref = True
        :return rad_cube: Radiance cube
        """
        d_map_round = self.config['ref_to_rad']['units']['d_map_round']
        min_modtran_range = min(self.config['ref_to_rad']['default_modtran_params']['Range'])
        max_modtran_range = max(self.config['ref_to_rad']['default_modtran_params']['Range'])
        ro_env_kernel_size = self.config['ref_to_rad']['units']['ro_env_kernel_size']

        # Preparing depth map
        depth_map = depth_map.round(decimals=d_map_round)
        min_depth = depth_map.min()
        depth_map[depth_map < min_modtran_range] = min_modtran_range
        depth_map[depth_map > max_modtran_range] = max_modtran_range

        unique_dist = np.unique(depth_map)
        if min_depth == -1: # In the case where the sky was preset to -1
            unique_dist = np.unique(depth_map)[1:] # Only take the positive values
        if len(unique_dist) == 1: # There is only one range in the depth map (edge case)
            A_inter = A
            B_inter = B
            S_inter = S
            La_inter = La
        else: # There are many ranges, we need to interpolate
            A_inter = np.zeros((self.num_wl, len(unique_dist)))
            B_inter = np.zeros((self.num_wl, len(unique_dist)))
            S_inter = np.zeros((self.num_wl, len(unique_dist)))
            La_inter = np.zeros((self.num_wl, len(unique_dist)))

            # Interpolate for each wavelength
            for i in range(0, self.num_wl):
                A_inter[i, :] = interp1d(modtran_range, A[:, i])(unique_dist)
                B_inter[i, :] = interp1d(modtran_range, B[:, i])(unique_dist)
                S_inter[i, :] = interp1d(modtran_range, S[:, i])(unique_dist)
                La_inter[i, :] = interp1d(modtran_range, La[:, i])(unique_dist)

        # For easy multiplication, making coefficient cubes
        A_mat = np.zeros(old_cube.shape)
        B_mat = np.zeros(old_cube.shape)
        S_mat = np.zeros(old_cube.shape)
        La_mat = np.zeros(old_cube.shape)
        for k in range(0, len(unique_dist)):
            indy = np.where(depth_map == unique_dist[k])
            A_mat[indy[0], indy[1], :] = A_inter[:, k]
            B_mat[indy[0], indy[1], :] = B_inter[:, k]
            S_mat[indy[0], indy[1], :] = S_inter[:, k]
            La_mat[indy[0], indy[1], :] = La_inter[:, k]

        ro_env = cv2.GaussianBlur(old_cube, ro_env_kernel_size, 0)
        # Transforming ref to RAD
        rad_cube = (A_mat * old_cube / (1 - S_mat * ro_env)) + (B_mat * ro_env / (1 - S_mat * ro_env)) + La_mat
        return rad_cube

    def registration_warp(self, cube):
        """
        The function simulates registration warp by shifting each channel by
        a random number of pixels, and rotating by a random number of pixels.
        :param cube: Hyperspectral cube.
        :return:
        """
        (h, w) = cube.shape[:2]
        (cX, cY) = (h//2, w//2)
        shifted_cube = np.zeros(cube.shape)
        dx_range = self.config['ref_to_rad']['units']['pixel shift x']
        dy_range = self.config['ref_to_rad']['units']['pixel shift y']
        rotation_range = self.config['ref_to_rad']['units']['rotation shift']
        # Warping each channel
        for i in range(self.num_wl):
            cur_channel = cube[:, :, i]
            rand_dx = np.random.randint(dx_range[0], dx_range[1])
            rand_dy = np.random.randint(dy_range[0], dy_range[1])
            x_shifted_img =  self.pixel_shift(cur_channel, rand_dx, x=True)
            xy_shifted_img = self.pixel_shift(x_shifted_img, rand_dy, x=False)
            rand_pixel_angle = np.random.randint(rotation_range[0], rotation_range[1])
            rad_angle = np.arctan((rand_pixel_angle / w))
            deg_angle = np.rad2deg(rad_angle)
            rotation_matrix = cv2.getRotationMatrix2D((cX, cY), deg_angle, 1.0)
            rotated_img = cv2.warpAffine(xy_shifted_img, rotation_matrix, (w, h))
            shifted_cube[:, :, i] = rotated_img

        return shifted_cube

    def pixel_shift(self, img, n, x):
        """
        The functions shifts an image by n pixels in the x or y direction
        :param img: input image
        :param n: number of pixels to shift by
        :param x: True if in x dimension, False if in y dimension.
        :return shifted_img: Image shifted by n pixels in appropriate direction
        """
        if n==0:
            return img
        shifted_img = np.zeros(img.shape)
        if n > 0:
            if x:
                rel_area = img[:, n:]
                shifted_img[:, n:] = rel_area
            else:
                rel_area = img[n:, :]
                shifted_img[n:, :] = rel_area
        else:
            if x:
                rel_area = img[:, :n]
                shifted_img[:, :n] = rel_area
            else:
                rel_area = img[:n, :]
                shifted_img[:n, :] = rel_area

        return shifted_img

    def psf_blurrer(self, cube, psf_dict, radial_dist_map):
        """
        :param cube: Hyperspectral cube to be blurred with PSF
        :param psf_dict: Dictionary containing all PSF data
        :param radial_dist_map: Map of relative distance from center of frame values are in range: [0,1]
        :return blurred_cube: PSF blurred cube
        """
        spectro_fov = self.config['ref_to_rad']['psf']['spectro_fov']
        unreal_fov = self.config['ref_to_rad']['psf']['unreal_fov']
        spectro_dist = self.config['ref_to_rad']['psf']['spectro_dist']
        unreal_dist = self.config['ref_to_rad']['psf']['unreal_dist']
        cube[cube == -1] = 0
        ex_psf = psf_dict[str(self.config['ref_to_rad']['wavelengths'][0])]['0.0']
        psf_dims = ex_psf.shape
        delta_spectro = spectro_dist*np.tan(np.deg2rad(spectro_fov[0]/2))
        delta_unreal = unreal_dist*np.tan(np.deg2rad(unreal_fov[0]/2))
        prop = delta_spectro/delta_unreal
        resize_x = int(prop*psf_dims[0])
        resize_y = int(prop*psf_dims[1])
        convert_tensor = transforms.ToTensor()
        radial_dist_map = convert_tensor(radial_dist_map)
        cube = convert_tensor(cube).cuda()
        blurred_cube = torch.zeros(cube.shape)
        unique_dist = torch.unique(radial_dist_map)
        for dist in unique_dist:
            psf_cube = self.psf_interpolator(dist.numpy(), psf_dict, resize_x, resize_y)
            psf_cube = convert_tensor(psf_cube).cuda()
            convolved_cube = torch.zeros(cube.shape)
            for i in range(self.num_wl):
                img = cube[i].view(1, 1, cube.shape[1], cube.shape[2])
                psf_img = psf_cube[i].view(1, 1, psf_cube.shape[1], psf_cube.shape[2])
                convolved_img = TF.conv2d(input=img, weight=psf_img, stride=[1, 1], padding='same')
                convolved_cube[i, :, :] = convolved_img[0, 0, :, :]
            ind = np.argwhere(radial_dist_map==dist)
            blurred_cube[:, ind[1, :], ind[2, :]] = convolved_cube[:, ind[1, :], ind[2, :]]
            print(dist)
        return blurred_cube

    def psf_interpolator(self, dist, psf_dict, reshape_x, reshape_y):
        """
        The function interpolates the psf to get the psf at a specific radial distance.
        :param dist: Radial distance from the center of the image
        :param psf_dict: PSF dictionary containing PSF info for the camera
        :param reshape_x: Reshape size x
        :param reshape_y: Reshape size x
        :return:
        """
        wvls = self.config['ref_to_rad']['wavelengths']
        ex_wvl = str(wvls[0])
        ex_dict = psf_dict[ex_wvl]
        d_keys = ex_dict.keys()
        d_list = list(d_keys)
        d_vec = np.array([float(i) for i in d_list])
        low, high = self.get_low_high(d_vec, dist)
        dims = [reshape_y, reshape_x, self.num_wl]
        psf_cube = np.zeros(dims)
        count = 0
        for wvl in wvls:
            wvl_key = str(wvl)
            if low==high:
                rel_psf = psf_dict[wvl_key][str(low)]
                resized_psf = cv2.resize(rel_psf, (reshape_x, reshape_y))
            else:
                low_psf = psf_dict[wvl_key][str(low)]
                high_psf = psf_dict[wvl_key][str(high)]
                interp_psf = (high_psf - low_psf)/(high - low)*(dist - high) + high_psf
                interp_psf = interp_psf/np.sum(interp_psf)
                resized_psf = cv2.resize(interp_psf, (reshape_x, reshape_y))
            psf_cube[:, :, count] = resized_psf/np.sum(resized_psf)  # Normalizing PSF by sum
            count+=1
        return psf_cube

    def get_low_high(self, vec, x):
        """
        The functions finds the higher and lower value of x in the vector vec.
        :param vec: vector
        :param x: float number to be found within the vector
        :return lower_val, upper_val: lower and upper value for x in vector vec.
        """
        deltas = np.abs(vec - x)
        if 0 in deltas:  # Case where the value is in the vector
            ind = np.argwhere(deltas == 0)
            val = float(vec[ind])
            return val, val
        sort_ind = np.argsort(deltas)
        lower = int(np.min([sort_ind[0], sort_ind[1]]))
        upper = int(np.max([sort_ind[0], sort_ind[1]]))
        lower_val = vec[lower]
        upper_val = vec[upper]
        return lower_val, upper_val

    def load_psf_files(self):
        """
        The function reads the pre-made numpy format psf files,
        and returns them in a dictionary.
        :return psf_dict: Dictionary containing the PSF info for the Spectro CU
        """
        psf_files_loc = self.config['ref_to_rad']['psf']['psf_npy_files_loc']
        psf_folders = os.listdir(psf_files_loc)
        psf_folders = np.sort(psf_folders)
        psf_dict = {

        }
        for folder in psf_folders:
            psf_dict[folder] = {}
            full_folder_path = os.path.join(psf_files_loc, folder)
            files = np.sort(os.listdir(full_folder_path))
            for f in files:
                full_file_path = os.path.join(full_folder_path, f)
                psf_dict[folder][f[:-4]] = np.load(full_file_path)

        return psf_dict

    def tens_to_numpy(self, cube):
        """
        The function converts a Tensor hyperspectral cube to Numpy
        :param cube: Hyperspectral cube
        :return converted_cube: Numpy cube
        """
        converted_cube = cube.permute([1, 2, 0]).cpu().numpy()
        return converted_cube

    def get_radial_distance_map(self, img_dims, normalize=False, round=None):
        """
        The function creates a radial distance map based off of the input dimensions.
        :param img_dims: Desired image dimensions
        :param normalize: (Optional) Flag to normalize the values in the map
        :param round: (Optional) Flag to round the values to "round" decimals
        :return radial_distance_map: Final radial distance map
        """
        radial_distance_map = np.zeros(img_dims)
        for i in range(img_dims[0]):
            for j in range(img_dims[1]):
                radial_distance_map[i, j] = np.sqrt((i - img_dims[0]/2)**2 + (j - img_dims[1]/2)**2)
        if normalize:
            radial_distance_map = radial_distance_map/radial_distance_map.max()
        if round is not None:
            radial_distance_map = radial_distance_map.round(round)
        return radial_distance_map

    def matched_cube(self, cube):
        slopes = self.config['ref_to_rad']['modtran_to_rad']['slopes']
        offsets = self.config['ref_to_rad']['modtran_to_rad']['offsets']

        matched_cube = cube*slopes + offsets
        return matched_cube

    def shot_noise(self, cube):
        noisy_cube = random_noise(cube, mode='poisson')
        return noisy_cube

    def generate_rad_cubes(self):
        test_folder = self.config['cubes_generator']['test_folder']
        main_masks = self.config['ref_to_rad']['main_masks_path']
        base_rgb_key = self.config['ref_to_rad']['base_rgb_key']
        dist_key = self.config['ref_to_rad']['distance_key']
        rgb_key = self.config['ref_to_rad']['rgb_key']
        main_frames = self.config['ref_to_rad']['main_base_rgb']
        mat_trans_type = self.config['ref_to_rad']['materials_transplant_type']
        ref_cubes_key = self.config['ref_to_rad']['ref_cubes_key']
        main_distance = os.path.join(main_masks, test_folder, dist_key)
        main_base_rgb = os.path.join(main_masks, test_folder, base_rgb_key)
        main_rgb = os.path.join(main_frames, test_folder, rgb_key)
        main_cubes = os.path.join(self.path['results'], test_folder, mat_trans_type, ref_cubes_key)
        rad_cubes_results_loc = os.path.join(self.path['results'], mat_trans_type, 'RAD_Cubes')

        # Getting Folder Contents
        cubes_list = np.sort(os.listdir(main_cubes))
        base_rgb_list = np.sort(os.listdir(main_base_rgb))
        rgb_list = np.sort(os.listdir(main_rgb))
        distance_list = np.sort(os.listdir(main_distance))

        cube_path_for_radial_map = os.path.join(main_cubes, cubes_list[0])
        cube_for_radial_map = np.load(cube_path_for_radial_map)

        radial_dist_map = self.get_radial_distance_map(cube_for_radial_map.shape[:2], normalize=True, round=1)
        psf_dict = self.load_psf_files()

        for i, cube in enumerate(cubes_list):
            path_dist_map = os.path.join(main_distance, distance_list[i])
            path_rgb = os.path.join(main_rgb, rgb_list[i])
            path_base_rgb = os.path.join(main_base_rgb, base_rgb_list[i])
            path_cube = os.path.join(main_cubes, cube)
            rad_cube, _ = self.pr_to_rad(path_cube=path_cube, path_rgb=path_rgb, path_base_rgb=path_base_rgb,
                                         path_d_map=path_dist_map, radial_dist_map=radial_dist_map, psf_dict=psf_dict,
                                         tm=False, reg=True, psf=True)
            rad_cube_save_loc = os.path.join(rad_cubes_results_loc, cube)
            np.save(rad_cube_save_loc, rad_cube)






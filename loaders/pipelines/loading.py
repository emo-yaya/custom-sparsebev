import os
import mmcv
import numpy as np
from mmdet.datasets.builder import PIPELINES
from numpy.linalg import inv
from mmcv.runner import get_dist_info
import torch
from pyquaternion import Quaternion
from mmdet3d.datasets.builder import PIPELINES as MMDET3D_PIPELINES

def compose_lidar2img(ego2global_translation_curr,
                      ego2global_rotation_curr,
                      lidar2ego_translation_curr,
                      lidar2ego_rotation_curr,
                      sensor2global_translation_past,
                      sensor2global_rotation_past,
                      cam_intrinsic_past):
    
    R = sensor2global_rotation_past @ (inv(ego2global_rotation_curr).T @ inv(lidar2ego_rotation_curr).T)
    T = sensor2global_translation_past @ (inv(ego2global_rotation_curr).T @ inv(lidar2ego_rotation_curr).T)
    T -= ego2global_translation_curr @ (inv(ego2global_rotation_curr).T @ inv(lidar2ego_rotation_curr).T) + lidar2ego_translation_curr @ inv(lidar2ego_rotation_curr).T

    lidar2cam_r = inv(R.T)
    lidar2cam_t = T @ lidar2cam_r.T

    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t

    viewpad = np.eye(4)
    viewpad[:cam_intrinsic_past.shape[0], :cam_intrinsic_past.shape[1]] = cam_intrinsic_past
    lidar2img = (viewpad @ lidar2cam_rt.T).astype(np.float32)

    return lidar2img

def get_sensor2ego_transformation(cam_info,
                                      key_info,
                                      cam_name,
                                      ego_cam=None, index=None):
    if ego_cam is None:
        ego_cam = cam_name
    try:
        sensor2ego_rotation_fbbev = cam_info['sensor2ego_rotation_fbbev'][index]
        sensor2ego_translation_fbbev = cam_info['sensor2ego_translation_fbbev'][index]
        ego2global_rotation_fbbev = cam_info['ego2global_rotation_fbbev'][index]
        ego2global_translation_fbbev = cam_info['ego2global_translation_fbbev'][index]
        assert len(sensor2ego_rotation_fbbev) == 4
    except:
        sensor2ego_rotation_fbbev = cam_info['sensor2ego_rotation_fbbev']
        sensor2ego_translation_fbbev = cam_info['sensor2ego_translation_fbbev']
        ego2global_rotation_fbbev = cam_info['ego2global_rotation_fbbev']
        ego2global_translation_fbbev = cam_info['ego2global_translation_fbbev']
    w, x, y, z = sensor2ego_rotation_fbbev
    # sweep sensor to sweep ego
    sweepsensor2sweepego_rot = torch.Tensor(
        Quaternion(w, x, y, z).rotation_matrix)
    sweepsensor2sweepego_tran = torch.Tensor(
        sensor2ego_translation_fbbev)
    sweepsensor2sweepego = sweepsensor2sweepego_rot.new_zeros((4, 4))
    sweepsensor2sweepego[3, 3] = 1
    sweepsensor2sweepego[:3, :3] = sweepsensor2sweepego_rot
    sweepsensor2sweepego[:3, -1] = sweepsensor2sweepego_tran
    # sweep ego to global
    w, x, y, z = ego2global_rotation_fbbev
    sweepego2global_rot = torch.Tensor(
        Quaternion(w, x, y, z).rotation_matrix)
    sweepego2global_tran = torch.Tensor(
        ego2global_translation_fbbev)
    sweepego2global = sweepego2global_rot.new_zeros((4, 4))
    sweepego2global[3, 3] = 1
    sweepego2global[:3, :3] = sweepego2global_rot
    sweepego2global[:3, -1] = sweepego2global_tran

    # global sensor to cur ego
    w, x, y, z = key_info['ego2global_rotation_fbbev'][index]
    keyego2global_rot = torch.Tensor(
        Quaternion(w, x, y, z).rotation_matrix)
    keyego2global_tran = torch.Tensor(
        key_info['ego2global_translation_fbbev'][index])
    keyego2global = keyego2global_rot.new_zeros((4, 4))
    keyego2global[3, 3] = 1
    keyego2global[:3, :3] = keyego2global_rot
    keyego2global[:3, -1] = keyego2global_tran
    global2keyego = keyego2global.inverse()

    sweepsensor2keyego = \
        global2keyego @ sweepego2global @ sweepsensor2sweepego

    # global sensor to cur ego
    w, x, y, z = key_info['ego2global_rotation_fbbev'][index]
    keyego2global_rot = torch.Tensor(
        Quaternion(w, x, y, z).rotation_matrix)
    keyego2global_tran = torch.Tensor(
        key_info['ego2global_translation_fbbev'][index])
    keyego2global = keyego2global_rot.new_zeros((4, 4))
    keyego2global[3, 3] = 1
    keyego2global[:3, :3] = keyego2global_rot
    keyego2global[:3, -1] = keyego2global_tran
    global2keyego = keyego2global.inverse()

    # cur ego to sensor
    w, x, y, z = key_info['sensor2ego_rotation_fbbev'][index]
    keysensor2keyego_rot = torch.Tensor(
        Quaternion(w, x, y, z).rotation_matrix)
    keysensor2keyego_tran = torch.Tensor(
        key_info['sensor2ego_translation_fbbev'][index])
    keysensor2keyego = keysensor2keyego_rot.new_zeros((4, 4))
    keysensor2keyego[3, 3] = 1
    keysensor2keyego[:3, :3] = keysensor2keyego_rot
    keysensor2keyego[:3, -1] = keysensor2keyego_tran
    keyego2keysensor = keysensor2keyego.inverse()
    keysensor2sweepsensor = (
        keyego2keysensor @ global2keyego @ sweepego2global
        @ sweepsensor2sweepego).inverse()
    return sweepsensor2keyego, keysensor2sweepsensor


@MMDET3D_PIPELINES.register_module(force=True)
class LoadMultiViewImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool, optional): Whether to convert the img to float32.
            Defaults to False.
        color_type (str, optional): Color type of the file.
            Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filename']
        # img is of shape (h, w, c, num_views)
        img = np.stack(
            [mmcv.imread(name, self.color_type) for name in filename], axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formatting.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        
        results['rots'] = []
        results['trans'] = []
        results['intrins'] = []
        results['post_rots'] = []
        results['post_trans'] = []
        cam_types = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]
        for index, sensor in enumerate(cam_types):
            
            sensor2keyego, sensor2sensor = get_sensor2ego_transformation(results, results, sensor, sensor, index)
            rot = sensor2keyego[:3, :3]
            tran = sensor2keyego[:3, 3]
            results['rots'].append(rot)
            results['trans'].append(tran)
            results['intrins'].append(torch.tensor(results['cam_intrinsic'][index], dtype=torch.float32))
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results
    

@PIPELINES.register_module()
class LoadMultiViewImageFromMultiSweeps(object):
    def __init__(self,
                 sweeps_num=5,
                 color_type='color',
                 test_mode=False):
        self.sweeps_num = sweeps_num
        self.color_type = color_type
        self.test_mode = test_mode

        self.train_interval = [4, 8]
        self.test_interval = 6

        try:
            mmcv.use_backend('turbojpeg')
        except ImportError:
            mmcv.use_backend('cv2')

    def load_offline(self, results):
        cam_types = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]

        results['adjacent'] = []
        
        if len(results['sweeps']['prev']) == 0:
            for _ in range(self.sweeps_num):
                sweep = {}
                for j in range(len(cam_types)):
                    results['img'].append(results['img'][j])
                    sensor2keyego, sensor2sensor = get_sensor2ego_transformation(results, results, cam_types[j], cam_types[j], index=j)
                    rot = sensor2keyego[:3, :3]
                    tran = sensor2keyego[:3, 3]
                    results['rots'].append(rot)
                    results['trans'].append(tran)
                    results['intrins'].append(torch.tensor(results['cam_intrinsic'][j], dtype=torch.float32))
                    results['img_timestamp'].append(results['img_timestamp'][j])
                    results['filename'].append(results['filename'][j])
                    results['lidar2img'].append(np.copy(results['lidar2img'][j]))
                    sweep[cam_types[j]] = {
                        'data_path' : results['img_filename'][j],
                        'sensor2ego_rotation_fbbev' : results['sensor2ego_rotation_fbbev'][j],
                        'sensor2ego_translation_fbbev' : results['sensor2ego_translation_fbbev'][j],
                        'ego2global_rotation_fbbev' : results['ego2global_rotation_fbbev'][j],
                        'ego2global_translation_fbbev' : results['ego2global_translation_fbbev'][j],
                    }
                results['adjacent'].append(sweep)
        else:
            if self.test_mode:
                interval = self.test_interval
                choices = [(k + 1) * interval - 1 for k in range(self.sweeps_num)]
            elif len(results['sweeps']['prev']) <= self.sweeps_num:
                pad_len = self.sweeps_num - len(results['sweeps']['prev'])
                choices = list(range(len(results['sweeps']['prev']))) + [len(results['sweeps']['prev']) - 1] * pad_len
            else:
                max_interval = len(results['sweeps']['prev']) // self.sweeps_num
                max_interval = min(max_interval, self.train_interval[1])
                min_interval = min(max_interval, self.train_interval[0])
                interval = np.random.randint(min_interval, max_interval + 1)
                choices = [(k + 1) * interval - 1 for k in range(self.sweeps_num)]
            for idx in sorted(list(choices)):
                sweep_idx = min(idx, len(results['sweeps']['prev']) - 1)
                sweep = results['sweeps']['prev'][sweep_idx]

                if len(sweep.keys()) < len(cam_types):
                    sweep = results['sweeps']['prev'][sweep_idx - 1]
                results['adjacent'].append(sweep)
                for sensor in cam_types:
                    results['img'].append(mmcv.imread(sweep[sensor]['data_path'], self.color_type))
                    sensor2keyego, sensor2sensor = get_sensor2ego_transformation(sweep[sensor], results, sensor, 'CAM_FRONT', index=0)
                    rot = sensor2keyego[:3, :3]
                    tran = sensor2keyego[:3, 3]
                    results['rots'].append(rot)
                    results['trans'].append(tran)
                    results['intrins'].append(torch.tensor(sweep[sensor]['cam_intrinsic'], dtype=torch.float32))
                    results['img_timestamp'].append(sweep[sensor]['timestamp'] / 1e6)
                    results['filename'].append(os.path.relpath(sweep[sensor]['data_path']))
                    results['lidar2img'].append(compose_lidar2img(
                        results['ego2global_translation'],
                        results['ego2global_rotation'],
                        results['lidar2ego_translation'],
                        results['lidar2ego_rotation'],
                        sweep[sensor]['sensor2global_translation'],
                        sweep[sensor]['sensor2global_rotation'],
                        sweep[sensor]['cam_intrinsic'],
                    ))

        return results

    def load_online(self, results):
        # only used when measuring FPS
        assert self.test_mode
        assert self.test_interval == 6

        cam_types = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]

        if len(results['sweeps']['prev']) == 0:
            for _ in range(self.sweeps_num):
                for j in range(len(cam_types)):
                    results['img_timestamp'].append(results['img_timestamp'][j])
                    results['filename'].append(results['filename'][j])
                    results['lidar2img'].append(np.copy(results['lidar2img'][j]))
        else:
            interval = self.test_interval
            choices = [(k + 1) * interval - 1 for k in range(self.sweeps_num)]

            for idx in sorted(list(choices)):
                sweep_idx = min(idx, len(results['sweeps']['prev']) - 1)
                sweep = results['sweeps']['prev'][sweep_idx]

                if len(sweep.keys()) < len(cam_types):
                    sweep = results['sweeps']['prev'][sweep_idx - 1]

                for sensor in cam_types:
                    # skip loading history frames
                    results['img_timestamp'].append(sweep[sensor]['timestamp'] / 1e6)
                    results['filename'].append(os.path.relpath(sweep[sensor]['data_path']))
                    results['lidar2img'].append(compose_lidar2img(
                        results['ego2global_translation'],
                        results['ego2global_rotation'],
                        results['lidar2ego_translation'],
                        results['lidar2ego_rotation'],
                        sweep[sensor]['sensor2global_translation'],
                        sweep[sensor]['sensor2global_rotation'],
                        sweep[sensor]['cam_intrinsic'],
                    ))

        return results

    def __call__(self, results):
        if self.sweeps_num == 0:
            return results

        world_size = get_dist_info()[1]
        if world_size == 1 and self.test_mode and False:
            return self.load_online(results)
        else:
            return self.load_offline(results)


@PIPELINES.register_module()
class LoadMultiViewImageFromMultiSweepsFuture(object):
    def __init__(self,
                 prev_sweeps_num=5,
                 next_sweeps_num=5,
                 color_type='color',
                 test_mode=False):
        self.prev_sweeps_num = prev_sweeps_num
        self.next_sweeps_num = next_sweeps_num
        self.color_type = color_type
        self.test_mode = test_mode

        assert prev_sweeps_num == next_sweeps_num

        self.train_interval = [4, 8]
        self.test_interval = 6

        try:
            mmcv.use_backend('turbojpeg')
        except ImportError:
            mmcv.use_backend('cv2')

    def __call__(self, results):
        if self.prev_sweeps_num == 0 and self.next_sweeps_num == 0:
            return results

        cam_types = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]

        if self.test_mode:
            interval = self.test_interval
        else:
            interval = np.random.randint(self.train_interval[0], self.train_interval[1] + 1)

        # previous sweeps
        if len(results['sweeps']['prev']) == 0:
            for _ in range(self.prev_sweeps_num):
                for j in range(len(cam_types)):
                    results['img'].append(results['img'][j])
                    results['img_timestamp'].append(results['img_timestamp'][j])
                    results['filename'].append(results['filename'][j])
                    results['lidar2img'].append(np.copy(results['lidar2img'][j]))
        else:
            choices = [(k + 1) * interval - 1 for k in range(self.prev_sweeps_num)]

            for idx in sorted(list(choices)):
                sweep_idx = min(idx, len(results['sweeps']['prev']) - 1)
                sweep = results['sweeps']['prev'][sweep_idx]

                if len(sweep.keys()) < len(cam_types):
                    sweep = results['sweeps']['prev'][sweep_idx - 1]

                for sensor in cam_types:
                    results['img'].append(mmcv.imread(sweep[sensor]['data_path'], self.color_type))
                    results['img_timestamp'].append(sweep[sensor]['timestamp'] / 1e6)
                    results['filename'].append(sweep[sensor]['data_path'])
                    results['lidar2img'].append(compose_lidar2img(
                        results['ego2global_translation'],
                        results['ego2global_rotation'],
                        results['lidar2ego_translation'],
                        results['lidar2ego_rotation'],
                        sweep[sensor]['sensor2global_translation'],
                        sweep[sensor]['sensor2global_rotation'],
                        sweep[sensor]['cam_intrinsic'],
                    ))

        # future sweeps
        if len(results['sweeps']['next']) == 0:
            for _ in range(self.next_sweeps_num):
                for j in range(len(cam_types)):
                    results['img'].append(results['img'][j])
                    results['img_timestamp'].append(results['img_timestamp'][j])
                    results['filename'].append(results['filename'][j])
                    results['lidar2img'].append(np.copy(results['lidar2img'][j]))
        else:
            choices = [(k + 1) * interval - 1 for k in range(self.next_sweeps_num)]

            for idx in sorted(list(choices)):
                sweep_idx = min(idx, len(results['sweeps']['next']) - 1)
                sweep = results['sweeps']['next'][sweep_idx]

                if len(sweep.keys()) < len(cam_types):
                    sweep = results['sweeps']['next'][sweep_idx - 1]

                for sensor in cam_types:
                    results['img'].append(mmcv.imread(sweep[sensor]['data_path'], self.color_type))
                    results['img_timestamp'].append(sweep[sensor]['timestamp'] / 1e6)
                    results['filename'].append(sweep[sensor]['data_path'])
                    results['lidar2img'].append(compose_lidar2img(
                        results['ego2global_translation'],
                        results['ego2global_rotation'],
                        results['lidar2ego_translation'],
                        results['lidar2ego_rotation'],
                        sweep[sensor]['sensor2global_translation'],
                        sweep[sensor]['sensor2global_rotation'],
                        sweep[sensor]['cam_intrinsic'],
                    ))

        return results


'''
This func loads previous and future frames in interleaved order, 
e.g. curr, prev1, next1, prev2, next2, prev3, next3...
'''
@PIPELINES.register_module()
class LoadMultiViewImageFromMultiSweepsFutureInterleave(object):
    def __init__(self,
                 prev_sweeps_num=5,
                 next_sweeps_num=5,
                 color_type='color',
                 test_mode=False):
        self.prev_sweeps_num = prev_sweeps_num
        self.next_sweeps_num = next_sweeps_num
        self.color_type = color_type
        self.test_mode = test_mode

        assert prev_sweeps_num == next_sweeps_num

        self.train_interval = [4, 8]
        self.test_interval = 6

        try:
            mmcv.use_backend('turbojpeg')
        except ImportError:
            mmcv.use_backend('cv2')

    def __call__(self, results):
        if self.prev_sweeps_num == 0 and self.next_sweeps_num == 0:
            return results

        cam_types = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]

        if self.test_mode:
            interval = self.test_interval
        else:
            interval = np.random.randint(self.train_interval[0], self.train_interval[1] + 1)

        results_prev = dict(
            img=[],
            img_timestamp=[],
            filename=[],
            lidar2img=[]
        )
        results_next = dict(
            img=[],
            img_timestamp=[],
            filename=[],
            lidar2img=[]
        )

        if len(results['sweeps']['prev']) == 0:
            for _ in range(self.prev_sweeps_num):
                for j in range(len(cam_types)):
                    results_prev['img'].append(results['img'][j])
                    results_prev['img_timestamp'].append(results['img_timestamp'][j])
                    results_prev['filename'].append(results['filename'][j])
                    results_prev['lidar2img'].append(np.copy(results['lidar2img'][j]))
        else:
            choices = [(k + 1) * interval - 1 for k in range(self.prev_sweeps_num)]

            for idx in sorted(list(choices)):
                sweep_idx = min(idx, len(results['sweeps']['prev']) - 1)
                sweep = results['sweeps']['prev'][sweep_idx]

                if len(sweep.keys()) < len(cam_types):
                    sweep = results['sweeps']['prev'][sweep_idx - 1]

                for sensor in cam_types:
                    results_prev['img'].append(mmcv.imread(sweep[sensor]['data_path'], self.color_type))
                    results_prev['img_timestamp'].append(sweep[sensor]['timestamp'] / 1e6)
                    results_prev['filename'].append(os.path.relpath(sweep[sensor]['data_path']))
                    results_prev['lidar2img'].append(compose_lidar2img(
                        results['ego2global_translation'],
                        results['ego2global_rotation'],
                        results['lidar2ego_translation'],
                        results['lidar2ego_rotation'],
                        sweep[sensor]['sensor2global_translation'],
                        sweep[sensor]['sensor2global_rotation'],
                        sweep[sensor]['cam_intrinsic'],
                    ))

        if len(results['sweeps']['next']) == 0:
            print(1, len(results_next['img']) )
            for _ in range(self.next_sweeps_num):
                for j in range(len(cam_types)):
                    results_next['img'].append(results['img'][j])
                    results_next['img_timestamp'].append(results['img_timestamp'][j])
                    results_next['filename'].append(results['filename'][j])
                    results_next['lidar2img'].append(np.copy(results['lidar2img'][j]))
        else:
            choices = [(k + 1) * interval - 1 for k in range(self.next_sweeps_num)]

            for idx in sorted(list(choices)):
                sweep_idx = min(idx, len(results['sweeps']['next']) - 1)
                sweep = results['sweeps']['next'][sweep_idx]

                if len(sweep.keys()) < len(cam_types):
                    sweep = results['sweeps']['next'][sweep_idx - 1]

                for sensor in cam_types:
                    results_next['img'].append(mmcv.imread(sweep[sensor]['data_path'], self.color_type))
                    results_next['img_timestamp'].append(sweep[sensor]['timestamp'] / 1e6)
                    results_next['filename'].append(os.path.relpath(sweep[sensor]['data_path']))
                    results_next['lidar2img'].append(compose_lidar2img(
                        results['ego2global_translation'],
                        results['ego2global_rotation'],
                        results['lidar2ego_translation'],
                        results['lidar2ego_rotation'],
                        sweep[sensor]['sensor2global_translation'],
                        sweep[sensor]['sensor2global_rotation'],
                        sweep[sensor]['cam_intrinsic'],
                    ))

        assert len(results_prev['img']) % 6 == 0
        assert len(results_next['img']) % 6 == 0

        for i in range(len(results_prev['img']) // 6):
            for j in range(6):
                results['img'].append(results_prev['img'][i * 6 + j])
                results['img_timestamp'].append(results_prev['img_timestamp'][i * 6 + j])
                results['filename'].append(results_prev['filename'][i * 6 + j])
                results['lidar2img'].append(results_prev['lidar2img'][i * 6 + j])

            for j in range(6):
                results['img'].append(results_next['img'][i * 6 + j])
                results['img_timestamp'].append(results_next['img_timestamp'][i * 6 + j])
                results['filename'].append(results_next['filename'][i * 6 + j])
                results['lidar2img'].append(results_next['lidar2img'][i * 6 + j])

        return results

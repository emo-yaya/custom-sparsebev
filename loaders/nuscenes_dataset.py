import math
import os
import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
from pyquaternion import Quaternion
import torch


def nuscenes_get_rt_matrix(
    src_sample,
    dest_sample,
    src_mod,
    dest_mod):
    
    """
    CAM_FRONT_XYD indicates going from 2d image coords + depth
        Note that image coords need to multiplied with said depths first to bring it into 2d hom coords.
    CAM_FRONT indicates going from camera coordinates xyz
    
    Method is: whatever the input is, transform to global first.
    """
    possible_mods = ['CAM_FRONT_XYD', 
                     'CAM_FRONT_RIGHT_XYD', 
                     'CAM_FRONT_LEFT_XYD', 
                     'CAM_BACK_XYD', 
                     'CAM_BACK_LEFT_XYD', 
                     'CAM_BACK_RIGHT_XYD',
                     'CAM_FRONT', 
                     'CAM_FRONT_RIGHT', 
                     'CAM_FRONT_LEFT', 
                     'CAM_BACK', 
                     'CAM_BACK_LEFT', 
                     'CAM_BACK_RIGHT',
                     'lidar',
                     'ego',
                     'global']

    assert src_mod in possible_mods and dest_mod in possible_mods
    
    src_lidar_to_ego = np.eye(4, 4)
    src_lidar_to_ego[:3, :3] = Quaternion(src_sample['lidar2ego_rotation']).rotation_matrix
    src_lidar_to_ego[:3, 3] = np.array(src_sample['lidar2ego_translation'])
    
    src_ego_to_global = np.eye(4, 4)
    src_ego_to_global[:3, :3] = Quaternion(src_sample['ego2global_rotation']).rotation_matrix
    src_ego_to_global[:3, 3] = np.array(src_sample['ego2global_translation'])
    
    dest_lidar_to_ego = np.eye(4, 4)
    dest_lidar_to_ego[:3, :3] = Quaternion(dest_sample['lidar2ego_rotation']).rotation_matrix
    dest_lidar_to_ego[:3, 3] = np.array(dest_sample['lidar2ego_translation'])
    
    dest_ego_to_global = np.eye(4, 4)
    dest_ego_to_global[:3, :3] = Quaternion(dest_sample['ego2global_rotation']).rotation_matrix
    dest_ego_to_global[:3, 3] = np.array(dest_sample['ego2global_translation'])
    
    src_mod_to_global = None
    dest_global_to_mod = None
    
    if src_mod == "global":
        src_mod_to_global = np.eye(4, 4)
    elif src_mod == "ego":
        src_mod_to_global = src_ego_to_global
    elif src_mod == "lidar":
        src_mod_to_global = src_ego_to_global @ src_lidar_to_ego
    elif "CAM" in src_mod:
        src_sample_cam = src_sample['cams'][src_mod.replace("_XYD", "")]
        
        src_cam_to_lidar = np.eye(4, 4)
        src_cam_to_lidar[:3, :3] = src_sample_cam['sensor2lidar_rotation']
        src_cam_to_lidar[:3, 3] = src_sample_cam['sensor2lidar_translation']
        
        src_cam_intrinsics = np.eye(4, 4)
        src_cam_intrinsics[:3, :3] = src_sample_cam['cam_intrinsic']
        
        if "XYD" not in src_mod:
            src_mod_to_global = (src_ego_to_global @ src_lidar_to_ego @ 
                                 src_cam_to_lidar)
        else:
            src_mod_to_global = (src_ego_to_global @ src_lidar_to_ego @ 
                                 src_cam_to_lidar @ np.linalg.inv(src_cam_intrinsics))
            
            
    
    if dest_mod == "global":
        dest_global_to_mod = np.eye(4, 4)
    elif dest_mod == "ego":
        dest_global_to_mod = np.linalg.inv(dest_ego_to_global)
    elif dest_mod == "lidar":
        dest_global_to_mod = np.linalg.inv(dest_ego_to_global @ dest_lidar_to_ego)
    elif "CAM" in dest_mod:
        dest_sample_cam = dest_sample['cams'][dest_mod.replace("_XYD", "")]
        
        dest_cam_to_lidar = np.eye(4, 4)
        dest_cam_to_lidar[:3, :3] = dest_sample_cam['sensor2lidar_rotation']
        dest_cam_to_lidar[:3, 3] = dest_sample_cam['sensor2lidar_translation']
        
        dest_cam_intrinsics = np.eye(4, 4)
        dest_cam_intrinsics[:3, :3] = dest_sample_cam['cam_intrinsic']
        
        if "XYD" not in dest_mod:
            dest_global_to_mod = np.linalg.inv(dest_ego_to_global @ dest_lidar_to_ego @ 
                                               dest_cam_to_lidar)
        else:
            dest_global_to_mod = np.linalg.inv(dest_ego_to_global @ dest_lidar_to_ego @ 
                                               dest_cam_to_lidar @ np.linalg.inv(dest_cam_intrinsics))
    
    return dest_global_to_mod @ src_mod_to_global


@DATASETS.register_module()
class CustomNuScenesDataset(NuScenesDataset):

    def __init__(self,
                 ann_file,
                 pipeline=None,
                 data_root=None,
                 classes=None,
                 load_interval=1,
                 with_velocity=True,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 eval_version='detection_cvpr_2019',
                 use_valid_flag=False):
        super().__init__(
            ann_file=ann_file,
            pipeline=pipeline,
            data_root=data_root,
            classes=classes,
            load_interval=load_interval,
            with_velocity=with_velocity,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            eval_version=eval_version,
            use_valid_flag=use_valid_flag)
        
        self.sequences_split_num = 2
        # sequences_split_num splits eacgh sequence into sequences_split_num parts.
        if self.test_mode:
            self.sequences_split_num = 1
        self._set_sequence_group_flag()


    def collect_sweeps(self, index, into_past=60, into_future=60):
        all_sweeps_prev = []
        curr_index = index
        while len(all_sweeps_prev) < into_past:
            curr_sweeps = self.data_infos[curr_index]['sweeps']
            if len(curr_sweeps) == 0:
                break
            all_sweeps_prev.extend(curr_sweeps)
            all_sweeps_prev.append(self.data_infos[curr_index - 1]['cams'])
            curr_index = curr_index - 1
        
        all_sweeps_next = []
        curr_index = index + 1
        while len(all_sweeps_next) < into_future:
            if curr_index >= len(self.data_infos):
                break
            curr_sweeps = self.data_infos[curr_index]['sweeps']
            all_sweeps_next.extend(curr_sweeps[::-1])
            all_sweeps_next.append(self.data_infos[curr_index]['cams'])
            curr_index = curr_index + 1

        return all_sweeps_prev, all_sweeps_next
    
    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """
           
        res = []
        curr_sequence = 0
        for idx in range(len(self.data_infos)):
            sweeps_prev, sweeps_next = self.collect_sweeps(idx)

            if idx != 0 and len(sweeps_prev) == 0:
                # Not first frame and # of sweeps is 0 -> new sequence
                curr_sequence += 1
            res.append(curr_sequence)

        self.flag_fbbev = np.array(res, dtype=np.int64)

        if self.sequences_split_num != 1:
            if self.sequences_split_num == 'all':
                self.flag_fbbev = np.array(range(len(self.data_infos)), dtype=np.int64)
            else:
                bin_counts = np.bincount(self.flag_fbbev)
                new_flags = []
                curr_new_flag = 0
                for curr_flag in range(len(bin_counts)):
                    curr_sequence_length = np.array(
                        list(range(0, 
                                bin_counts[curr_flag], 
                                math.ceil(bin_counts[curr_flag] / self.sequences_split_num)))
                        + [bin_counts[curr_flag]])
                    for sub_seq_idx in (curr_sequence_length[1:] - curr_sequence_length[:-1]):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1

                assert len(new_flags) == len(self.flag_fbbev)
                assert len(np.bincount(new_flags)) == len(np.bincount(self.flag_fbbev)) * self.sequences_split_num
                self.flag_fbbev = np.array(new_flags, dtype=np.int64)

    def get_data_info(self, index):
        info = self.data_infos[index]
        sweeps_prev, sweeps_next = self.collect_sweeps(index)

        ego2global_translation = info['ego2global_translation']
        ego2global_rotation = info['ego2global_rotation']
        lidar2ego_translation = info['lidar2ego_translation']
        lidar2ego_rotation = info['lidar2ego_rotation']
        ego2global_rotation = Quaternion(ego2global_rotation).rotation_matrix
        lidar2ego_rotation = Quaternion(lidar2ego_rotation).rotation_matrix

        input_dict = dict(
            index=index,
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps={'prev': sweeps_prev, 'next': sweeps_next},
            sweeps_fbbev=info['sweeps'],
            timestamp=info['timestamp'] / 1e6,
            ego2global_translation=ego2global_translation,
            ego2global_rotation=ego2global_rotation,
            lidar2ego_translation=lidar2ego_translation,
            lidar2ego_rotation=lidar2ego_rotation,
        )

        if self.modality['use_camera']:
            img_paths = []
            img_timestamps = []
            lidar2img_rts = []
            cam_intrinsics = []
            sensor2ego_translation = []
            sensor2ego_rotation = []
            ego2global_translation = []
            ego2global_rotation = []
            for _, cam_info in info['cams'].items():
                img_paths.append(os.path.relpath(cam_info['data_path']))
                img_timestamps.append(cam_info['timestamp'] / 1e6)

                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T

                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)
                cam_intrinsics.append(intrinsic)
                sensor2ego_translation.append(cam_info['sensor2ego_translation'])
                sensor2ego_rotation.append(cam_info['sensor2ego_rotation'])
                ego2global_translation.append(cam_info['ego2global_translation'])
                ego2global_rotation.append(cam_info['ego2global_rotation'])

            input_dict.update(dict(
                img_filename=img_paths,
                img_timestamp=img_timestamps,
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsics,
                sensor2ego_translation_fbbev = sensor2ego_translation,
                sensor2ego_rotation_fbbev = sensor2ego_rotation,
                ego2global_translation_fbbev=ego2global_translation,
                ego2global_rotation_fbbev=ego2global_rotation,
            ))

        input_dict['sequence_group_idx'] = self.flag_fbbev[index]
        input_dict['start_of_sequence'] = index == 0 or self.flag_fbbev[index - 1] != self.flag_fbbev[index]
        input_dict['curr_to_prev_ego_rt'] = torch.FloatTensor(nuscenes_get_rt_matrix(
            self.data_infos[index], self.data_infos[index - 1],
            "ego", "ego"))
        
        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict

# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion
import os.path as osp
from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet.datasets.builder import PIPELINES
from copy import deepcopy
import cv2
import os
from torchvision.transforms.functional import rotate


def mmlabNormalize(img, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True, debug=False):
    from mmcv.image.photometric import imnormalize
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    to_rgb = to_rgb
    if debug:
        print('warning, debug in mmlabNormalize')
        img = np.asarray(img) # not normalize for visualization
    else:
        img = imnormalize(np.array(img), mean, std, to_rgb)
    img = torch.tensor(img).float().permute(2, 0, 1).contiguous()
    return img



@PIPELINES.register_module()
class PrepareImageInputs(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
        self,
        data_config,
        is_train=False,
        sequential=False,
        ego_cam='CAM_FRONT',

        normalize_cfg=dict(
             mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True, debug=False
        )
    ):
        self.is_train = is_train
        self.data_config = data_config
        self.normalize_img = mmlabNormalize
        self.sequential = sequential
        self.ego_cam = ego_cam
        self.normalize_cfg = normalize_cfg

    def get_rot(self, h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform(self, img, post_rot, post_tran, resize, resize_dims,
                      crop, flip, rotate):
        # adjust image
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img

    def choose_cams(self):
        if self.is_train and self.data_config['Ncams'] < len(
                self.data_config['cams']):
            cam_names = np.random.choice(
                self.data_config['cams'],
                self.data_config['Ncams'],
                replace=False)
        else:
            cam_names = self.data_config['cams']
        return cam_names

    def sample_augmentation(self, H, W, flip=None, scale=None):
        fH, fW = self.data_config['input_size']
        if self.is_train:
            resize = float(fW) / float(W)
            resize += np.random.uniform(*self.data_config['resize'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) *
                         newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])
        else:
            resize = float(fW) / float(W)
            resize += self.data_config.get('resize_test', 0.0)
            if scale is not None:
                resize = scale
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_sensor2ego_transformation(self,
                                      cam_info,
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


    def get_sensor_transforms(self, cam_info, cam_name, index=None):
        try:
            sensor2ego_rotation_fbbev = cam_info['sensor2ego_rotation_fbbev'][index]
            sensor2ego_translation_fbbev = cam_info['sensor2ego_translation_fbbev'][index]
            ego2global_rotation_fbbev = cam_info['ego2global_rotation_fbbev'][index]
            ego2global_translation_fbbev = cam_info['ego2global_translation_fbbev'][index]
        except:
            sensor2ego_rotation_fbbev = cam_info['sensor2ego_rotation_fbbev']
            sensor2ego_translation_fbbev = cam_info['sensor2ego_translation_fbbev']
            ego2global_rotation_fbbev = cam_info['ego2global_rotation_fbbev']
            ego2global_translation_fbbev = cam_info['ego2global_translation_fbbev']
        w, x, y, z = sensor2ego_rotation_fbbev
        # sweep sensor to sweep ego
        sensor2ego_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        sensor2ego_tran = torch.Tensor(
            sensor2ego_translation_fbbev)
        sensor2ego = sensor2ego_rot.new_zeros((4, 4))
        sensor2ego[3, 3] = 1
        sensor2ego[:3, :3] = sensor2ego_rot
        sensor2ego[:3, -1] = sensor2ego_tran
        # sweep ego to global
        w, x, y, z = ego2global_rotation_fbbev
        ego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        ego2global_tran = torch.Tensor(
            ego2global_translation_fbbev)
        ego2global = ego2global_rot.new_zeros((4, 4))
        ego2global[3, 3] = 1
        ego2global[:3, :3] = ego2global_rot
        ego2global[:3, -1] = ego2global_tran
        return sensor2ego, ego2global

    def get_inputs(self, results, scale=None):

        rots = torch.stack(results['rots'])
        trans = torch.stack(results['trans'])
        intrins = torch.stack(results['intrins'])
        post_rots = torch.stack(results['post_rots'])
        post_trans = torch.stack(results['post_trans'])
       
        return (0, rots, trans, intrins, post_rots, post_trans), (0, 0)

    def __call__(self, results):
        results['img_inputs'], results['aux_cam_params'] = self.get_inputs(results)
        return results


@PIPELINES.register_module()
class LoadAnnotationsBEVDepth(object):

    def __init__(self, bda_aug_conf, classes, is_train=True):
        self.bda_aug_conf = bda_aug_conf
        self.is_train = is_train
        self.classes = classes

    def sample_bda_augmentation(self, tta_config=None):
        """Generate bda augmentation values based on bda_config."""
        if self.is_train:
            rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
            scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
            flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
            flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
        else:
            rotate_bda = 0
            scale_bda = 1.0
            if tta_config is not None:
                flip_dx = tta_config['flip_dx']
                flip_dy = tta_config['flip_dy']
            else:
                flip_dx = False
                flip_dy = False

        return rotate_bda, scale_bda, flip_dx, flip_dy

    def bev_transform(self, gt_boxes, rotate_angle, scale_ratio, flip_dx,
                      flip_dy):
        rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                                [0, 0, 1]])
        scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                                  [0, 0, scale_ratio]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0],
                                                [0, 0, 1]])
        if flip_dy:
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0],
                                                [0, 0, 1]])
        rot_mat = flip_mat @ (scale_mat @ rot_mat)
        if gt_boxes.shape[0] > 0:
            gt_boxes[:, :3] = (
                rot_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)
            gt_boxes[:, 3:6] *= scale_ratio
            gt_boxes[:, 6] += rotate_angle
            if flip_dx:
                gt_boxes[:,
                         6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:,
                                                                           6]
            if flip_dy:
                gt_boxes[:, 6] = -gt_boxes[:, 6]
            gt_boxes[:, 7:] = (
                rot_mat[:2, :2] @ gt_boxes[:, 7:].unsqueeze(-1)).squeeze(-1)
        return gt_boxes, rot_mat

    def __call__(self, results):
        # gt_boxes, gt_labels = results['ann_infos']
        gt_boxes, gt_labels = torch.Tensor(np.array([])), torch.tensor(np.array([]))
        tta_confg = results.get('tta_config', None)
        rotate_bda = 0
        scale_bda = 1.0
        flip_dx = False
        flip_dy = False

        # rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation(tta_confg)
        bda_mat = torch.zeros(4, 4)
        bda_mat[3, 3] = 1
        gt_boxes, bda_rot = self.bev_transform(gt_boxes, rotate_bda, scale_bda,
                                               flip_dx, flip_dy)
        bda_mat[:3, :3] = bda_rot

        # bda_rot = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if len(gt_boxes) == 0:
            gt_boxes = torch.zeros(0, 9)
        # results['gt_bboxes_3d'] = \
        #     LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1],
        #                          origin=(0.5, 0.5, 0.5))
        # results['gt_labels_3d'] = gt_labels
        
        
        if self.is_train:
            bda_mat = results['bda_mat']
       
        imgs, rots, trans, intrins = results['img_inputs'][:4]
        post_rots, post_trans = results['img_inputs'][4:]

        results['img_inputs'] = (imgs, rots, trans, intrins, post_rots,
                                 post_trans, bda_rot)
        
        # results['flip_dx'] = flip_dx
        # results['flip_dy'] = flip_dy
        # results['rotate_bda'] = rotate_bda
        # results['scale_bda'] = scale_bda

        return results


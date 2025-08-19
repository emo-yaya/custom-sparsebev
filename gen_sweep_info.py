# Generate info files manually
import os
import mmcv
import tqdm
import pickle
import argparse
import numpy as np
from nuscenes import NuScenes
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box

map_name_from_general_to_detection = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
}
classes = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

parser = argparse.ArgumentParser()
parser.add_argument('--data-root', default='data/nuscenes')
parser.add_argument('--version', default='v1.0-trainval')
args = parser.parse_args()


def get_cam_info(nusc, sample_data):
    pose_record = nusc.get('ego_pose', sample_data['ego_pose_token'])
    cs_record = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
    
    sensor2ego_translation = cs_record['translation']
    ego2global_translation = pose_record['translation']
    sensor2ego_rotation = Quaternion(cs_record['rotation']).rotation_matrix
    ego2global_rotation = Quaternion(pose_record['rotation']).rotation_matrix
    cam_intrinsic = np.array(cs_record['camera_intrinsic'])

    sensor2global_rotation = sensor2ego_rotation.T @ ego2global_rotation.T
    sensor2global_translation = sensor2ego_translation @ ego2global_rotation.T + ego2global_translation

    return {
        'data_path': os.path.join(args.data_root, sample_data['filename']),
        'sensor2global_rotation': sensor2global_rotation,
        'sensor2global_translation': sensor2global_translation,
        'cam_intrinsic': cam_intrinsic,
        'timestamp': sample_data['timestamp'],
        'sensor2ego_translation_fbbev': sensor2ego_translation,
        'sensor2ego_rotation_fbbev': Quaternion(cs_record['rotation']),
        "ego2global_translation_fbbev": ego2global_translation,
        "ego2global_rotation_fbbev": Quaternion(pose_record['rotation'])
    }


def add_sweep_info(nusc, sample_infos):

    token2info = {}
    for info in sample_infos['infos']:
        # info 里常见键名有 'token' 或 'sample_token'
        token2info[info['token']] = info


    for curr_id in tqdm.tqdm(range(len(sample_infos['infos']))):
        sample = nusc.get('sample', sample_infos['infos'][curr_id]['token'])

        scene = nusc.get('scene',  sample['scene_token'])
        sample_infos['infos'][curr_id]['scene_name'] = scene['name']
        
        cam_types = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'
        ]

        curr_cams = dict()
        for cam in cam_types:
            curr_cams[cam] = nusc.get('sample_data', sample['data'][cam])

        for cam in cam_types:
            sample_data = nusc.get('sample_data', sample['data'][cam])
            sweep_cam = get_cam_info(nusc, sample_data)
            sample_infos['infos'][curr_id]['cams'][cam].update(sweep_cam)

        # remove unnecessary
        for cam in cam_types:
            del sample_infos['infos'][curr_id]['cams'][cam]['sample_data_token']
            # del sample_infos['infos'][curr_id]['cams'][cam]['sensor2ego_translation']
            # del sample_infos['infos'][curr_id]['cams'][cam]['sensor2ego_rotation']
            # del sample_infos['infos'][curr_id]['cams'][cam]['ego2global_translation']
            # del sample_infos['infos'][curr_id]['cams'][cam]['ego2global_rotation']

        sweep_infos = []
        sweep_infos_gt = []
        sample_cur = sample
        if sample['prev'] != '':  # add sweep frame between two key frame
            for idx in range(5):
                sweep_info = dict()
                for cam in cam_types:
                    if curr_cams[cam]['prev'] == '':
                        sweep_info = sweep_infos[-1]
                        break
                    sample_data = nusc.get('sample_data', curr_cams[cam]['prev'])
                    sweep_cam = get_cam_info(nusc, sample_data)
                    curr_cams[cam] = sample_data
                    sweep_info[cam] = sweep_cam
                if sample_cur['prev'] == '':
                    sweep_info_gt = sweep_infos_gt[-1]
                else:
                    sample_prev = nusc.get('sample', sample_cur['prev'])
                    sweep_info_gt = token2info.get(sample_prev['token'])['gt_boxes']
                    sample_cur = sample_prev
                    
                sweep_infos_gt.append(sweep_info_gt)
                sweep_infos.append(sweep_info)

        sample_infos['infos'][curr_id]['sweeps'] = sweep_infos
        sample_infos['infos'][curr_id]['sweeps_gts'] = sweep_infos_gt

    return sample_infos

def get_gt(info):
    """Generate gt labels from info.

    Args:
        info(dict): Infos needed to generate gt labels.

    Returns:
        Tensor: GT bboxes.
        Tensor: GT labels.
    """
    ego2global_rotation = info['cams']['CAM_FRONT']['ego2global_rotation']
    ego2global_translation = info['cams']['CAM_FRONT'][
        'ego2global_translation']
    trans = -np.array(ego2global_translation)
    rot = Quaternion(ego2global_rotation).inverse
    gt_boxes = list()
    gt_labels = list()
    for ann_info in info['ann_infos']:
        # Use ego coordinate.
        if (map_name_from_general_to_detection[ann_info['category_name']]
                not in classes
                or ann_info['num_lidar_pts'] + ann_info['num_radar_pts'] <= 0):
            continue
        box = Box(
            ann_info['translation'],
            ann_info['size'],
            Quaternion(ann_info['rotation']),
            velocity=ann_info['velocity'],
        )
        box.translate(trans)
        box.rotate(rot)
        box_xyz = np.array(box.center)
        box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
        box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
        box_velo = np.array(box.velocity[:2])
        gt_box = np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])
        gt_boxes.append(gt_box)
        gt_labels.append(
            classes.index(
                map_name_from_general_to_detection[ann_info['category_name']]))
    return gt_boxes, gt_labels


def add_ann_adj_info(nuscenes):
    # nuscenes_version = 'v1.0-trainval'
    # dataroot = './data/nuscenes/'
    # nuscenes = NuScenes(nuscenes_version, dataroot)
    for set in ['train', 'val']:
        dataset = pickle.load(
            open('./data/nuscenes/nuscenes_infos_%s_sweep.pkl' % (set), 'rb'))
        for id in range(len(dataset['infos'])):
            if id % 10 == 0:
                print('%d/%d' % (id, len(dataset['infos'])))
            info = dataset['infos'][id]
            # get sweep adjacent frame info
            sample = nuscenes.get('sample', info['token'])
            ann_infos = list()
            for ann in sample['anns']:
                ann_info = nuscenes.get('sample_annotation', ann)
                velocity = nuscenes.box_velocity(ann_info['token'])
                if np.any(np.isnan(velocity)):
                    velocity = np.zeros(3)
                ann_info['velocity'] = velocity
                ann_infos.append(ann_info)
            dataset['infos'][id]['ann_infos'] = ann_infos
            dataset['infos'][id]['ann_infos'] = get_gt(dataset['infos'][id])
            dataset['infos'][id]['scene_token'] = sample['scene_token']

            scene = nuscenes.get('scene', sample['scene_token'])
            dataset['infos'][id]['occ_path'] = \
                './data/nuscenes/gts/%s/%s'%(scene['name'], info['token'])
        with open('./data/nuscenes/nuscenes_infos_%s_sweep.pkl' % (set), 'wb') as fid:
            pickle.dump(dataset, fid)

if __name__ == '__main__':
    nusc = NuScenes(args.version, args.data_root)

    if args.version == 'v1.0-trainval':
        sample_infos = pickle.load(open(os.path.join(args.data_root, 'nuscenes_infos_train.pkl'), 'rb'))
        sample_infos = add_sweep_info(nusc, sample_infos)
        mmcv.dump(sample_infos, os.path.join(args.data_root, 'nuscenes_infos_train_sweep.pkl'))

        sample_infos = pickle.load(open(os.path.join(args.data_root, 'nuscenes_infos_val.pkl'), 'rb'))
        sample_infos = add_sweep_info(nusc, sample_infos)
        mmcv.dump(sample_infos, os.path.join(args.data_root, 'nuscenes_infos_val_sweep.pkl'))
        # add_ann_adj_info(nusc)

    elif args.version == 'v1.0-test':
        sample_infos = pickle.load(open(os.path.join(args.data_root, 'nuscenes_infos_test.pkl'), 'rb'))
        sample_infos = add_sweep_info(nusc, sample_infos)
        mmcv.dump(sample_infos, os.path.join(args.data_root, 'nuscenes_infos_test_sweep.pkl'))

    elif args.version == 'v1.0-mini':
        sample_infos = pickle.load(open(os.path.join(args.data_root, 'nuscenes_infos_train.pkl'), 'rb'))
        sample_infos = add_sweep_info(nusc, sample_infos)
        mmcv.dump(sample_infos, os.path.join(args.data_root, 'nuscenes_infos_train_mini_sweep.pkl'))

        sample_infos = pickle.load(open(os.path.join(args.data_root, 'nuscenes_infos_val.pkl'), 'rb'))
        sample_infos = add_sweep_info(nusc, sample_infos)
        mmcv.dump(sample_infos, os.path.join(args.data_root, 'nuscenes_infos_val_mini_sweep.pkl'))

    else:
        raise ValueError

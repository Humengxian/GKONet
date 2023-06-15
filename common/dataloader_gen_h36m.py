import numpy as np
import torch


import torch.utils.data as data
import numpy as np

from common.camera import *
from common.h36m_dataset import Human36mDataset
from common.generators_h36m import ChunkedGenerator
from common.utils import *

class Fusion(data.Dataset):
    def __init__(self, generator, num_joint_dim=5):
        self.generator = generator
        self.num_joint_dim=num_joint_dim
        
    def __len__(self):
        return len(self.generator.pairs)

    def get_vector(self, keypoints, cam=None, add_norm=True):
        if not add_norm:
            # print(keypoints.shape)
            keypoints[1:] += keypoints[:1] 
        joint_vector_list = []
        J, d = keypoints.shape
        for i in range(keypoints.shape[0]):
            joint_vector = (keypoints[i, :].reshape(1, keypoints.shape[1]) - keypoints.copy())
            if add_norm:
                f = np.mean(cam[:2])
                # cosine
                OD = np.array([0, 0, f]).reshape(1, 3) - np.concatenate((keypoints, np.zeros((J, 1))), axis=1)
                DC = np.concatenate((keypoints[i, :].reshape(1, d) - keypoints.copy(), np.zeros((J, 1))), axis=1)
                cos_ODC = - np.sum(OD * DC, axis=1) / ((np.linalg.norm(OD, axis=1) + 1e-4) * (np.linalg.norm(DC, axis=1) + 1e-4))
                cos_ODC[i] = 0
                cos_ODC = cos_ODC.reshape(J, 1)
                
                # OD length
                OD_l = np.linalg.norm(OD, axis=1, keepdims=True)
                
                # CD length
                CD = np.linalg.norm(joint_vector, axis=1, keepdims=True)
                
                joint_vector = np.concatenate((joint_vector, CD), axis=1)
                joint_vector = np.concatenate((joint_vector, OD_l), axis=1)
                joint_vector = np.concatenate((joint_vector, cos_ODC), axis=1)

            joint_vector_list.append(joint_vector)
        joint_vector = np.float32(np.array(joint_vector_list))
        return joint_vector

    def __getitem__(self, index):
        cam, gt_3D, input_2D, action = self.generator.get_batch(index)
        
        # 2d input
        joint_vector_2d = self.get_vector(input_2D[0].copy(), cam)

        # 3d input
        joint_vector_3d = self.get_vector(gt_3D[0].copy(), add_norm=False)

        return cam, gt_3D[0], input_2D[0], joint_vector_3d, joint_vector_2d, action

def get_data(args, logger):
    dataset_path = 'data/data_3d_' + args.dataset + '.npz'
    logger.info(f"Load 3d data: {dataset_path}")

    if args.dataset == 'h36m':
        dataset = Human36mDataset(dataset_path)         # 基操
    else:
        raise KeyError('Invalid dataset')

    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]

            if 'positions' in anim:
                positions_3d = []
                for cam in anim['cameras']:
                    pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                    pos_3d[:, 1:] -= pos_3d[:, :1] # Remove global offset, but keep trajectory in first position
                    positions_3d.append(pos_3d)
                anim['positions_3d'] = positions_3d

    keypoints = np.load('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz', allow_pickle=True)
    logger.info(f"Load 3d data: " + 'data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz')
    keypoints_metadata = keypoints['metadata'].item()
    keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
        
    keypoints = keypoints['positions_2d'].item()

    for subject in dataset.subjects():
        assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
        for action in dataset[subject].keys():
            assert action in keypoints[subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action, subject)
            if 'positions_3d' not in dataset[subject][action]:
                continue

            for cam_idx in range(len(keypoints[subject][action])):
                mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
                assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length

                if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                    keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

            assert len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d'])

    for subject in keypoints.keys():
        for action in keypoints[subject]:
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                cam = dataset.cameras()[subject][cam_idx]
                kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                if args.num_joints == 16:
                    kps = np.concatenate((kps[:, :9], kps[:, 10:]), axis=1)
                keypoints[subject][action][cam_idx] = kps

    subjects_train = args.subjects_train.split(',')
    subjects_test = args.subjects_test.split(',')

    def fetch(subjects, action_filter=None, subset=1, parse_3d_poses=True):
        out_poses_3d = {}
        out_poses_2d = {}
        out_camera_params = {}

        for subject in subjects:
            for action in keypoints[subject].keys():
                poses_2d = keypoints[subject][action]

                for i in range(len(poses_2d)):
                    out_poses_2d[(subject, action, i)] = poses_2d[i]

                if subject in dataset.cameras():
                    cams = dataset.cameras()[subject]
                    assert len(cams) == len(poses_2d), 'Camera count mismatch'
                    for i, cam in enumerate(cams):
                        if 'intrinsic' in cam:
                            out_camera_params[(subject, action, i)] = cam['intrinsic']

                if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                    poses_3d = dataset[subject][action]['positions_3d']
                    assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                    for i in range(len(poses_3d)): 
                        out_poses_3d[(subject, action, i)] = poses_3d[i]

        if len(out_camera_params) == 0:
            out_camera_params = None
        if len(out_poses_3d) == 0:
            out_poses_3d = None

        return out_camera_params, out_poses_3d, out_poses_2d

    action_filter = None if args.actions == '*' else args.actions.split(',')
    if action_filter is not None:
        print('Selected actions:', action_filter)

    pad = 0 # Padding on each side
    causal_shift = 0
    cameras_valid, poses_valid, poses_valid_2d = fetch(subjects_test, action_filter)

    test_generator = ChunkedGenerator(args.batchsize//args.stride, cameras_valid, poses_valid, poses_valid_2d, args.stride,
                                        causal_shift=causal_shift, shuffle=False, augment=False,
                                        kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
   
    cameras_train, poses_train, poses_train_2d = fetch(subjects_train, action_filter, subset=args.subset)
    train_generator = ChunkedGenerator(args.batchsize//args.stride, cameras_train, poses_train, poses_train_2d, args.stride,
                                        causal_shift=causal_shift, shuffle=True, augment=args.data_augmentation,
                                        kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)

    train_set = Fusion(train_generator, num_joint_dim=args.num_joint_dim)
    train_dataloader = torch.utils.data.DataLoader(train_set,
                                                batch_size=args.batchsize,
                                                shuffle=True,
                                                num_workers=args.num_workers,
                                                pin_memory=True)

    test_set = Fusion(test_generator, num_joint_dim=args.num_joint_dim)
    test_dataloader = torch.utils.data.DataLoader(test_set,
                                                batch_size=args.batchsize,
                                                shuffle=True,
                                                num_workers=args.num_workers,
                                                pin_memory=True)

    logger.info(f"Train data num: {len(train_set)}")
    logger.info(f"Test data num: {len(test_set)}")
    
    return train_dataloader, test_dataloader, kps_left, kps_right, joints_left, joints_right



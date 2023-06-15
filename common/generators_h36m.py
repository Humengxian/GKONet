# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from itertools import zip_longest
import numpy as np


class ChunkedGenerator:
    """
    Batched data generator, used for training.
    The sequences are split into equal-length chunks and padded as necessary.
    
    Arguments:
    batch_size -- the batch size to use for training
    cameras -- list of cameras, one element for each video (optional, used for semi-supervised training)
    poses_3d -- list of ground-truth 3D poses, one element for each video (optional, used for supervised training)
    poses_2d -- list of input 2D keypoints, one element for each video
    chunk_length -- number of output frames to predict for each training example (usually 1)
    pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
    causal_shift -- asymmetric padding offset when causal convolutions are used (usually 0 or "pad")
    shuffle -- randomly shuffle the dataset before each epoch
    random_seed -- initial seed to use for the random generator
    augment -- augment the dataset by flipping poses horizontally
    kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
    joints_left and joints_right -- list of left/right 3D joints if flipping is enabled
    """
    def __init__(self, batch_size, cameras, poses_3d, poses_2d,
                 pad=0, causal_shift=0,
                 shuffle=True, random_seed=1234,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None,
                 endless=False):
        assert poses_3d is None or len(poses_3d) == len(poses_2d), (len(poses_3d), len(poses_2d))
        assert cameras is None or len(cameras) == len(poses_2d)
    
        # Build lineage info
        pairs = [] # (seq_idx, start_frame, end_frame, flip) tuples
        start_index = 0
        self.saved_index = {}
        for key in poses_2d.keys():
            assert poses_3d is None or poses_3d[key].shape[0] == poses_3d[key].shape[0]
            n_chunks = poses_2d[key].shape[0]
            bounds = np.arange(n_chunks + 1)
            augment_vector = np.full(len(bounds - 1), False, dtype=bool)
            reverse_augment_vector = np.full(len(bounds - 1), False, dtype=bool)
            if isinstance(key, tuple):
                keys = np.tile(np.array(key).reshape([1, 3]),(len(bounds - 1),1))
            else:
                keys = np.tile(np.array(key).reshape([1, 1]),(len(bounds - 1),1))
            pairs += list(zip(keys, bounds[:-1], bounds[1:], augment_vector, reverse_augment_vector))
            if augment:
                pairs += list(zip(keys, bounds[:-1], bounds[1:], ~augment_vector, reverse_augment_vector))

            end_index = start_index + poses_3d[key].shape[0]
            self.saved_index[key] = [start_index,end_index]
            start_index = start_index + poses_3d[key].shape[0]

        self.num_batches = (len(pairs) + batch_size - 1) // batch_size
        self.batch_size = batch_size
        self.random = np.random.RandomState(random_seed)
        self.pairs = pairs
        self.shuffle = shuffle
        self.pad = pad
        self.causal_shift = causal_shift
        self.endless = endless
        self.state = None
        
        self.cameras = cameras
        self.poses_3d = poses_3d
        self.poses_2d = poses_2d
        
        self.augment = augment
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right
        
    def num_frames(self):
        return self.num_batches * self.batch_size
    
    def random_state(self):
        return self.random
    
    def set_random_state(self, random):
        self.random = random
        
    def augment_enabled(self):
        return self.augment
    
    def next_pairs(self):
        if self.state is None:
            if self.shuffle:
                pairs = self.random.permutation(self.pairs)
            else:
                pairs = self.pairs
            return 0, pairs
        else:
            return self.state
    
    def get_batch(self, index):
        # start_idx, pairs = self.next_pairs()
        seq_i, start_3d, end_3d, flip, reverse = self.pairs[index]
        
        if len(seq_i) == 3:
            subject, action, cam_index = seq_i
            seq_i = (subject, action, int(cam_index))
        else:
            seq_i = seq_i[0]
            action = 0
            cam_index = 0
            
        start_2d = start_3d
        end_2d = end_3d
        i= 0
        # 2D poses
        seq_2d = self.poses_2d[seq_i].copy()
        low_2d = max(start_2d, 0)
        high_2d = min(end_2d, seq_2d.shape[0])
        batch_2d = seq_2d[low_2d:high_2d]

        if flip:
            # Flip 2D keypoints
            batch_2d[:, :, 0] *= -1
            batch_2d[:, self.kps_left + self.kps_right] = batch_2d[:, self.kps_right + self.kps_left]

        # 3D poses
        if self.poses_3d is not None:
            seq_3d = self.poses_3d[seq_i]
            low_3d = max(start_3d, 0)
            high_3d = min(end_3d, seq_3d.shape[0])
            batch_3d = seq_3d[low_3d:high_3d]

            if flip:
                # Flip 3D joints
                batch_3d[:, :, 0] *= -1
                batch_3d[:, self.joints_left + self.joints_right] = \
                        batch_3d[:, self.joints_right + self.joints_left]

        # Cameras
        if self.cameras is not None:
            batch_cam = self.cameras[seq_i]
            if flip:
                # Flip horizontal distortion coefficients
                batch_cam[2] *= -1
                batch_cam[7] *= -1
        else:
            batch_cam = 0

        return batch_cam, batch_3d, batch_2d, action


class UnchunkedGenerator:
    """
    Non-batched data generator, used for testing.
    Sequences are returned one at a time (i.e. batch size = 1), without chunking.
    
    If data augmentation is enabled, the batches contain two sequences (i.e. batch size = 2),
    the second of which is a mirrored version of the first.
    
    Arguments:
    cameras -- list of cameras, one element for each video (optional, used for semi-supervised training)
    poses_3d -- list of ground-truth 3D poses, one element for each video (optional, used for supervised training)
    poses_2d -- list of input 2D keypoints, one element for each video
    pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
    causal_shift -- asymmetric padding offset when causal convolutions are used (usually 0 or "pad")
    augment -- augment the dataset by flipping poses horizontally
    kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
    joints_left and joints_right -- list of left/right 3D joints if flipping is enabled
    """
    
    def __init__(self, cameras, poses_3d, poses_2d, pad=0, causal_shift=0,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None):
        assert poses_3d is None or len(poses_3d) == len(poses_2d)
        assert cameras is None or len(cameras) == len(poses_2d)

        self.augment = False
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right
        
        self.pad = pad
        self.causal_shift = causal_shift
        self.cameras = [] if cameras is None else cameras
        self.poses_3d = [] if poses_3d is None else poses_3d
        self.poses_2d = poses_2d
    
    def augment_enabled(self):
        return self.augment
    
    def set_augment(self, augment):
        self.augment = augment
    
    def next_epoch(self):
        for seq_cam, seq_3d, seq_2d in zip_longest(self.cameras, self.poses_3d, self.poses_2d):
            batch_cam = None if seq_cam is None else np.expand_dims(seq_cam, axis=0)
            batch_3d = None if seq_3d is None else np.expand_dims(seq_3d, axis=0)
            batch_2d = np.expand_dims(np.pad(seq_2d,
                            ((self.pad + self.causal_shift, self.pad - self.causal_shift), (0, 0), (0, 0)),
                            'edge'), axis=0)
            if self.augment:
                # Append flipped version
                if batch_cam is not None:
                    batch_cam = np.concatenate((batch_cam, batch_cam), axis=0)
                    batch_cam[1, 2] *= -1
                    batch_cam[1, 7] *= -1
                
                if batch_3d is not None:
                    batch_3d = np.concatenate((batch_3d, batch_3d), axis=0)
                    batch_3d[1, :, :, 0] *= -1
                    batch_3d[1, :, self.joints_left + self.joints_right] = batch_3d[1, :, self.joints_right + self.joints_left]

                batch_2d = np.concatenate((batch_2d, batch_2d), axis=0)
                batch_2d[1, :, :, 0] *= -1
                batch_2d[1, :, self.kps_left + self.kps_right] = batch_2d[1, :, self.kps_right + self.kps_left]

            yield batch_cam, batch_3d, batch_2d
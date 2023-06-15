import os
import pandas as pd
import numpy as np
import torch
from rich.progress import (BarColumn, Progress, TextColumn, TimeElapsedColumn,
                           TimeRemainingColumn)

def fine_tune(fine_pose, predicted_3d_joint, predicted_3d_joint_filp):
    joint_embedding = (predicted_3d_joint + predicted_3d_joint_filp) / 2
    joint_vector_filp = -torch.transpose(joint_embedding, 1, 2) 
    predicted_3d_joint = (joint_embedding + joint_vector_filp) / 2
    fine_pose[:, 6] = (fine_pose[:, 5] - predicted_3d_joint[:, 5, 6] + fine_pose[:, 6]) / 2
    fine_pose[:, 3] = (fine_pose[:, 2] - predicted_3d_joint[:, 2, 3] + fine_pose[:, 3]) / 2
    fine_pose[:, 16] = (fine_pose[:, 15] - predicted_3d_joint[:, 15, 16] + fine_pose[:, 16]) / 2
    fine_pose[:, 13] = (fine_pose[:, 12] - predicted_3d_joint[:, 12, 13] + fine_pose[:, 13]) / 2
    
    return fine_pose

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class Trainer(object):
    def __init__(self, model, experiment, kps_left, kps_right, joints_left, joints_right, num_joints, eval=False) -> None:
        self.model = model
        self.experiment = experiment
        self.kps_left, self.kps_right, self.joints_left, self.joints_right = kps_left, kps_right, joints_left, joints_right
        self.actions = ["Directions", "Discussion", "Eating", "Greeting",
                        "Phoning", "Photo", "Posing", "Purchases", "SittingDown",
                        "Sitting", "Smoking", "Waiting",
                        "WalkDog", "Walking", "WalkTogether"]
        self.best_epoch_p1 = 10000
        self.best_epoch_p2 = 10000
        
        self.best_epoch_p1_test = 10000

        self.pre_act_p1 = None
        self.pre_act_p2 = None
        
        self.num_joints = num_joints
        self.eval_ = eval

    def train(self, epochs, train_dataloader, test_dataloader, optimizer, lr_decay, logger):
        for epoch in range(epochs):
            logger.info(f"================== model train info <epoch: {epoch:3d} > ==================")
            logger.info(f"lr: {optimizer.param_groups[0]['lr']:.7f}")

            train_p1, train_p2, _, _ =\
                self.train_one_epoch(train_dataloader, optimizer)
            test_p1, test_p2, pre_act_p1, pre_act_p2= \
                self.test_one_epoch(test_dataloader)

            if test_p1 < self.best_epoch_p1:
                self.best_epoch_p1 = test_p1
                self.best_epoch_p2 = test_p2
                self.save(test_p1, test_p2)
                self.pre_act_p1 = pre_act_p1
                self.pre_act_p2 = pre_act_p2

            logger.info(f"train p1: {train_p1:.4f}, test p1:  {test_p1:.4f}.")
            logger.info(f"train p2: {train_p2:.4f}, test p2: {test_p2:.4f}.")
            logger.critical(f"now model best test p1: {self.best_epoch_p1:.4f};")
            logger.info(f"====================================================================")
            
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay

    def eval(self, test_dataloader, logger):
        p1_root, test_pose_p1, test_pose_p2, pre_act_p1, pre_act_p2= \
                self.test_one_epoch(test_dataloader)
        self.best_epoch_p1 = test_pose_p1
        self.best_epoch_p2 = test_pose_p2
        self.pre_act_p1 = pre_act_p1
        self.pre_act_p2 = pre_act_p2

        logger.info(f"test p1:  {test_pose_p1:.4f} test p2:  {test_pose_p2:.4f} test root p1: {p1_root:.4f}")

    def train_one_epoch(self, train_dataloader, optimizer):
        mpjpe_p1 = {}
        mpjpe_p2 = {}
        for act in self.actions:
            mpjpe_p1[act] = AverageMeter('p1_%s' % act, '.4f')
            mpjpe_p2[act] = AverageMeter('p2_%s' % act, '.4f')

        self.model.train()
        with Progress(TextColumn("[progress.description]{task.description}"),
                      BarColumn(),
                      TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                      TimeRemainingColumn(),
                      TimeElapsedColumn()) as progress:
            epoch_tqdm = progress.add_task(description="TRAIN", total=len(train_dataloader))

            for cameras_train, inputs_3d, inputs_2d, joint_vector_3d, joint_vector_2d, action in train_dataloader:

                inputs_3d = inputs_3d.cuda()
                inputs_2d = inputs_2d.cuda()
                joint_vector_2d = joint_vector_2d.cuda()
                joint_vector_3d = joint_vector_3d.cuda()
                cameras_train = cameras_train
                inputs_3d[:, 0] = 0
                optimizer.zero_grad()

                _, predicted_3d_joint, predicted_3d_fine_pose = self.model(inputs_2d, joint_vector_2d)
                predicted_3d_joint = predicted_3d_joint.reshape(predicted_3d_joint.shape[0], self.num_joints * self.num_joints, 3)
                joint_vector_3d = joint_vector_3d.reshape(predicted_3d_joint.shape[0], self.num_joints * self.num_joints, 3)
                loss_3d_joint = self.cal_loss(predicted_3d_joint, joint_vector_3d)
                # loss_3d_joint = self.cal_weight_loss(predicted_3d_fine_pose, inputs_3d, inputs_2d)
                loss_3d_joint.backward()
                optimizer.step()
                
                predicted_3d_fine_pose[:, 0] = 0
                mpjpe_p1 = self.calculate_mpjpe_p1(predicted_3d_fine_pose, inputs_3d, action, self.actions, mpjpe_p1)
                mpjpe_p2 = self.calculate_mpjpe_p2(predicted_3d_fine_pose, inputs_3d, action, self.actions, mpjpe_p2)
     
                progress.advance(epoch_tqdm, advance=1)

        p1_avg = []
        p2_avg = []
        pre_act_p1 = {'method': ['our']}
        pre_act_p2 = {'method': ['our']}
        for act in self.actions:
            p1_avg.append(mpjpe_p1[act].avg * 1000)
            p2_avg.append(mpjpe_p2[act].avg * 1000)
            pre_act_p1[act] = [mpjpe_p1[act].avg * 1000]
            pre_act_p2[act] = [mpjpe_p2[act].avg * 1000]

        return np.mean(p1_avg), np.mean(p2_avg), pre_act_p1, pre_act_p2

    def test_one_epoch(self, test_dataloader):
        mpjpe_p1 = {}
        mpjpe_p2 = {}
        mpjpe_root = {}
        for act in self.actions:
            mpjpe_p1[act] = AverageMeter('p1_%s' % act, '.4f')
            mpjpe_p2[act] = AverageMeter('p2_%s' % act, '.4f')
            mpjpe_root[act] = AverageMeter('p2_%s' % act, '.4f')

        self.model.eval()
        with torch.no_grad():
            with Progress(TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TimeRemainingColumn(),
                    TimeElapsedColumn()) as progress:
                epoch_tqdm = progress.add_task(description=" TEST", total=len(test_dataloader))

                for cameras_test, inputs_3d, inputs_2d, joint_vector_3d, joint_vector_2d, action in test_dataloader:
                    inputs_3d = inputs_3d.cuda()
                    inputs_2d = inputs_2d.cuda()
                    joint_vector_2d = joint_vector_2d.cuda()
                    joint_vector_3d = joint_vector_3d.cuda()
                    cameras_test = cameras_test.cuda()
                    
                    # only test root in eval time
                    if self.eval_:
                        inputs_root = inputs_3d[:, :1].clone()
                        
                    inputs_3d[:, 0] = 0

                    inputs_2d_flip = inputs_2d.clone()
                    inputs_2d_flip[:, :, 0] *= -1
                    inputs_2d_flip[:, self.kps_left + self.kps_right] = inputs_2d_flip[:, self.kps_right + self.kps_left]

                    joint_vector_2d_flip = joint_vector_2d.clone()
                    joint_vector_2d_flip[:, :, :, 0] *= -1
                    joint_vector_2d_flip[:, :, self.kps_left + self.kps_right, :] = joint_vector_2d_flip[:, :, self.kps_right + self.kps_left, :]
                    joint_vector_2d_flip[:, self.kps_left + self.kps_right, :, :] = joint_vector_2d_flip[:, self.kps_right + self.kps_left, :, :]

                    _, _, predicted_3d_fine_pose = self.model(inputs_2d, joint_vector_2d)
                    _, _, predicted_3d_fine_pose_filp = self.model(inputs_2d_flip, joint_vector_2d_flip)
                    
                    predicted_3d_fine_pose_filp[:, :, 0] *= -1
                    predicted_3d_fine_pose_filp[:, self.joints_left + self.joints_right] = predicted_3d_fine_pose_filp[:, self.joints_right + self.joints_left]
                    predicted_3d_pos = (predicted_3d_fine_pose_filp + predicted_3d_fine_pose) / 2
                    
                    predicted_3d_pos[:, 0] = 0
                    mpjpe_p1 = self.calculate_mpjpe_p1(predicted_3d_pos, inputs_3d, action, self.actions, mpjpe_p1)           
                    mpjpe_p2 = self.calculate_mpjpe_p2(predicted_3d_pos, inputs_3d, action, self.actions, mpjpe_p2)

                    if self.eval_:
                        root = self.cal_root(predicted_3d_pos, inputs_2d, cameras_test, k=4)
                        mpjpe_root = self.calculate_mpjpe_p1(root.unsqueeze(1), inputs_root, action, self.actions, mpjpe_root)
                        
                    progress.advance(epoch_tqdm, advance=1)

        p1_avg = []
        p2_avg = []
        pre_act_p1 = {'method': ['our']}
        pre_act_p2 = {'method': ['our']}
        for act in self.actions:
            p1_avg.append(mpjpe_p1[act].avg * 1000)
            p2_avg.append(mpjpe_p2[act].avg * 1000)
            pre_act_p1[act] = [mpjpe_p1[act].avg * 1000]
            pre_act_p2[act] = [mpjpe_p2[act].avg * 1000]

        p1_root_avg = []
        for act in self.actions:
            p1_root_avg.append(mpjpe_root[act].avg * 1000)
            
        if not self.eval_:
            return np.mean(p1_avg), np.mean(p2_avg), pre_act_p1, pre_act_p2
        else:
            return np.mean(p1_root_avg), np.mean(p1_avg), np.mean(p2_avg), pre_act_p1, pre_act_p2

    def cal_loss(self, predicted, target):
        assert predicted.shape == target.shape
        return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))
    
    @staticmethod
    def calculate_mpjpe_p1(predicted, target, action, actions, mpjpe_p1):
        assert target.shape == predicted.shape
        dist = torch.mean(torch.norm(predicted - target, dim=len(predicted.shape) - 1), dim=len(target.shape) - 2)
        for i, act in enumerate(action):
            find_act = False
            for a in actions:
                if a in act:
                    mpjpe_p1[a].update(dist[i].item(), 1)
                    find_act = True
                    break
            assert find_act is True, f'{act} not find'

        return mpjpe_p1

    @staticmethod
    def calculate_mpjpe_p2(pose_3d_predict, pose_3d_target, action, actions, mpjpe_p2):
        assert pose_3d_target.shape == pose_3d_predict.shape
        n = pose_3d_predict.size(0)
        pose_3d_predict = pose_3d_predict.detach().cpu().numpy().reshape(-1, pose_3d_predict.shape[-2], pose_3d_predict.shape[-1])
        pose_3d_target = pose_3d_target.detach().cpu().numpy().reshape(-1, pose_3d_target.shape[-2], pose_3d_target.shape[-1])

        muX = np.mean(pose_3d_target, axis=1, keepdims=True)
        muY = np.mean(pose_3d_predict, axis=1, keepdims=True)

        X0 = pose_3d_target - muX
        Y0 = pose_3d_predict - muY

        normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
        normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

        X0 /= normX
        Y0 /= normY

        H = np.matmul(X0.transpose(0, 2, 1), Y0)
        U, s, Vt = np.linalg.svd(H)
        V = Vt.transpose(0, 2, 1)
        R = np.matmul(V, U.transpose(0, 2, 1))

        sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
        V[:, :, -1] *= sign_detR
        s[:, -1] *= sign_detR.flatten()
        R = np.matmul(V, U.transpose(0, 2, 1))

        tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

        a = tr * normX / normY
        t = muX - a * np.matmul(muY, R)

        predicted_aligned = a * np.matmul(pose_3d_predict, R) + t
        dist = np.mean(np.linalg.norm(predicted_aligned - pose_3d_target, axis=len(pose_3d_target.shape) - 1), axis=len(pose_3d_target.shape) - 2)
        for i, act in enumerate(action):
            find_act = False
            for a in actions:
                if a in act:
                    mpjpe_p2[a].update(dist[i].item(), 1)
                    find_act = True
                    break
            assert find_act is True, f'{act} not find'

        return mpjpe_p2

    def save(self, p1, p2):
        if not os.path.exists(f"experiment/{self.experiment}/model_para"):
            os.mkdir(f"experiment/{self.experiment}/model_para")

        save_path = f"experiment/{self.experiment}/model_para/model_full_para.pth"
        torch.save(obj={'para': self.model.state_dict(), 'p1': p1, 'p2': p2}, f=save_path)

    @staticmethod
    def cal_root(pose3d, pose2d, cam, k=4, inputs_root=None):
        b, j, _ = pose3d.shape
        pose2d = pose2d - cam[:, 2:4].reshape(b, 1, 2).repeat(1, j, 1)
        f = torch.mean(cam[:, :2], dim=-1, keepdim=True)

        pose2d = torch.cat((pose2d, torch.zeros((b, 17, 1), device='cuda:0', requires_grad=False)), dim=-1)
        z2_1 = pose3d[:, :, 2]
        
        org = torch.zeros((b, j, 3), device='cuda:0', requires_grad=False)
        org[:, :, 2] = torch.mean(cam[:, :2].reshape(b, 1, 2).repeat(1, j, 1), dim=-1)
        DO_v = org - pose2d
        OD_l = torch.norm(DO_v, dim=-1)
        OC_v = pose2d[:, 0].reshape(b, 1, 3).repeat(1, j, 1) - pose2d
        cos_alpha = -torch.sum(DO_v * OC_v, dim=-1) / (torch.norm(DO_v, dim=-1) * torch.norm(OC_v, dim=-1))
        AB = torch.norm(pose3d, dim=-1)
        BE = z2_1 * OD_l / f
        AE = BE * cos_alpha + torch.sqrt(BE ** 2 * cos_alpha ** 2 - (BE) ** 2 + AB **2)
        CD = torch.norm(pose2d[:, 0, :2].reshape(b, 1, 2).repeat(1, j, 1) - pose2d[:, :, :2], dim=-1)
        z1 = AE / CD * f
        # z1 = torch.where(torch.isnan(z1), torch.full_like(z1, 0).cuda(), z1)
        z1 = z1[:, 1:]

        z1 = torch.sort(z1, dim=-1)[0]

        z1 = z1[:, k:(j - k)]
        z1 = torch.mean(z1, dim=-1, keepdim=True)

        xy = pose2d[:, 0, :2] / cam[:, :2] * z1.repeat(1, 2)
        xyz = torch.cat((xy, z1), dim=-1)
        return xyz
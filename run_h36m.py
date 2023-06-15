import os
import pandas as pd
import torch
from common.arguments import parse_args
import torch.optim as optim

from common.arguments import parse_args
from common.dataloader_gen_h36m import get_data
from common.model import GKONet
from common.trainer_h36m import Trainer

def count(model, logger):
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
    logger.info(f'Model Flops: {model.count_flops() / 1000 ** 2:.3f}' + 'M;' + f'Model Parameters: {model_params / 1000 ** 2:.3f}' + 'M')
    return model.count_flops() / 1000 ** 2, model_params / 1000 ** 2

def write_result(experiment, pre_act_p1, pre_act_p2):
    data_recorder_p1 = pd.DataFrame(pre_act_p1)
    data_recorder_p2 = pd.DataFrame(pre_act_p2)
    writer = pd.ExcelWriter(os.path.join('experiment', experiment, 'BEST_RESULT_OF_H36M.xlsx'), engine='openpyxl')
    data_recorder_p1.to_excel(writer, sheet_name="p1 measure", index=False)
    data_recorder_p2.to_excel(writer, sheet_name="p2 measure", index=False)
    writer.save()
    
if __name__ == "__main__":
    args, logger = parse_args()
    
    train_dataloader, test_dataloader, kps_left, kps_right, joints_left, joints_right = get_data(args, logger)
    model = GKONet(num_joints=args.num_joints,
                   in_chans=(2, args.num_joint_dim),
                   embed_dim_pose=args.pose_dim,
                   embed_dim_joint=args.joint_dim,
                   depth=args.depth,
                   drop_path_rate=args.drop_path_rate,
                   mlp_drop_pose=args.mlp_drop_pose,
                   attn_drop_pose=args.attn_drop_pose,
                   mlp_drop_joint=args.mlp_drop_joint,
                   attn_drop_joint=args.attn_drop_joint).cuda()
    
    flops, para = count(model, logger)
    
    if not args.eval:
        lr = args.learning_rate
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
        trainer = Trainer(model, args.experiment, kps_left, kps_right, joints_left, joints_right, num_joints=args.num_joints)
        trainer.train(args.epochs, train_dataloader, test_dataloader, optimizer, args.lr_decay, logger)
        
        write_result(args.experiment, trainer.pre_act_p1, trainer.pre_act_p2)
    else:
        checkpoint = torch.load(args.checkpoint)['para']
        model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
        logger.info(f'load checkpoint from {args.checkpoint}')
        
        trainer = Trainer(model, args.experiment, kps_left, kps_right, joints_left, joints_right, num_joints=args.num_joints, eval=True)
        trainer.eval(test_dataloader, logger)
        
        write_result(args.experiment, trainer.pre_act_p1, trainer.pre_act_p2)
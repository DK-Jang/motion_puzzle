import torch
import numpy as np
import os
import sys
import argparse
from trainer import Trainer
sys.path.append('./motion')
sys.path.append('./etc')
sys.path.append('./preprocess')
from Quaternions import Quaternions
import Animation as Animation
import BVH as BVH
from remove_fs import remove_foot_sliding
from utils import ensure_dirs, get_config
from generate_dataset import process_data
from output2bvh import compute_posture

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

chosen_joints = np.array([
    0,
    2,  3,  4,  5,
    7,  8,  9, 10,
    12, 13, 15, 16,
    18, 19, 20, 22,
    25, 26, 27, 29])

parents = np.array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 10, 13, 14, 15, 10, 17, 18, 19])

def initialize_path(config):
    config['main_dir'] = os.path.join('.', config['name'])
    config['model_dir'] = os.path.join(config['main_dir'], "pth")
    ensure_dirs([config['main_dir'], config['model_dir']])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', 
                        type=str, 
                        default='./model_ours/info/config.yaml',
                        help='Path to the config file.')
    parser.add_argument('--content', 
                        type=str, 
                        default='./datasets/edin_locomotion/test_bvh/locomotion_walk_sidestep_000_000.bvh',
                        help='Path to the content bvh file.')
    parser.add_argument('--style_leg', 
                        type=str, 
                        default='./datasets/Xia/test_bvh/old_normal_walking_002.bvh',
                        help='Path to the style bvh file.')
    parser.add_argument('--style_spine', 
                        type=str, 
                        default='./datasets/Xia/test_bvh/old_normal_walking_002.bvh',
                        help='Path to the style bvh file.')
    parser.add_argument('--style_arm', 
                        type=str, 
                        default='./datasets/Xia/test_bvh/childlike_running_003.bvh',
                        help='Path to the style bvh file.')
    parser.add_argument('--output_dir', 
                        type=str, 
                        default='./output_3bodyparts')
    args = parser.parse_args()

    # initialize path
    cfg = get_config(args.config)
    initialize_path(cfg)

    # make output path folder
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # input content and style bvh file
    content_bvh_file = args.content  
    style_bvh_file_leg = args.style_leg
    style_bvh_file_spine = args.style_spine
    style_bvh_file_arm = args.style_arm
    
    # w/w.o post-processing
    remove_fs = True

    content_name = os.path.split(content_bvh_file)[-1].split('.')[0]
    style_name_leg = os.path.split(style_bvh_file_leg)[-1].split('.')[0]
    style_name_spine = os.path.split(style_bvh_file_spine)[-1].split('.')[0]
    style_name_arm = os.path.split(style_bvh_file_arm)[-1].split('.')[0]
    trans_name = 'Leg_' + style_name_leg + '_Spine_' + style_name_spine + '_Arm_' + style_name_arm + '_Content_' + content_name

    # import norm
    data_norm_dir = os.path.join(cfg['data_dir'], 'norm')
    mean_path = os.path.join(data_norm_dir, "motion_mean.npy")
    std_path = os.path.join(data_norm_dir, "motion_std.npy")
    mean = np.load(mean_path, allow_pickle=True).astype(np.float32)
    std = np.load(std_path, allow_pickle=True).astype(np.float32)
    mean = mean[:, np.newaxis, np.newaxis]
    std = std[:, np.newaxis, np.newaxis]

    # import motion(bvh) and pre-processing
    cnt_mot, cnt_root, cnt_fc = \
        process_data(content_bvh_file, divide=False)
    cnt_mot, cnt_root, cnt_fc = cnt_mot[0], cnt_root[0], cnt_fc[0]

    sty_mot_leg, sty_root_leg, sty_fc_leg = \
        process_data(style_bvh_file_leg, divide=False)
    sty_mot_leg, sty_root_leg, sty_fc_leg = sty_mot_leg[0], sty_root_leg[0], sty_fc_leg[0]

    sty_mot_spine, sty_root_spine, sty_fc_arm = \
        process_data(style_bvh_file_spine, divide=False)
    sty_mot_spine, sty_root_spine, sty_fc_arm = sty_mot_spine[0], sty_root_spine[0], sty_fc_arm[0]

    sty_mot_arm, sty_root_arm, sty_fc_arm = \
        process_data(style_bvh_file_arm, divide=False)
    sty_mot_arm, sty_root_arm, sty_fc_arm = sty_mot_arm[0], sty_root_arm[0], sty_fc_arm[0]

    # normalization
    cnt_motion_raw = np.transpose(cnt_mot, (2, 1, 0))
    sty_motion_raw_leg = np.transpose(sty_mot_leg, (2, 1, 0))
    sty_motion_raw_spine = np.transpose(sty_mot_spine, (2, 1, 0))
    sty_motion_raw_arm = np.transpose(sty_mot_arm, (2, 1, 0))
    cnt_motion = (cnt_motion_raw - mean) / std
    sty_motion_leg = (sty_motion_raw_leg - mean) / std
    sty_motion_spine = (sty_motion_raw_spine - mean) / std
    sty_motion_arm = (sty_motion_raw_arm - mean) / std
    
    cnt_motion = torch.from_numpy(cnt_motion[np.newaxis].astype('float32'))
    sty_motion_leg = torch.from_numpy(sty_motion_leg[np.newaxis].astype('float32'))
    sty_motion_spine = torch.from_numpy(sty_motion_spine[np.newaxis].astype('float32'))
    sty_motion_arm = torch.from_numpy(sty_motion_arm[np.newaxis].astype('float32'))

    # Trainer
    trainer = Trainer(cfg)
    epochs = trainer.load_checkpoint()
    trainer.gen_ema.eval()
    
    # for bvh
    rest, names, _ = BVH.load(content_bvh_file)
    names = np.array(names)
    names = names[chosen_joints].tolist()
    offsets = rest.copy().offsets[chosen_joints]
    orients = Quaternions.id(len(parents))
    
    with torch.no_grad():
        cnt_data = cnt_motion.to(device) 
        sty_data_leg = sty_motion_leg.to(device)
        sty_data_spine = sty_motion_spine.to(device)
        sty_data_arm = sty_motion_arm.to(device)
        cnt_fc = np.transpose(cnt_fc, (1,0))

        c_x = trainer.gen_ema.enc_content(cnt_data)
        s_leg = trainer.gen_ema.enc_style(sty_data_leg)
        s_spine = trainer.gen_ema.enc_style(sty_data_spine)
        s_arm = trainer.gen_ema.enc_style(sty_data_arm)

        out = trainer.gen_ema.dec(c_x[-1], s_leg[::-1], s_leg[::-1], s_spine[::-1], s_arm[::-1], s_arm[::-1])
        
        sty_gt_leg = sty_data_leg.squeeze()
        sty_gt_spine = sty_data_spine.squeeze()
        sty_gt_arm = sty_data_arm.squeeze()
        con_gt = cnt_data.squeeze()
        tra = out.squeeze()

        sty_gt_leg = sty_gt_leg.numpy()*std + mean
        sty_gt_spine = sty_gt_spine.numpy()*std + mean
        sty_gt_arm = sty_gt_arm.numpy()*std + mean
        con_gt = con_gt.numpy()*std + mean
        tra = tra.numpy()*std + mean

        tra_root = cnt_root

        con_gt_mot = compute_posture(con_gt, cnt_root)
        tra_mot = compute_posture(tra, tra_root)
        sty_gt_mot_leg = compute_posture(sty_gt_leg, sty_root_leg)
        sty_gt_mot_spine = compute_posture(sty_gt_spine, sty_root_spine)
        sty_gt_mot_arm = compute_posture(sty_gt_arm, sty_root_arm)
        mots = [sty_gt_mot_leg, sty_gt_mot_spine, sty_gt_mot_arm, con_gt_mot, tra_mot]
        fnames = [style_name_leg, style_name_spine, style_name_arm, content_name, trans_name]
        for mot, fname in zip(mots, fnames):
            local_joint_xforms = mot['local_joint_xforms']

            s = local_joint_xforms.shape[:2]
            rotations = Quaternions.id(s)
            for f in range(s[0]):
                for j in range(s[1]):
                    rotations[f, j] = Quaternions.from_transforms2(local_joint_xforms[f, j])
            
            positions = offsets[np.newaxis].repeat(len(rotations), axis=0)
            positions[:, 0:1] = mot['positions'][:, 0:1]

            anim = Animation.Animation(rotations, positions, 
                                        orients, offsets, parents)

            file_path = os.path.join(output_dir, fname + ".bvh")
            BVH.save(file_path, anim, names, frametime=1.0/60.0)
            if remove_fs and 'Leg_' in fname:
                glb = Animation.positions_global(anim)
                anim = remove_foot_sliding(anim, glb, cnt_fc)
                file_path = os.path.join(output_dir, fname + "_fixed.bvh")
                BVH.save(file_path, anim, names, frametime=1.0/60.0)


if __name__ == '__main__':
    main()
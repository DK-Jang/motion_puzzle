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
                        default='model_ours/info/config.yaml',
                        help='Path to the config file.')
    parser.add_argument('--content', 
                        type=str, 
                        default='datasets/cmu/test_bvh/41_02.bvh',
                        help='Path to the content bvh file.')
    parser.add_argument('--style1', 
                        type=str, 
                        default='datasets/cmu/test_bvh/137_11.bvh',
                        help='Path to the style1 bvh file.')
    parser.add_argument('--style2', 
                        type=str, 
                        default='datasets/cmu/test_bvh/55_07.bvh',
                        help='Path to the style2 bvh file.')
    parser.add_argument('--weight', 
                        type=int, 
                        default=0.5)
    parser.add_argument('--output_dir', 
                        type=str, 
                        default='./output_interpolation')
    args = parser.parse_args()

    # initialize path
    cfg = get_config(args.config)
    initialize_path(cfg)

    # make output path folder
    output_path = args.output_dir
    os.makedirs(output_path, exist_ok=True)
    
    content_bvh_file = args.content
    style_bvh_file1 = args.style1
    style_bvh_file2 = args.style2
    
    # interpolation weight
    inter_p = args.weight

    # w/w.o post-processing
    remove_fs = True

    # name of the output file
    content_name = os.path.split(content_bvh_file)[-1].split('.')[0]
    style_name1 = os.path.split(style_bvh_file1)[-1].split('.')[0]
    style_name2 = os.path.split(style_bvh_file2)[-1].split('.')[0]
    trans_name = 'All_' + str(inter_p) + '_Style1_' + style_name1 + '_Style2_' + style_name2 + '_Content_' + content_name
    
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
    
    sty_mot1, sty_root1, sty_fc1 = \
        process_data(style_bvh_file1, divide=False)
    # sty_mot1, sty_root1, sty_fc1 = sty_mot1[0], sty_root1[0], sty_fc1[0]
    sty_mot1, sty_root1, sty_fc1 = sty_mot1[0][250:1000], sty_root1[0][250:1000], sty_fc1[0][250:1000]

    sty_mot2, sty_root2, sty_fc2 = \
        process_data(style_bvh_file2, divide=False)
    # sty_mot2, sty_root2, sty_fc2 = sty_mot2[0], sty_root2[0], sty_fc2[0]
    sty_mot2, sty_root2, sty_fc2 = sty_mot2[0][250:1000], sty_root2[0][250:1000], sty_fc2[0][250:1000]

    # normalization
    cnt_motion_raw = np.transpose(cnt_mot, (2, 1, 0))
    sty_motion_raw1 = np.transpose(sty_mot1, (2, 1, 0))
    sty_motion_raw2 = np.transpose(sty_mot2, (2, 1, 0))
    cnt_motion = (cnt_motion_raw - mean) / std
    sty_motion1 = (sty_motion_raw1 - mean) / std
    sty_motion2 = (sty_motion_raw2 - mean) / std
    
    cnt_motion = torch.from_numpy(cnt_motion[np.newaxis].astype('float32'))     # (1, dim, joint, seq)
    sty_motion1 = torch.from_numpy(sty_motion1[np.newaxis].astype('float32'))
    sty_motion2 = torch.from_numpy(sty_motion2[np.newaxis].astype('float32'))

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
        sty_data1 = sty_motion1.to(device)
        sty_data2 = sty_motion2.to(device)
        cnt_fc = np.transpose(cnt_fc, (1,0))

        c_x = trainer.gen_ema.enc_content(cnt_data)
        s_c = trainer.gen_ema.enc_style(cnt_data)
        s1 = trainer.gen_ema.enc_style(sty_data1)
        s2 = trainer.gen_ema.enc_style(sty_data2)

        aap0 = torch.nn.AdaptiveAvgPool2d((120,21))
        aap1 = torch.nn.AdaptiveAvgPool2d((60,10))
        aap2 = torch.nn.AdaptiveAvgPool2d((30,5))
        aap3 = torch.nn.AdaptiveAvgPool2d((30,5))

        # for interpolation
        s1[0], s1[1], s1[2], s1[3] = aap0(s1[0]), aap1(s1[1]), aap2(s1[2]), aap3(s1[3])
        s2[0], s2[1], s2[2], s2[3] = aap0(s2[0]), aap1(s2[1]), aap2(s2[2]), aap3(s2[3])

        # s_inter = s1 + (s2-s1)*inter_p if inter_p < 1 -> interpolation, else extrapolation
        s_inter = [0] * 4
        s_inter[0] = s1[0] + (s2[0] - s1[0]) * inter_p
        s_inter[1] = s1[1] + (s2[1] - s1[1]) * inter_p
        s_inter[2] = s1[2] + (s2[2] - s1[2]) * inter_p
        s_inter[3] = s1[3] + (s2[3] - s1[3]) * inter_p

        out = trainer.gen_ema.dec(c_x[-1], s_inter[::-1], s_inter[::-1], s_inter[::-1], s_inter[::-1], s_inter[::-1])
        
        sty_gt1 = sty_data1.squeeze()
        sty_gt2 = sty_data2.squeeze()
        con_gt = cnt_data.squeeze()
        tra = out.squeeze()

        sty_gt1 = sty_gt1.numpy()*std + mean
        sty_gt2 = sty_gt2.numpy()*std + mean
        con_gt = con_gt.numpy()*std + mean
        tra = tra.numpy()*std + mean

        tra_root = cnt_root

        con_gt_mot = compute_posture(con_gt, cnt_root)
        sty_gt_mot1 = compute_posture(sty_gt1, sty_root1)
        sty_gt_mot2 = compute_posture(sty_gt2, sty_root2)
        tra_mot = compute_posture(tra, tra_root)

        mots = [sty_gt_mot1, sty_gt_mot2, con_gt_mot, tra_mot]
        fnames = [style_name1, style_name2, content_name, trans_name]
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

            file_path = os.path.join(output_path, fname + ".bvh")
            BVH.save(file_path, anim, names, frametime=1.0/60.0)
            if remove_fs and '_Content_' in fname:
                glb = Animation.positions_global(anim)
                anim = remove_foot_sliding(anim, glb, cnt_fc, force_on_floor=True)
                file_path = os.path.join(output_path, fname + "_fixed.bvh")
                BVH.save(file_path, anim, names, frametime=1.0/60.0)


if __name__ == '__main__':
    main()
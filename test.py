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
                        default='./datasets/cmu/test_bvh/127_21.bvh',
                        help='Path to the content bvh file.')
    parser.add_argument('--style', 
                        type=str, 
                        default='./datasets/cmu/test_bvh/142_21.bvh',
                        help='Path to the style bvh file.')
    parser.add_argument('--output_dir', 
                        type=str, 
                        default='./output')
    parser.add_argument('--remove_fs',
                        type=bool, 
                        default=True)
    args = parser.parse_args()

    # initialize path
    cfg = get_config(args.config)
    initialize_path(cfg)

    # make output path folder
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # input content and style bvh file
    content_bvh_file = args.content
    style_bvh_file = args.style

    # w/w.o post-processing
    remove_fs = args.remove_fs

    content_name = os.path.split(content_bvh_file)[-1].split('.')[0]
    style_name = os.path.split(style_bvh_file)[-1].split('.')[0]
    recon_name = content_name + '_recon'
    trans_name = 'Style_' + style_name + '_Content_' + content_name
    
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
    
    sty_mot, sty_root, sty_fc = \
        process_data(style_bvh_file, divide=False)
    sty_mot, sty_root, sty_fc = sty_mot[0], sty_root[0], sty_fc[0]

    # normalization
    cnt_motion_raw = np.transpose(cnt_mot, (2, 1, 0))
    sty_motion_raw = np.transpose(sty_mot, (2, 1, 0))
    cnt_motion = (cnt_motion_raw - mean) / std
    sty_motion = (sty_motion_raw - mean) / std
    
    cnt_motion = torch.from_numpy(cnt_motion[np.newaxis].astype('float32'))     # (1, dim, joint, seq)
    sty_motion = torch.from_numpy(sty_motion[np.newaxis].astype('float32'))

    # Trainer
    trainer = Trainer(cfg)
    epochs = trainer.load_checkpoint()
    
    # for bvh
    rest, names, _ = BVH.load(content_bvh_file)
    names = np.array(names)
    names = names[chosen_joints].tolist()
    offsets = rest.copy().offsets[chosen_joints]
    orients = Quaternions.id(len(parents))
    
    loss_test = {}
    with torch.no_grad():
        cnt_data = cnt_motion.to(device) 
        sty_data = sty_motion.to(device)
        cnt_fc = np.transpose(cnt_fc, (1,0))

        outputs, loss_test_dict = trainer.test(cnt_data, sty_data)
        rec = outputs["recon_con"].squeeze()
        tra = outputs["stylized"].squeeze()
        con_gt = outputs["con_gt"].squeeze()
        sty_gt = outputs["sty_gt"].squeeze()

        rec = rec.numpy()*std + mean
        tra = tra.numpy()*std + mean
        con_gt = con_gt.numpy()*std + mean
        sty_gt = sty_gt.numpy()*std + mean

        tra_root = cnt_root

        con_gt_mot, rec_mot = [compute_posture(raw, cnt_root) for raw in [con_gt, rec]]
        tra_mot = compute_posture(tra, tra_root)
        sty_gt_mot = compute_posture(sty_gt, sty_root)
        mots = [sty_gt_mot, con_gt_mot, rec_mot, tra_mot]
        fnames = [style_name, content_name, recon_name, trans_name]
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
            if remove_fs and 'Style_' in fname:
                glb = Animation.positions_global(anim)
                anim = remove_foot_sliding(anim, glb, cnt_fc)
                file_path = os.path.join(output_dir, fname + "_fixed.bvh")
                BVH.save(file_path, anim, names, frametime=1.0/60.0)

        for key in loss_test_dict.keys():
            loss = loss_test_dict[key]
            if key not in loss_test:
                loss_test[key] = []
            loss_test[key].append(loss)

        log = f'Load epoch [{epochs}], '
        loss_test_avg = dict()
        for key, loss in loss_test.items():
            loss_test_avg[key] = sum(loss) / len(loss)
        log += ' '.join([f'{key:}: [{value:}]' for key, value in loss_test_avg.items()])
        print(log)


if __name__ == '__main__':
    main()
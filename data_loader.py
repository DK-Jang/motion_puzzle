import os
import random
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class MotionDataset(Dataset):
    def __init__(self, phase, data_dir, transform=None):
        super(MotionDataset, self).__init__()
        if phase == 'train':
            data_npz_path = os.path.join(data_dir, 'train_dataset.npz')
        else:
            data_npz_path = os.path.join(data_dir, 'test_dataset.npz')

        mdataset = np.load(data_npz_path, allow_pickle=True)
        self.motions = mdataset["motion"]
        # self.roots = mdataset['root']
        # self.foot_contacts = mdataset["foot_contact"]
        
        data_norm_dir = os.path.join(data_dir, 'norm')
        motion_mean_path = os.path.join(data_norm_dir, "motion_mean.npy")
        motion_std_path = os.path.join(data_norm_dir, "motion_std.npy")
        # root_mean_path = os.path.join(data_norm_dir, "root_mean.npy")
        # root_std_path = os.path.join(data_norm_dir, "root_std.npy")
        if os.path.exists(motion_mean_path) and os.path.exists(motion_std_path):
            self.motion_mean = np.load(motion_mean_path, allow_pickle=True).astype(np.float32)
            self.motion_std = np.load(motion_std_path, allow_pickle=True).astype(np.float32)
            # self.root_mean = np.load(root_mean_path, allow_pickle=True).astype(np.float32)
            # self.root_std = np.load(root_std_path, allow_pickle=True).astype(np.float32)
        else:
            assert self.motion_mean and self.motion_std, 'no motion_mean or no motion_std'

        self.transform = transform

    def __len__(self):
        return len(self.motions)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        motion_raw = self.motions[index].astype(np.float32)
        motion = np.transpose(motion_raw, (2, 1, 0)) # (seq, joint, dim) -> (dim, joint, seq)
        motion = torch.from_numpy(motion)

        trans_p = float(np.random.rand(1))
        if self.transform and trans_p < 0.2:
            motion = self.transform(motion)

        motion = (motion - self.motion_mean[:, np.newaxis, np.newaxis]) \
                    / self.motion_std[:, np.newaxis, np.newaxis]   # normalization

        # root_raw = self.roots[index].astype(np.float32)
        # root = np.transpose(root_raw, (2, 1, 0)) # (seq, joint, dim) -> (dim, joint, seq)
        # root = torch.from_numpy(root)
        # root = (root - self.root_mean[:, np.newaxis, np.newaxis]) \
        #             / self.root_std[:, np.newaxis, np.newaxis]   # normalization

        # foot_contact = self.foot_contacts[index].astype(np.float32)

        data = {
                "motion_raw": motion_raw,
                "motion": motion,
                # "root_raw": root_raw,
                # "root": root,
                # "foot_contact": foot_contact
                }

        return data


class RandomResizedCrop(object):
    """Crop and resize randomly the motion in a sample."""
    def __call__(self, sample):
        global crop
        c, j, s = sample.shape      # (dim, joint, seq)

        idx = random.randint(30, 90)
        size = random.randint(60, 120)

        if idx > (size//2)+(size%2) and idx+(size//2) < 120:
            crop = sample[..., idx-(size//2)-(size%2):idx+(size//2)]
        elif idx <= (size//2)+(size%2):
            crop = sample[..., :idx+(size//2)]
        elif  idx+(size//2) >= 120:
            crop = sample[..., idx-(size//2)-(size%2):]

        if size < 90:
            scale = random.uniform(1, 2)
        else:
            scale = random.uniform(0.5, 1)
        crop = crop.unsqueeze(0)
        # crop = torch.from_numpy(crop).unsqueeze(0)
        scale_crop = F.interpolate(crop, scale_factor=(1, scale), mode='bilinear',
                                   align_corners=True, recompute_scale_factor=True)
        scale_crop = scale_crop.squeeze(0)

        if scale_crop.shape[-1] > 120:
            scale_crop = scale_crop[..., scale_crop.shape[-1]//2-60:scale_crop.shape[-1]//2+60]
            return scale_crop
        else:
            # padding
            left = scale_crop[..., :1].repeat_interleave(
                (120-scale_crop.shape[-1])//2 + (120-scale_crop.shape[-1]) % 2, dim=-1)
            left[-3:] = 0.0
            right = scale_crop[..., -1:].repeat_interleave((120-scale_crop.shape[-1])//2, dim=-1)
            right[-3:] = 0.0
            padding_scale_crop = torch.cat([left, scale_crop, right], dim=-1)

        return padding_scale_crop


def get_dataloader(subset_name, config, seed=None, shuffle=None, transform=None):
    dataset = MotionDataset(subset_name, config['data_dir'], transform)
    batch_size = config['batch_size'] if subset_name == 'train' else 1  # since dataloader
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=(subset_name == 'train') if shuffle is None else shuffle,
                      num_workers=config['num_workers'] if subset_name == 'train' else 0,
                    #   worker_init_fn=np.random.seed(seed) if seed else None,
                      pin_memory=True,
                      drop_last=False)


if __name__ == '__main__':
    import sys
    from etc.utils import print_composite
    sys.path.append('./motion')
    sys.path.append('./etc')
    from viz_motion import animation_plot    # for checking dataloader
    data_dir = './datasets/cmu/'
    
    batch_size = 2
    dataset = MotionDataset('train', data_dir)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for batch in data_loader:
        print_composite(batch)
        motion_raw = batch['motion_raw'].cpu().numpy()
        root_raw = batch['root_raw'].cpu().numpy()
        foot_contact = batch['foot_contact'].cpu().numpy()
        anim1 = [motion_raw[0], root_raw[0], foot_contact[0]]
        anim2 = [motion_raw[1], root_raw[1], foot_contact[1]]
        animation_plot([anim1, anim2])

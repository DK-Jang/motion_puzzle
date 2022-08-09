import sys
import os
import numpy as np
import shutil
import scipy.ndimage.filters as filters

sys.path.append('../motion')
sys.path.append('../etc')
from Pivots import Pivots
from Quaternions import Quaternions
import Animation as Animation
import BVH as BVH

def softmax(x, **kw):
    softness = kw.pop('softness', 1.0)
    maxi, mini = np.max(x, **kw), np.min(x, **kw)
    return maxi + np.log(softness + np.exp(mini - maxi))

def softmin(x, **kw):
    return -softmax(-x, **kw)

def pad_to_window(slice, window):
    def get_reflection(src, tlen):
        x = src.copy()
        x = np.flip(x, axis=0)
        ret = x.copy()
        while len(ret) < tlen:
            x = np.flip(x, axis=0)
            ret = np.concatenate((ret, x), axis=0)
        ret = ret[:tlen]
        return ret

    if len(slice) >= window:
        return slice
    left_len = (window - len(slice)) // 2 + (window - len(slice)) % 2
    right_len = (window - len(slice)) // 2
    left = np.flip(get_reflection(np.flip(slice, axis=0), left_len), axis=0)
    right = get_reflection(slice, right_len)
    slice = np.concatenate([left, slice, right], axis=0)
    assert len(slice) == window
    return slice

def divide_clip(input, window, window_step, divide):
    if not divide:  # return the whole clip
        t = ((input.shape[0]) // 4) * 4 + 4
        t = max(t, 12)
        if len(input) < t:
            input = pad_to_window(input, t)

        return [input]

    """ Slide over windows """
    windows = []
    for j in range(0, len(input)-window//4, window_step):
        """ If slice too small pad out by repeating start and end poses """
        slice = input[j:j+window]
        if len(slice) < window:
            left = slice[:1].repeat(
                (window-len(slice))//2 + (window-len(slice)) % 2, axis=0)
            left[..., -3:] = 0.0
            right = slice[-1:].repeat((window-len(slice))//2, axis=0)
            right[..., -3:] = 0.0
            slice = np.concatenate([left, slice, right], axis=0)

        if len(slice) != window:
            raise Exception()

        windows.append(slice)


    return windows

def process_data(filename, window=120, window_step=60, divide=True):
    anim, names, frametime = BVH.load(filename)

    """ Convert to 60 fps """
    anim = anim[::2]

    """ Do FK """
    global_xforms = Animation.transforms_global(anim)

    """ Remove trash joints """
    global_xforms = global_xforms[:, np.array([0,
                                               2,  3,  4,  5,
                                               7,  8,  9, 10,
                                               12, 13, 15, 16,
                                               18, 19, 20, 22,
                                               25, 26, 27, 29])]

    global_positions = global_xforms[:, :, :3, 3] / global_xforms[:, :, 3:, 3]
    global_rotations = Quaternions.from_transforms(global_xforms)
    global_forwards = global_xforms[:, :, :3, 2]
    global_ups = global_xforms[:, :, :3, 1]

    """ Put on Floor """
    fid_l, fid_r = np.array([3, 4]), np.array([7, 8])
    foot_heights = np.minimum(global_positions[:, fid_l, 1], global_positions[:, fid_r, 1]).min(axis=1)
    floor_height = softmin(foot_heights, softness=0.5, axis=0)
    global_positions[:, :, 1] -= floor_height
    global_xforms[:, :, 1, 3] -= floor_height

    """ Extract Forward Direction and smooth """
    sdr_l, sdr_r, hip_l, hip_r = 13, 17, 1, 5
    across = (
        (global_positions[:, sdr_l] - global_positions[:, sdr_r]) +
        (global_positions[:, hip_l] - global_positions[:, hip_r])
        )
    across = across / np.sqrt((across**2).sum(axis=-1))[..., np.newaxis]

    direction_filterwidth = 20
    forward = filters.gaussian_filter1d(
        np.cross(across, np.array([[0, 1, 0]])), direction_filterwidth, axis=0, mode='nearest')
    forward = forward / np.sqrt((forward**2).sum(axis=-1))[..., np.newaxis]

    target = np.array([[0, 0, 1]]).repeat(len(forward), axis=0)
    root_rotation = Quaternions.between(forward, target)[:, np.newaxis]

    """ Local Space """
    root_xforms = np.zeros((len(global_xforms), 1, 4, 4))
    root_xforms[:, :, :3, :3] = root_rotation.transforms()
    root_position = global_positions[:, 0:1]
    root_position[..., 1] = 0
    root_xforms[:, :, :3, 3] = np.matmul(-root_xforms[:, :, :3, :3],
                                            root_position[..., np.newaxis]).squeeze(axis=-1)  # root translation
    root_xforms[:, :, 3:4, 3] = 1.0

    local_xforms = global_xforms.copy()
    local_xforms = np.matmul(root_xforms[:-1], local_xforms[:-1])
    local_positions = local_xforms[:, :, :3, 3] / local_xforms[:, :, 3:, 3]
    local_velocities = np.matmul(root_xforms[:-1, :, :3, :3],
                                    (global_positions[1:] - global_positions[:-1])[..., np.newaxis]).squeeze(axis=-1)
    local_forwards = local_xforms[:, :, :3, 2]
    local_ups = local_xforms[:, :, :3, 1]

    root_velocity = root_rotation[:-1] * (global_positions[1:, 0:1] - global_positions[:-1, 0:1])
    root_rvelocity = Pivots.from_quaternions(root_rotation[1:] * -root_rotation[:-1]).ps    # to angle-axis

    """ Foot Contacts """
    fid_l, fid_r = np.array([3, 4]), np.array([7, 8])
    velfactor, heightfactor = np.array([0.05, 0.05]), np.array([3.0, 2.0])
    feet_l_x = (global_positions[1:, fid_l, 0] - global_positions[:-1, fid_l, 0])**2
    feet_l_y = (global_positions[1:, fid_l, 1] - global_positions[:-1, fid_l, 1])**2
    feet_l_z = (global_positions[1:, fid_l, 2] - global_positions[:-1, fid_l, 2])**2
    feet_l_h = global_positions[:-1, fid_l, 1]
    feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor)
              & (feet_l_h < heightfactor)).astype(np.float)

    feet_r_x = (global_positions[1:, fid_r, 0] - global_positions[:-1, fid_r, 0])**2
    feet_r_y = (global_positions[1:, fid_r, 1] - global_positions[:-1, fid_r, 1])**2
    feet_r_z = (global_positions[1:, fid_r, 2] - global_positions[:-1, fid_r, 2])**2
    feet_r_h = global_positions[:-1, fid_r, 1]
    feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)
              & (feet_r_h < heightfactor)).astype(np.float)

    foot_contacts = np.concatenate([feet_l, feet_r], axis=-1).astype(np.int32)

    """ Stack all features """
    local_full = np.concatenate([local_positions, local_forwards, local_ups, local_velocities], axis=-1)  # for joint-wise
    root_full = np.concatenate([root_velocity[:, :, 0:1], root_velocity[:, :, 2:3], np.expand_dims(root_rvelocity, axis=-1)], axis=-1)

    """ Slide over windows """
    local_windows = divide_clip(local_full, window, window_step, divide=divide)
    root_windows = divide_clip(root_full, window, window_step, divide=divide)
    foot_contacts_windows = divide_clip(foot_contacts, window, window_step, divide=divide)

    return local_windows, root_windows, foot_contacts_windows

def get_files(directory):
    return [os.path.join(directory, f) for f in sorted(list(os.listdir(directory)))
            if os.path.isfile(os.path.join(directory, f))
            and f.endswith('.bvh') and f != 'rest.bvh']

def generate_motion_mean_std(input, norm_dir):
    # input: [n_data, (seq, joint, dim)] -> np.array(n_data, seq, joint, dim)
    input = np.array(input)
    # dim : local_position(3) + local_forward(3) + local_up(3) + local_vel(3)
    os.makedirs(os.path.dirname(norm_dir), exist_ok=True)

    """ Compute mean and std """
    mean = input.mean(axis=(0, 1, 2))
    std = input.std(axis=(0, 1, 2))

    std[:3] = std[:3].mean()    # Xstd_pos
    std[3:6] = std[3:6].mean()  # Xstd_forward
    std[6:9] = std[6:9].mean()  # Xstd_up
    std[9:12] = std[9:12].mean()  # Xstd_vel

    std[np.where(std == 0)] = 1e-9

    mean_path = os.path.join(norm_dir, "motion_mean.npy")
    std_path = os.path.join(norm_dir, "motion_std.npy")
    np.save(mean_path, mean.astype(np.float32))
    np.save(std_path, std.astype(np.float32))

def generate_root_mean_std(input, norm_dir):
    # input: [n_data, (seq, joint, dim)] -> np.array(n_data, seq, joint, dim)
    input = np.array(input)
    # dim : root_vel(2) + root_rvel(1)
    os.makedirs(os.path.dirname(norm_dir), exist_ok=True)

    """ Compute mean and std """
    mean = input.mean(axis=(0, 1, 2))
    std = input.std(axis=(0, 1, 2))

    std[:2] = std[:2].mean()  # Translational Velocity
    std[2:3] = std[2:3].mean()  # Rotational Velocity

    std[np.where(std == 0)] = 1e-9

    mean_path = os.path.join(norm_dir, "root_mean.npy")
    std_path = os.path.join(norm_dir, "root_std.npy")
    np.save(mean_path, mean.astype(np.float32))
    np.save(std_path, std.astype(np.float32))

def main():
    root_path = '../datasets/'
    dataset_name = 'cmu'
    dataset_dir = os.path.join(root_path, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    print("Create folder ", dataset_dir)

    bvh_dir = '../database/cmu'
    bvh_files = np.array(get_files(bvh_dir), dtype=np.str_)

    total_len = len(bvh_files)
    test_idx = np.random.choice(total_len, total_len//10, replace=False)
    test_bvh_files = bvh_files[test_idx]
    test_filename = [x.split('/')[-1] for x in test_bvh_files]

    m_bvh_dir = '../database/cmu_m'
    m_bvh_files = np.array(get_files(m_bvh_dir), dtype=np.str_)
    bvh_files = np.concatenate([bvh_files, m_bvh_files], axis=0)
    np.random.shuffle(bvh_files)

    # for train and test clips
    train_motions, train_roots, train_foot_contacts = [], [], []
    test_motions, test_roots, test_foot_contacts = [], [], []
    test_files = []
    for i, item in enumerate(bvh_files):
        print('Processing %i of %i (%s)' % (i+1, len(bvh_files), item))

        if item.split('/')[-1] in test_filename:
            local_uclip, root_uclip, ufc = process_data(item, window=120, window_step=60, divide=False)
            test_motions += local_uclip
            test_roots += root_uclip
            test_foot_contacts += ufc
            test_files.append(item)
        else:
            local_clips, root_clips, fcs = process_data(item, window=120, window_step=60, divide=True)
            train_motions += local_clips
            train_roots += root_clips
            train_foot_contacts += fcs
        
    # collect train dataset
    train_dataset_path = os.path.join(dataset_dir, 'train_dataset.npz')
    np.savez_compressed(train_dataset_path, motion=train_motions, 
                                            root=train_roots, 
                                            foot_contact=train_foot_contacts)
    print('Save train dataset')

    # collect test dataset
    test_dataset_path = os.path.join(dataset_dir, 'test_dataset.npz')
    np.savez_compressed(test_dataset_path, motion=test_motions, 
                                           root=test_roots, 
                                           foot_contact=test_foot_contacts)

    test_bvh_dir = os.path.join(dataset_dir, 'test_bvh')
    if not os.path.exists(test_bvh_dir):
        os.makedirs(test_bvh_dir)
    for file in test_files:
        shutil.copy(file, os.path.join(test_bvh_dir, file.split('/')[-1]))
    print('Save test dataset')

    # calculate mean and std
    norm_dir = os.path.join(root_path, dataset_name, 'norm')
    os.makedirs(norm_dir, exist_ok=True)

    generate_motion_mean_std(train_motions, norm_dir)
    generate_root_mean_std(train_roots, norm_dir)

if __name__ == '__main__':
    main()
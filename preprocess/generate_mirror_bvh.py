import sys
import os
import numpy as np
sys.path.append('../motion')
from Quaternions import Quaternions
from Animation import Animation
import BVH as BVH

def qfix(q):
    """
    Enforce quaternion continuity across the time dimension by selecting
    the representation (q or -q) with minimal distance (or, equivalently, maximal dot product)
    between two consecutive frames.
    
    Expects a tensor of shape (L, J, 4), where L is the sequence length and J is the number of joints.
    Returns a tensor of the same shape.
    """
    assert len(q.shape) == 3
    assert q.shape[-1] == 4

    result = q.copy()
    dot_products = np.sum(q[1:]*q[:-1], axis=2)
    mask = dot_products < 0
    mask = (np.cumsum(mask, axis=0)%2).astype(bool)
    result[1:][mask] *= -1
    return result

def get_files(directory):
    return [os.path.join(directory, f) for f in sorted(list(os.listdir(directory)))
            if os.path.isfile(os.path.join(directory, f))
            and f.endswith('.bvh') and f != 'rest.bvh']

def main():
    bvh_dir = '../database/cmu'
    bvh_files = get_files(bvh_dir)

    for i, item in enumerate(bvh_files):
        print('Processing %i of %i (%s)' % (i+1, len(bvh_files), item))
        filename = item.split('/')[-1]
        filename = filename.split('.')[0]
        mirrored_filename = filename +'m.bvh'
        
        anim, _, _ = BVH.load(item)
        rotations = anim.rotations.qs
        positions = anim.positions
        trajectories = positions[:, 0].copy()
        orients   = anim.orients
        offsets   = anim.offsets
        parents   = anim.parents
        joints_left = np.array([1, 2, 3, 4, 5, 17, 18, 19, 20, 21, 22, 23], dtype='int64') 
        joints_right = np.array([6, 7, 8, 9, 10, 24, 25, 26, 27, 28, 29, 30], dtype='int64')

        mirrored_rotations = rotations.copy()
        mirrored_positions = positions.copy()
        mirrored_trajectory = trajectories.copy()

        # Flip left/right joints
        mirrored_rotations[:, joints_left] = rotations[:, joints_right]
        mirrored_rotations[:, joints_right] = rotations[:, joints_left]
                
        mirrored_rotations[:, :, [2, 3]] *= -1
        mirrored_rotations = Quaternions(qfix(mirrored_rotations))
        mirrored_trajectory[:, 0] *= -1
        mirrored_positions[:, 0] = mirrored_trajectory

        mirrored_anim = Animation(mirrored_rotations, mirrored_positions, orients, offsets, parents)
        
        file_dir = bvh_dir + '_m'
        if not os.path.exists(file_dir):
            print("Create folder ", file_dir)
            os.makedirs(file_dir)
        mirrored_file_path = os.path.join(file_dir, mirrored_filename)
        BVH.save(mirrored_file_path, mirrored_anim)

if __name__ == '__main__':
    main()
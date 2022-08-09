import numpy as np
import sys
sys.path.append('../motion')
from Quaternions import Quaternions

chosen_joints = np.array([
    0,
    2,  3,  4,  5,
    7,  8,  9, 10,
    12, 13, 15, 16,
    18, 19, 20, 22,
    25, 26, 27, 29])

parents = np.array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 10, 13, 14, 15, 10, 17, 18, 19])

def normalized(v):
    norm = np.sum(v**2.0, axis=-1)**0.5
    return v / norm[..., np.newaxis]

def transforms_inv(ts):
    fts = ts.reshape(-1, 4, 4)
    fts = np.array(list(map(lambda x: np.linalg.inv(x), fts)))
    return fts.reshape(ts.shape)

def transforms_blank(shape):
    # type(shape): tuple
    ts = np.zeros(shape + (4, 4))
    ts[..., 0, 0] = 1.0; ts[..., 1, 1] = 1.0;
    ts[..., 2, 2] = 1.0; ts[..., 3, 3] = 1.0;
    return ts

def compute_posture(raw_motion, root):
    anim = raw_motion.copy()    # (dim, joint, seq)
    anim = np.swapaxes(anim, 0, 2)     # (seq, joint, dim)
    n_joint = anim.shape[1]
    seq = anim.shape[0]

    positions = anim[..., :3]
    forwards = anim[..., 3:6]
    ups = anim[..., 6:9]
    velocities = anim[..., 9:12]
    root_x = root[:, 0, 0:1]
    root_z = root[:, 0, 1:2]
    root_r = root[:, 0, 2:]
    
    current_root = transforms_blank((1, ))
    rotation = Quaternions.id(1)
    translation = np.array([[0,0,0]])
    global_xforms = transforms_blank(anim.shape[:2])
    local_joint_xforms = transforms_blank(anim.shape[:2])
    for i in range(seq):
        positions[i] = np.matmul(current_root[:, :3, :3].repeat(n_joint, axis=0), positions[i][..., np.newaxis]).squeeze(-1)
        positions[i,:,0] = positions[i,:,0] + current_root[0, 0, 3]
        positions[i,:,2] = positions[i,:,2] + current_root[0, 2, 3]
        forwards[i] = np.matmul(current_root[:, :3, :3].repeat(n_joint, axis=0), normalized(forwards[i])[..., np.newaxis]).squeeze(-1)
        ups[i] = np.matmul(current_root[:, :3, :3].repeat(n_joint, axis=0), normalized(ups[i])[..., np.newaxis]).squeeze(-1)
        velocities[i] = np.matmul(current_root[:, :3, :3].repeat(n_joint, axis=0), velocities[i][..., np.newaxis]).squeeze(-1)

        # lerp positions using velocity
        positions[i] = 0.5*(positions[i-1] + velocities[i]) + 0.5*positions[i]

        # update root
        rotation = Quaternions.from_angle_axis(-root_r[i], np.array([0,1,0])) * rotation
        translation = translation + rotation * np.array([root_x[i], 0, root_z[i]], dtype=np.float32)
        current_root[:, :3, 3] = translation
        current_root[:, 1, 3] = 0.0  # fix root height
        current_root[:, :3, :3] = rotation.transforms()

        # update global joint xforms
        global_xforms[i, :, :3, 0] = normalized(np.cross(ups[i], forwards[i]))
        global_xforms[i, :, :3, 1] = ups[i]
        global_xforms[i, :, :3, 2] = forwards[i]
        global_xforms[i, :, :3, 3] = positions[i]

        # update local joint xforms
        for j in range(n_joint):
            if j == 0:  # root
                local_joint_xforms[i, j] = global_xforms[i, j]
            else:
                local_joint_xforms[i, j] =  np.matmul(transforms_inv(global_xforms[i, parents[j]]), global_xforms[i, j])
    
    mot = {}
    mot['positions'] = positions
    mot['global_xforms'] = global_xforms
    mot['local_joint_xforms'] = local_joint_xforms

    return mot
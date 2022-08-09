import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.patheffects as pe
sys.path.append('../motion')
from Quaternions import Quaternions

parents = np.array([-1,0,1,2,3,0,5,6,7,0,9,10,11,11,13,14,15,11,17,18,19])
joint_foot_indicies = [3, 4, 7, 8]

def animation_plot(animations, interval=33.33):
    foot_contacts = [None] * len(animations)
    for ai in range(len(animations)):
        anim = animations[ai].copy()
        joints_, roots, contact = anim[0], anim[1], anim[2]
                
        joints_pos = joints_[..., :3]
        # joints_forward = joints_[..., 3:6]
        # joints_up = joints_[..., 6:9]
        # joints_vel = joints_[..., 9:12]
        root_x = roots[:, 0, 0:1]
        root_z = roots[:, 0, 1:2]
        root_r = roots[:, 0, 2:3]
        
        rotation = Quaternions.id(1)
        offsets = []
        translation = np.array([[0,0,0]])
        for i in range(len(joints_pos)):
            joints_pos[i,:,:] = rotation * joints_pos[i]
            joints_pos[i,:,0] = joints_pos[i,:,0] + translation[0,0]
            joints_pos[i,:,2] = joints_pos[i,:,2] + translation[0,2]
            rotation = Quaternions.from_angle_axis(-root_r[i], np.array([0,1,0])) * rotation
            offsets.append(rotation * np.array([0,0,1]))
            translation = translation + rotation * np.array([root_x[i], 0, root_z[i]], dtype=np.float32)
        
        animations[ai] = joints_pos
        foot_contacts[ai] = contact
            
    scale = 1.25*((len(animations))/2)
    
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')
    rscale = scale * 30
    ax.set_xlim3d(-rscale, rscale)
    ax.set_zlim3d(0, rscale*2)
    ax.set_ylim3d(-rscale, rscale)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(20, -60) # (-40, 60): up view

    acolors = list(sorted(colors.cnames.keys()))[::-1]
    acolors.pop(3)
    lines = []
    contact_dots = []
    for ai, anim in enumerate(animations):
        lines.append([plt.plot([0,0], [0,0], [0,0], color=acolors[ai], 
            lw=2, path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()])[0] for _ in range(anim.shape[1])])
        contact_dots.append([plt.plot([0,0], [0,0], [0,0], color='white', zorder=3,
                    linewidth=2, linestyle='',
                    marker="o", markersize=2 * scale,
                    path_effects=[pe.Stroke(linewidth=2, foreground='black'), pe.Normal()]
                    )[0] for _ in range(contact.shape[1])])
    
    def animate(i):
        changed = []
        for ai in range(len(animations)):
            offset = 25*(ai-((len(animations))/2))
            for j in range(len(parents)):
                if parents[j] != -1:
                    lines[ai][j].set_data(
                        [ animations[ai][i,j,0]+offset, animations[ai][i,parents[j],0]+offset],
                        [-animations[ai][i,j,2],       -animations[ai][i,parents[j],2]])
                    lines[ai][j].set_3d_properties(
                        [ animations[ai][i,j,1],        animations[ai][i,parents[j],1]])
            changed += lines
        
        # foot contact
            for j, f_idx in enumerate(joint_foot_indicies):
                contact_dots[ai][j].set_data([animations[ai][i,f_idx,0]+offset], [-animations[ai][i,f_idx,2]])      # left toe
                contact_dots[ai][j].set_3d_properties([animations[ai][i,f_idx,1]])
                color = 'red' if foot_contacts[ai][i, j] == 1.0 else 'blue'
                contact_dots[ai][j].set_color(color)
            
        return changed
        
    plt.tight_layout()
        
    ani = animation.FuncAnimation(fig, 
        animate, np.arange(len(animations[0])), interval=interval)
        
    plt.show()


if __name__ == '__main__':
    data_path = '../datasets/cmu/test_dataset.npz'
    dataset = np.load(data_path, allow_pickle=True)
    motions, roots, foot_contacts = dataset["motion"], dataset["root"], dataset["foot_contact"]

    for i in range(len(motions)):
        motion1 = [motions[i], roots[i], foot_contacts[i]]
        motion2 = [motions[i+1], roots[i+1], foot_contacts[i+1]]
        animation_plot([motion1, motion2])
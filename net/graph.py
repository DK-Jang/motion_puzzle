import numpy as np
import torch
import torch.nn as nn


class Graph_Joint():
    def __init__(self, layout='cmu', strategy='uniform', max_hop=2, dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation
        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        if layout == 'cmu':
            self.num_node = 21
            self_link = [(i, i) for i in range(self.num_node)]
            parents = [-1,                           # Hip
                           0,  1,  2,  3,            # Left Leg
                           0,  5,  6,  7,            # Right Leg
                           0,  9, 10, 11,            # Spine
                                  10, 13, 14, 15,    # Left Arm
                                  10, 17, 18, 19]    # Right Arm

            neighbor_link = [(i, parents[i]) for i in range(len(parents))]
            neighbor_link = neighbor_link[1:]   # remove (0, -1)
            self.edge = self_link + neighbor_link
            self.center = 0

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A


class Graph_Mid():
    def __init__(self, layout='cmu', strategy='uniform', max_hop=1, dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation
        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        if layout == 'cmu':
            self.num_node = 10
            self_link = [(i, i) for i in range(self.num_node)]
            # LeftHip = 0, LeftLeg = 1
            # RightHip = 2, RightLeg = 3
            # Back = 4, Neck = 5
            # LeftShoulder = 6, LeftArm = 7
            # RightShoulder = 8, RightArm = 9
            neighbor_link = [(4,0), (4,2), (4,5), (4,6), (4,8),
                             (0,1), (2,3), (6,7), (8,9)]
            self.edge = self_link + neighbor_link
            self.center = 4

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A


class Graph_Bodypart():
    def __init__(self, layout='cmu', strategy='uniform', max_hop=1, dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation
        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):        
        if layout=='cmu':
            self.num_node = 5
            self_link = [(i, i) for i in range(self.num_node)]
            # Left Leg  -> 0
            # Right Leg -> 1
            # Spine     -> 2
            # Left Arm  -> 3
            # Right Arm -> 4
            neighbor_link = [(0,2), (1,2), (3,2), (4,2)]
            self.edge = self_link + neighbor_link
            self.center = 2

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
            

def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD


class PoolJointToMid(nn.Module):
    def __init__(self):
        super().__init__()      # kernel=3, stride=2 forward kinematics chain Pooling
        self.LeftHip = [0, 1, 2]
        self.LeftLeg = [2, 3, 4]
        self.RightHip = [0, 5, 6]
        self.RightLeg = [6, 7, 8]
        self.Back = [0, 9, 10]
        self.Neck = [10, 11, 12]
        self.LeftShoulder = [10, 13, 14]
        self.LeftArm = [14, 15, 16]
        self.RightShoulder = [10, 17, 18]
        self.RightArm = [18, 19, 20]

        njoints = 21
        nmid = 10
        weight = torch.zeros(njoints, nmid, dtype=torch.float32, requires_grad=False)
        weight[self.LeftHip, 0] = 1.0
        weight[self.LeftLeg, 1] = 1.0
        weight[self.RightHip, 2] = 1.0
        weight[self.RightLeg, 3] = 1.0
        weight[self.Back, 4] = 1.0
        weight[self.Neck, 5] = 1.0
        weight[self.LeftShoulder, 6] = 1.0
        weight[self.LeftArm, 7] = 1.0
        weight[self.RightShoulder, 8] = 1.0
        weight[self.RightArm, 9] = 1.0

        scale = torch.sum(weight, axis=0, keepdim=True)
        weight = weight / scale
        self.register_buffer('weight', weight)
        
    def forward(self, x):
        x = torch.einsum('nctv,vw->nctw', (x, self.weight))        
        return x


class PoolMidToBodypart(nn.Module):
    def __init__(self):     # kernel=3, stride=2 forward kinematics chain Pooling
        super().__init__()
        self.LeftLeg = [0, 1, 4]
        self.RightLeg = [2, 3, 4]
        self.Spine = [4, 5]
        self.LeftArm = [6, 7, 4]
        self.RightArm = [8, 9, 4]

        nmid = 10
        nbody = 5
        weight = torch.zeros(nmid, nbody, dtype=torch.float32, requires_grad=False)
        weight[self.LeftLeg, 0] = 1.0
        weight[self.RightLeg, 1] = 1.0
        weight[self.Spine, 2] = 1.0
        weight[self.LeftArm, 3] = 1.0
        weight[self.RightArm, 4] = 1.0

        scale = torch.sum(weight, axis=0, keepdim=True)
        weight = weight / scale
        self.register_buffer('weight', weight)
        
    def forward(self, x):        
        x = torch.einsum('nctv,vw->nctw', (x, self.weight))        
        return x


class UnpoolBodypartToMid(nn.Module):
    def __init__(self):
        super().__init__()
        self.LeftLeg = [0, 1, 4]
        self.RightLeg = [2, 3, 4]
        self.Spine = [4, 5]
        self.LeftArm = [6, 7, 4]
        self.RightArm = [8, 9, 4]

        nbody = 5
        nmid = 10
        weight = torch.zeros(nbody, nmid, dtype=torch.float32, requires_grad=False)
        weight[0, self.LeftLeg] = 1.0
        weight[1, self.RightLeg] = 1.0
        weight[2, self.Spine] = 1.0
        weight[3, self.LeftArm] = 1.0
        weight[4, self.RightArm] = 1.0

        scale = torch.sum(weight, axis=0, keepdim=True)
        weight = weight / scale
        self.register_buffer('weight', weight)
        
    def forward(self, x):
        x = torch.einsum('nctv,vw->nctw', (x, self.weight))        
        return x


class UnpoolMidToJoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.LeftHip = [0, 1, 2]
        self.LeftLeg = [2, 3, 4]
        self.RightHip = [0, 5, 6]
        self.RightLeg = [6, 7, 8]
        self.Back = [0, 9, 10]
        self.Neck = [10, 11, 12]
        self.LeftShoulder = [10, 13, 14]
        self.LeftArm = [14, 15, 16]
        self.RightShoulder = [10, 17, 18]
        self.RightArm = [18, 19, 20]

        nmid = 10
        njoints = 21
        weight = torch.zeros(nmid, njoints, dtype=torch.float32, requires_grad=False)
        weight[0, self.LeftHip] = 1.0
        weight[1, self.LeftLeg] = 1.0
        weight[2, self.RightHip] = 1.0
        weight[3, self.RightLeg] = 1.0
        weight[4, self.Back] = 1.0
        weight[5, self.Neck] = 1.0
        weight[6, self.LeftShoulder] = 1.0
        weight[7, self.LeftArm] = 1.0
        weight[8, self.RightShoulder] = 1.0
        weight[9, self.RightArm] = 1.0

        scale = torch.sum(weight, axis=0, keepdim=True)
        weight = weight / scale
        self.register_buffer('weight', weight)
        
    def forward(self, x):
        x = torch.einsum('nctv,vw->nctw', (x, self.weight))        
        return x


if __name__ == '__main__':
    x_in = torch.randn(16, 15, 120, 21)
    pool_test = PoolJointToMid()

    print(PoolJointToMid()(x_in).shape)
    # print(pool_test(x_in).shape)
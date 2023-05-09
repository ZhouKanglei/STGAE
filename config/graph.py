import numpy as np

class Graph():
    """ The Graph to model the skeletons extracted by the openpose

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).

        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """

    def __init__(self,
                 layout='nyu',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        if layout == 'nyu':
            self.num_node = 36
            self.self_link = [(i, i) for i in range(self.num_node)]
            neighbor_direct = [(1, 2), (3, 2), (4, 3), (5, 4), (6, 5), (30, 6),
                               (8, 7), (9, 8), (10, 9), (11, 10), (12, 11), (30, 12),
                               (14, 13), (15, 14), (16, 15), (17, 16), (18, 17), (30, 18),
                               (20, 19), (21, 20), (22, 21), (23, 22), (24, 23), (30, 24),
                               (26, 25), (27, 26), (28, 27), (29, 28), (30, 29)]

            neighbor_indirect = [(1, 7), (8, 2), (9, 3), (10, 4), (11, 5), (12, 6),
                                 (13, 7), (14, 8), (15, 9), (16, 10), (17, 11), (18, 12),
                                 (19, 13), (20, 14), (21, 15), (22, 16), (23, 17), (24, 18),
                                 (25, 19), (26, 20), (27, 21), (28, 22), (29, 23), (30, 24)]

            self.neighbor_direct_link = [(i - 1, j - 1) for (i, j) in neighbor_direct]
            self.neighbor_indirect_link = [(i - 1, j - 1) for (i, j) in neighbor_indirect]
            self.edge = self.self_link + self.neighbor_direct_link + self.neighbor_indirect_link

        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):

        if strategy == 'spatial':
            A = np.zeros((3, self.num_node, self.num_node))
            for i, j in self.self_link:
                A[0, i, j] = 1
                A[0, j, i] = 1
            A[0, :, :] = normalize_undigraph(A[0, :, :])

            for i, j in self.neighbor_direct_link:
                A[1, i, j] = 1
                A[1, j, i] = 1
            A[1, :, :] = normalize_undigraph(A[1, :, :])

            for i, j in self.neighbor_indirect_link:
                A[2, i, j] = 1
                A[2, j, i] = 1
            A[2, :, :] = normalize_undigraph(A[2, :, :])

        else:
            A = np.zeros((1, self.num_node, self.num_node))
            for i, j in self.edge:
                A[0, i, j] = 1
                A[0, j, i] = 1
            A[0, :, :] = normalize_undigraph(A[0, :, :])

        self.A = A


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
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

if __name__ == '__main__':
    skeleton = Graph(layout='nyu',
                 strategy='spatial')
    A = skeleton.A

    print('----')
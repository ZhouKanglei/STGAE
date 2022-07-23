import numpy as np
import multiprocessing
import tensorflow as tf

from tqdm import tqdm

from sklearn.metrics import mean_squared_error
from config.graph import Graph

# Multi-processing callback func.
def get_idx(args):
    '''A callback function of multiprocessing'''
    return args

############################################################
# Calculate bone length error between two frames
###########################################################
def bone_length_error(x, x_hat, edges):
    mse = 0

    for i, j in edges:
        orignal_bone_len = np.sqrt(np.sum(np.square(x[i, :] - x[j, :])))
        predict_bone_len = np.sqrt(np.sum(np.square(x_hat[i, :] - x_hat[j, :])))

        mse += np.square(orignal_bone_len - predict_bone_len)

    return mse / len(edges)

def caculate_all_bone_length_error(x, x_hat, edge='direct'):
    n, t, _, _ = x.shape
    mse = np.zeros(shape=(n, t))

    # load direct adjacent matrix
    A = Graph(layout='nyu', strategy='spatial').A

    if edge == 'direct':
        A = A[1]
    elif edge == 'indirect':
        A = A[2]
    else:
        A = A[1] + A[2]

    # edges
    edges = []
    for i in range(A.shape[0]):
        for j in range(i):
            if A[i, j] != 0: # bone length
                edges.append((i, j))

    # multiprocessing
    p = multiprocessing.Pool(64)
    args = [i for i in range(n)]
    pbar = tqdm(range(n))

    for i in p.imap(get_idx, args):
        pbar.set_description('Bone length: %d' % (i + 1))
        for j in range(t):
            mse[i, j] = bone_length_error(x[i, j, :, :], x_hat[i, j, :, :], edges)
        pbar.update()

    pbar.close()
    p.close()
    p.join()

    return np.sum(np.sum(mse)) / n / t



############################################################
# Calculate pose mse
###########################################################
def caculate_all_mse(x, x_hat):

    x = x.reshape([-1,1])
    x_hat = x_hat.reshape([-1,1])
    mse = np.sum(np.square(x - x_hat)) / x.shape[0]

    return mse

############################################################
# Custom loss function
###########################################################
def pose_bone_loss(y_true, y_pred):
    l_1 = tf.keras.losses.MSE(y_true, y_pred)
    # l_2 = caculate_all_bone_length_error(y_true, y_pred, edge='direct')
    # l_3 = caculate_all_bone_length_error(y_true, y_pred, edge='indirect')

    return l_1
# time
import time
TIME_START = time.strftime("%Y_%m_%d-%H_%M_%S", time.localtime())

# seed
import numpy as np
np.random.seed(1024)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

import tensorflow as tf
tf.keras.backend.set_floatx('float64')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import warnings
warnings.filterwarnings('ignore')

##################################################
# dataset
#################################################
NYU_PATH = '/media/zhoukanglei/Windows/chen/pycharm/dataset/dataset'
TIM_TREMOR_PATH = '/media/zhoukanglei/Windows/ZKL/Data/TIM-Tremor/data/'
NYU_TRAIN_NUM = 72757
NYU_TEST_NUM = 8252
NYU_KINECT_NUM = 3

# Hyper-parameters
SIGMA_O = 0.1
SIGMA_S = 0.1
BETA = 50
TREMOR_AMPLITUDE = 6
TREMOR_CYCLE = 5

# Noise type
NOISE = 'Uniform'

##################################################
# model
#################################################
OUTPUT_LOG_PATH = './output/log/tensorboard'
OUTPUT_HISTORY_PATH = './output/history'
VALIDATION_RATE = 0.15
T_SIZE = 36
OPTION = False

OPT_MODEL_PATH = './output/weight/best_weights-%s.h5' % NOISE
FIG_PATH = './output/plots/predict-%s' % NOISE

MAX_EPOCH = 300
PATIENCE = 10
MIN_LR = 1e-7

# Tools
OPT_IDX = 9 # If not existing acceptable MSE (MSE < 1), then set a defalut batch to make video.

# Graph type
STRATEGY = 'spatial' # Spatial strategy, return A (3 x N x N); otherwise, return A = A_1 + A_2 + A_3 (1 x N x N)
if STRATEGY == '':
    OPT_MODEL_PATH = './output/weight/best_weights-%s-%s.h5' % (NOISE, STRATEGY)
    FIG_PATH = './output/plots/predict-%s-%s' % (NOISE, STRATEGY)

# Attention mechanism
ATTENTION_LIST = ['A', 'A+M', 'A*M', 'A+B+C']
ATTENTION = ATTENTION_LIST[3]

if ATTENTION != '':
    OPT_MODEL_PATH = './output/weight/best_weights-%s-%s-%s.h5' % (NOISE, STRATEGY, ATTENTION)
    FIG_PATH = './output/plots/predict-%s-%s-%s' % (NOISE, STRATEGY, ATTENTION)

# temp model path
TEMP_BEST_MODEL = OPT_MODEL_PATH[:-3] + '_temp' + '.h5'

# logging
from Tool.logs import Logger
log = Logger('./output/log/logger/all_log-%s-%s.log' % (OPT_MODEL_PATH[28:-3], TIME_START), level='debug')

##################################################
# path verification
#################################################
def path_verification(path):
    if os.path.exists(path) != True:
        if os.path.isdir(path):
            os.mkdir()
            log.logger.info('Dir %s is not existing, mkdir...' % path)

        if os.path.isfile(path):
            log.logger.error('File %s is not existing...' % path)
            assert os.path.isfile(path) == True

path_verification(NYU_PATH)
path_verification(TIM_TREMOR_PATH)

path_verification(NYU_PATH)
path_verification(TIM_TREMOR_PATH)
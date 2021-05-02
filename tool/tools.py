import shutil

from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model

from config.config import *
from tool.visualization import *
from tool.metrics import *


###################################################################
# Path verification
###################################################################
def path_verification(path):
    if os.path.exists(path) != True:
        if os.path.isdir(path):
            os.mkdir()
            log.logger.info('Dir %s is not existing, mkdir...' % path)

        if os.path.isfile(path):
            log.logger.error('File %s is not existing...' % path)
            assert os.path.isfile(path) == True

path_verification(OUTPUT_LOGGER_PATH)
path_verification(OUTPUT_LOG_PATH)
path_verification(OUTPUT_HISTORY_PATH)
path_verification(FIG_PATH)

###################################################################
# Delete logger file
###################################################################
def delete_file_logger(src):
    '''delete files and folders'''
    log.logger.debug('...')
    for root, dirs, files in os.walk(src):
        for name in files:
            if name.endswith('.log') and os.stat(os.path.join(root, name)).st_size == 0:
                os.remove(os.path.join(root, name))
                log.logger.info("Delete empty logger file: " + os.path.join(root, name))
            else:
                with open(os.path.join(root, name), 'r', encoding='utf-8') as f:
                    if 'pose error' not in f.read() and TIME_START not in name:
                        os.remove(os.path.join(root, name))

        for dir in dirs:
            delete_file_logger(os.path.join(root, dir))

###################################################################
# Delete history log every time before processing
###################################################################
def delete_file_folder(src):
    '''delete files and folders'''
    for root, dirs, files in os.walk(src):
        for name in files:
            os.remove(os.path.join(root, name))
            log.logger.info("Delete File: " + os.path.join(root, name))

        for dir in dirs:
            delete_file_folder(os.path.join(root, dir))

    for root, dirs, files in os.walk(src):
        for dir in dirs:
            shutil.rmtree(os.path.join(root, dir), True)
            log.logger.info("Delete Dir: " + os.path.join(root, dir))

delete_file_folder(OUTPUT_LOG_PATH)

###################################################################
# Delete temp model
###################################################################
def delete_temp_model(src):
    '''delete files and folders'''
    for root, dirs, files in os.walk(src):
        for name in files:
            if 'temp' in name:
                os.remove(os.path.join(root, name))
                log.logger.info("Delete temp model: " + os.path.join(root, name))

        for dir in dirs:
            delete_file_logger(os.path.join(root, dir))

delete_temp_model(MODEL_PATH)

###################################################################
# Delete old history
###################################################################
def delete_old_history(src):
    '''delete files and folders'''
    for root, dirs, files in os.walk(src):
        for name in files:
            if time.strftime("history-%Y_%m_%d-", time.localtime()) not in name:
                os.remove(os.path.join(root, name))
                log.logger.info("Delete old history: " + os.path.join(root, name))

        for dir in dirs:
            delete_file_logger(os.path.join(root, dir))

delete_old_history(OUTPUT_HISTORY_PATH)


###################################################################
# Load NYU tremor data (only translation).
###################################################################
def load_data(mode='train', t_size=27, sample_interval=100, start_pos=0):
    # traverse all the tremor data
    dataset_dir = os.path.join(NYU_PATH, 'tremor/' + mode) # tremor dataset path

    file_names = []
    for root, dirs, files in os.walk(dataset_dir):
        if files == []:
            continue
        for file in files:
            if file.endswith("mat"):
                file_names.append(root + "/" + file) # tremor data file name

    # select files by the sampling interval
    selected_file_names = []
    pos = start_pos
    while pos < len(file_names):
        selected_file_names.append(os.path.join(dataset_dir, 'jnt_tremor_%d' % pos))
        pos += sample_interval

    tra_num = len(selected_file_names) # the number of selected tremor data

    # get the number of samples by the mode
    if mode == 'train':
        num = NYU_TRAIN_NUM
    else:
        num = NYU_TEST_NUM

    # get the data shape: single tremor file and all tremor file
    x_shape = [tra_num + 1, NYU_KINECT_NUM, num, 36, 3]
    x = np.empty(shape=x_shape)
    y = np.empty(shape=x_shape)

    # load all data in the first position
    dat = loadmat(os.path.join(NYU_PATH, '%s/jnt_tremor.mat' % mode))
    x[0, :, :, :, :] = dat['jnt_tremor']
    y[0, :, :, :, :] = dat['jnt_gt']

    # load the single tremor data one by one
    p = multiprocessing.Pool(64) # multi-process pool
    args = [i for i in range(tra_num)]
    pbar = tqdm(range(tra_num)) # processing bar
    for tra_index in p.imap(get_idx, args):
        file = selected_file_names[tra_index]
        pbar.set_description('Get %d - %s data: %s' % (tra_index + 1, mode, file))

        dat = loadmat(file)

        x[tra_index + 1, :, :, :, :] = dat['jnt_tremor']
        y[tra_index + 1, :, :, :, :] = dat['jnt_gt']

        pbar.update()
    pbar.close()
    p.close()
    p.join()

    # swap the axes between tra index and the Kinect index and reshape
    x = x.swapaxes(0, 1)
    y = y.swapaxes(0, 1)

    # reshape to 3 dimension
    x = x.reshape([-1, 36, 3])
    y = y.reshape([-1, 36, 3])

    # make data and return
    all_num = ((tra_num + 1) * NYU_KINECT_NUM * num) // t_size
    x = x[0:all_num * t_size, :, :].reshape([-1, t_size, 36, 3])
    y = y[0:all_num * t_size, :, :].reshape([-1, t_size, 36, 3])

    return x, y

###################################################################
# Load NYU corrupted data.
###################################################################
def load_corrupt_data(mode='train', t_size=27, sample_interval=100, start_pos=0):
    # corrupted data dir
    dat = loadmat(os.path.join(NYU_PATH, '%s/joint_data.mat' % mode))
    jnt_xyz = dat['joint_xyz']

    dat = loadmat(os.path.join(NYU_PATH, 'corrupt/%s/jnt_corruption_%s.mat' % (mode, NOISE)))
    jnt_corruption = dat['jnt_corruption']

    # some variables
    image_num = jnt_xyz.shape[1]
    kinect_num = jnt_xyz.shape[0]
    joint_num = jnt_xyz.shape[2]
    channel_num = jnt_xyz.shape[3]

    stack_num = image_num // (t_size //  2) - 1 # overlap t_size // 2, stride t_size

    x = np.empty(shape=(stack_num * kinect_num, t_size, joint_num, channel_num))
    y = np.empty(shape=(stack_num * kinect_num, t_size, joint_num, channel_num))

    pbar = tqdm(range(kinect_num * stack_num))  # processing bar
    p = multiprocessing.Pool(64)
    args = [i for i in range(stack_num)]
    # make data and return
    for kinect_idx in range(kinect_num):
        for image_idx in p.imap(get_idx, args):
            pbar.set_description('Get %d - %s data: %d' % (kinect_idx + 1, mode, image_idx))
            start = image_idx * t_size // 2
            end = start + t_size
            idx = kinect_idx * stack_num + image_idx

            x[idx, :, :, :] = jnt_corruption[kinect_idx, start:end, :, :]
            y[idx, :, :, :] = jnt_xyz[kinect_idx, start:end, :, :]

            pbar.update()

    pbar.close()

    p.close()
    p.join()

    return x, y

###################################################################
# Load train mode data: training set and validation set
###################################################################
def load_train_data(t_size=27, validation_rate=0.15, sample_interval=10, start_pos=0):
    # get nyu data
    x, y = load_corrupt_data(mode='train', t_size=t_size, sample_interval=sample_interval, start_pos=start_pos)

    # split the data to training data and validation data
    x_train, x_valid, y_train, y_valid = train_test_split(x, y,
                                                          test_size=validation_rate,
                                                          random_state=1024)

    return x_train, y_train, x_valid, y_valid

############################################################
# search file from folder
###########################################################
def search_file_from_folder(folder_path, file_key):
    files_selected = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.index(file_key) != -1:
                files_selected.append(file)

    return files_selected

############################################################
# search optimal model and delete the extra model
###########################################################
def search_acceptable_model(model, x_test, y_test):
    # search an acceptable loss
    loss_all = np.empty(shape=(x_test.shape[0], 1))
    p = multiprocessing.Pool(64)
    args = [i for i in range(x_test.shape[0])]
    pbar = tqdm(range(x_test.shape[0]))
    for i in p.imap(get_idx, args):
        pbar.update()

        loss = model.evaluate(x_test[i:i + 1, :, :, :], y_test[i:i + 1, :, :, :], verbose=0)
        loss_all[i] = loss

        pbar.set_description('Loss %d: %.4f' % (i, loss))

        if loss < ACCEPTENCE_LOSS:
            pbar.close()
            break

    p.close()
    p.join()

    return i, loss, loss_all

############################################################
#  Plot training history
###########################################################
def plot_hat(model, x_test, y_test, y_test_hat, fig_path):
    i, loss, loss_all = search_acceptable_model(model, x_test, y_test)
    # judge
    if i == x_test.shape[0] - 1:
        opt_idx = OPT_IDX
    else:
        opt_idx = i

    log.logger.info('Acceptable loss %d: %.4f' % (i, loss))
    log.logger.info('Selected %d, loss = %.4f' % (opt_idx, loss_all[opt_idx]))

    fig_path = os.path.join(fig_path, str(opt_idx))
    if os.path.exists(fig_path) == False:
        os.mkdir(fig_path)

    # delete the remaining files
    video_name = os.path.join(fig_path, '%d-tremor_result.mp4' % opt_idx)
    if os.path.exists(video_name) == False:
        pass
    # delete the orignal files before making the new file
    delete_file_folder(fig_path)

    # save video and figs of result
    plot_save_video(x_test, y_test, y_test_hat, fig_path, opt_idx, video_name)

    # save trajectory
    plot_save_trajectory(x_test, y_test, y_test_hat, fig_path, opt_idx)

    # save mse error
    plot_mse_error(x_test, y_test, y_test_hat, fig_path, opt_idx)

############################################################
# save training phase
###########################################################
def save_training_phase(history):
    # save history
    model_history = "./output/history/training_history-" + time.strftime("%Y_%m_%d-%H_%M_%S", time.localtime()) + ".csv"
    pd.DataFrame(history.history).to_csv(model_history, index=False)
    log.logger.info("Save history: %s" % model_history)

    # plot history
    plot_history(model_history)

############################################################
# Dealing with the test phase: saving figs and search optimal model
###########################################################
def deal_learnable_adj(A, fig_path):
    for A_num in range(A.shape[0]):
        A_ = A[A_num, :30, :30]
        plot_adj(adj=A_, fig_path=fig_path, num=A_num + 1)

    # only using the top 30 joints of NYU hand model
    A_ = np.zeros(shape=(30, 30))
    for A_num in range(A.shape[0]):
        A_ = A_ + A[A_num, :30, :30]
    plot_adj(adj=A_, fig_path=fig_path, num=0)

# Optimal model dealing
def deal_opt(model, model_path, opt_model, x_test, y_test, search_opt, loss):
    # delete the non-optimal model and leave the optimal model
    if search_opt & (os.path.normcase(opt_model) == os.path.normcase(model_path)):
        log.logger.info('%s is the optimal model' % model_path)

    if search_opt & (os.path.normcase(opt_model) != os.path.normcase(model_path)):
        if os.path.exists(opt_model) == False:
            loss_opt = np.inf

        else:
            model.load_weights(opt_model)
            log.logger.info('Load current pre-trained model')

            y_opt_hat = model.predict(x_test)
            loss_opt = model.evaluate(x_test, y_test, verbose=1)
            log.logger.info('Test loss (Optimal model): %f' % loss_opt)
            log.logger.info('Test loss (test model): %f' % loss)

        # optimal model update
        if loss_opt > loss:
            if os.path.exists(opt_model):
                os.remove(opt_model)
                log.logger.info('Delete the current optimal model: %s' % opt_model)

            os.rename(model_path, opt_model)
            log.logger.info('%s -> %s' % (model_path, opt_model))

        else:
            os.remove(model_path)
            log.logger.info('Delete the test model %s' % model_path)

def save_loss(mse_pose, mse_bone_len_direct, mse_bone_len_indirect):

    with open(BEST_RES_LOSS_PATH, 'a', encoding='utf-8') as f:
        f.write('%s\t%s\t%s\t%s\t%.4f\t%.4f\t%.4f\n' %
                (TIME_START, NOISE, STRATEGY, ATTENTION,
                 mse_pose, mse_bone_len_direct, mse_bone_len_indirect))

        log.logger.info('%s\t%s\t%s\t%s\t%.4f\t%.4f\t%.4f' %
                (TIME_START, NOISE, STRATEGY, ATTENTION,
                 mse_pose, mse_bone_len_direct, mse_bone_len_indirect))

        f.close()

def deal_test_phase(model, model_path, opt_model, x_test, y_test, y_test_hat, search_opt, loss):
    # deal opt searching
    deal_opt(model, model_path, opt_model, x_test, y_test, search_opt, loss)

    # pose mse
    mse_pose = caculate_all_mse(y_test, y_test_hat)
    log.logger.info('Hand pose error: %f' % mse_pose)

    # bone length evaluation
    mse_bone_len_direct = caculate_all_bone_length_error(y_test, y_test_hat)
    log.logger.info('Bone length error: %f' % mse_bone_len_direct)

    # Symmetrical neighbor length evaluation
    mse_bone_len_indirect = caculate_all_bone_length_error(y_test, y_test_hat, edge='indirect')
    log.logger.info('Symmetrical neighbor error: %f' % mse_bone_len_indirect)

    # save loss
    save_loss(mse_pose, mse_bone_len_direct, mse_bone_len_indirect)

    # plot model
    fig_path = FIG_PATH
    if os.path.exists(fig_path) == False:
        os.mkdir(fig_path)

    plot_model(model.build_graph(input_shape=x_test.shape[1:]), to_file=fig_path + '/model_stgcn.pdf', show_shapes=True)

    # plot learnable adjacency matrix
    if ATTENTION == 'A_M':
        A = model.trainable_weights[6]
        deal_learnable_adj(A, fig_path)
    else:
        A = model.A
        deal_learnable_adj(A, fig_path)

    # plot result
    plot_hat(model, x_test=x_test, y_test=y_test,
             y_test_hat=y_test_hat, fig_path=fig_path)



if __name__ == '__main__':
    plot_history('./output/history/training_history-2021_03_29-20_33_08.csv')

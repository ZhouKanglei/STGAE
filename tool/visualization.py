import os

import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = True

from mpl_toolkits.axes_grid1 import host_subplot
from matplotlib.animation import FFMpegWriter

from sklearn.metrics import mean_squared_error
from data.NYU.nyu import show_joint_skeleton, jnt_color
from config.config import log


############################################################
# Visualization tool funcs
###########################################################

# Plot model fit history loss & val_loss
def plot_history(history_csv_path):
    df = pd.read_csv(history_csv_path)
    # loss and val_loss
    loss = df.iloc[:, 0].values
    val_loss = df.iloc[:, 1].values

    if df.shape[1] == 3:
        # learning rate
        lr = df.iloc[:, 2].values

        fig = plt.figure()
        host = host_subplot(111)  # row=1 col=1 first pic
        plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window
        par1 = host.twinx()  #

        # set labels
        host.set_xlabel("Epoch")
        host.set_ylabel("Loss")
        par1.set_ylabel("Learning rate")


        # plot curves
        host.plot(range(len(loss)), loss, label="Training loss")
        host.plot(range(len(val_loss)), val_loss, label="Validation loss")
        par1.plot(range(len(lr)), lr, label="Learning rate")

        host.legend()
        plt.title('Loss \& Learning rate')

    else:
        fig = plt.figure()

        plt.plot(range(len(loss)), loss, label="Training loss")
        plt.plot(range(len(val_loss)), val_loss, label="Validation loss")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title('Loss')

    # plt.show()

    # save figure
    fig_name = '%s.pdf' % history_csv_path[:-4]
    fig.savefig(fig_name, transparent=True, dpi=300)
    print('Save history plot: %s' % fig_name)


############################################################
# plot test hat and test truth
###########################################################
def plot_save_video(x_test, y_test, y_test_hat, fig_path, opt_idx, video_name):
    # made the video of result
    metadata = dict(title='Hand tremor cleaning result', artist='Matplotlib', comment='Paper showing!')
    writer = FFMpegWriter(fps=15, metadata=metadata)
    fig = plt.figure(figsize=(12, 5))

    # judge the fig path
    fig_path = os.path.join(fig_path, 'single')
    if os.path.exists(fig_path) == False:
        os.makedirs(fig_path)

    with writer.saving(fig=fig,
                       outfile=video_name,
                       dpi=300):
        plt.axis('off')

        # save every fig
        for t in range(x_test.shape[1]):
            ax_1 = plt.subplot(131, projection='3d')
            ax_2 = plt.subplot(132, projection='3d')
            ax_3 = plt.subplot(133, projection='3d')

            show_joint_skeleton(ax=ax_1, jnt_xyz=x_test[opt_idx, t, :, :], jnt_color=jnt_color)
            show_joint_skeleton(ax=ax_1, jnt_xyz=y_test[opt_idx, t, :, :], jnt_color=jnt_color, alpha=0.25)

            ax_1.set_title(
                'Input (MSE = %.4f)' % (mean_squared_error(x_test[opt_idx, t, :, :], y_test[opt_idx, t, :, :])))

            show_joint_skeleton(ax=ax_2, jnt_xyz=y_test_hat[opt_idx, t, :, :], jnt_color=jnt_color)
            show_joint_skeleton(ax=ax_2, jnt_xyz=y_test[opt_idx, t, :, :], jnt_color=jnt_color, alpha=0.25)
            ax_2.set_title(
                'Output (MSE = %.4f)' % (mean_squared_error(y_test_hat[opt_idx, t, :, :], y_test[opt_idx, t, :, :])))

            show_joint_skeleton(ax=ax_3, jnt_xyz=y_test[opt_idx, t, :, :], jnt_color=jnt_color)
            ax_3.set_title('Ground truth')

            fig_name = os.path.join(fig_path, '%d-%d.pdf' % (opt_idx, t))
            fig.savefig(fig_name, transparent=True, bbox_inches='tight')

            log.logger.info('Save file: %s' % fig_name)

            ax_1.axis('off')
            ax_2.axis('off')
            ax_3.axis('off')

            writer.grab_frame()

        plt.close()

        log.logger.info('Save video: %s' % video_name)

    # save every fig in single plot
    for t in range(x_test.shape[1]):
        # save input
        fig = plt.figure(figsize=(4, 3))
        ax_1 = plt.subplot(111, projection='3d')

        show_joint_skeleton(ax=ax_1, jnt_xyz=x_test[opt_idx, t, :, :], jnt_color=jnt_color)
        show_joint_skeleton(ax=ax_1, jnt_xyz=y_test[opt_idx, t, :, :], jnt_color=jnt_color, alpha=0.25)

        ax_1.set_title(
            'Input (MSE = %.4f)' % (mean_squared_error(x_test[opt_idx, t, :, :], y_test[opt_idx, t, :, :])))

        fig_name = os.path.join(fig_path, '%d-%d-input.pdf' % (opt_idx, t))
        fig.savefig(fig_name, transparent=True, bbox_inches='tight')

        log.logger.info('Save file: %s' % fig_name)
        plt.close()

        # save predict
        fig = plt.figure(figsize=(4, 3))
        ax_2 = plt.subplot(111, projection='3d')

        show_joint_skeleton(ax=ax_2, jnt_xyz=y_test_hat[opt_idx, t, :, :], jnt_color=jnt_color)
        show_joint_skeleton(ax=ax_2, jnt_xyz=y_test[opt_idx, t, :, :], jnt_color=jnt_color, alpha=0.25)
        ax_2.set_title(
            'Output (MSE = %.4f)' % (
                mean_squared_error(y_test_hat[opt_idx, t, :, :], y_test[opt_idx, t, :, :])))

        fig_name = os.path.join(fig_path, '%d-%d-predict.pdf' % (opt_idx, t))
        fig.savefig(fig_name, transparent=True, bbox_inches='tight')

        log.logger.info('Save file: %s' % fig_name)
        plt.close()

        # save ground truth
        fig = plt.figure(figsize=(4, 3))
        ax_3 = plt.subplot(111, projection='3d')

        show_joint_skeleton(ax=ax_3, jnt_xyz=y_test[opt_idx, t, :, :], jnt_color=jnt_color)
        ax_3.set_title('Ground truth')

        fig_name = os.path.join(fig_path, '%d-%d-gt.pdf' % (opt_idx, t))
        fig.savefig(fig_name, transparent=True, bbox_inches='tight')

        log.logger.info('Save file: %s' % fig_name)
        plt.close()

############################################################
# plot and save the trajectory
###########################################################
def plot_save_trajectory(x_test, y_test, y_test_hat, fig_path, opt_idx):
    # plot x-axis position of index finger tip
    index_finger_idx = 18
    num_axis = 3
    axis_name = ['x', 'y', 'z']
    x = [i + 1 for i in range(x_test.shape[1])]

    # make folder
    fig_axis_path = os.path.join(fig_path, 'axis')
    if os.path.exists(fig_axis_path) == False:
        os.makedirs(fig_axis_path)

    # save entire figure
    fig = plt.figure(figsize=(4.5 * num_axis, 3))

    for num in range(num_axis):
        y_test_index_fingtip = y_test[opt_idx, :, index_finger_idx, num]
        y_test_hat_index_fingtip = y_test_hat[opt_idx, :, index_finger_idx, num]
        x_test_index_fingtip = x_test[opt_idx, :, index_finger_idx, num]

        ax_1 = plt.subplot(1, num_axis, num + 1)
        ax_1.plot(x, y_test_index_fingtip, 'r', label='Ground truth')
        ax_1.plot(x, x_test_index_fingtip, 'g', label='Input')
        ax_1.plot(x, y_test_hat_index_fingtip, 'b', label='Output')

        ax_1.set_xlabel('Frame No.')
        ax_1.set_ylabel('$%s$-axis' % axis_name[num])
        ax_1.set_title('Trajectory of $%s$-axis' % axis_name[num])

        ax_1.legend()


    fig_name = os.path.join(fig_path, 'axis/%d-axis.pdf' % opt_idx)
    if os.path.exists(os.path.dirname(fig_name)) == False:
        os.makedirs(os.path.dirname(fig_name))

    fig.savefig(fig_name, bbox_inches='tight', transparent=True)
    log.logger.info('Save trajectory: %s' % fig_name)

    # save entire figure separately
    for num in range(num_axis):
        fig = plt.figure(figsize=(4, 3))

        y_test_index_fingtip = y_test[opt_idx, :, index_finger_idx, num]
        y_test_hat_index_fingtip = y_test_hat[opt_idx, :, index_finger_idx, num]
        x_test_index_fingtip = x_test[opt_idx, :, index_finger_idx, num]

        ax_1 = plt.subplot(1, 1, 1)
        ax_1.plot(x, y_test_index_fingtip, 'r', label='Ground truth')
        ax_1.plot(x, x_test_index_fingtip, 'g', label='Input')
        ax_1.plot(x, y_test_hat_index_fingtip, 'b', label='Output')

        ax_1.set_xlabel('Frame No.')
        ax_1.set_ylabel('$%s$-axis' % axis_name[num])
        ax_1.set_title('Trajectory of $%s$-axis' % axis_name[num])

        ax_1.legend()

        fig_name = os.path.join(fig_path, 'axis/%d-axis-%s.pdf' % (opt_idx, axis_name[num]))
        if os.path.exists(os.path.dirname(fig_name)) == False:
            os.makedirs(os.path.dirname(fig_name))

        fig.savefig(fig_name, bbox_inches='tight', transparent=True)
        log.logger.info('Save trajectory: %s' % fig_name)

        plt.close()

    # save single figure
    for num in range(num_axis):
        y_test_index_fingtip = y_test[opt_idx, :, index_finger_idx, num]
        y_test_hat_index_fingtip = y_test_hat[opt_idx, :, index_finger_idx, num]
        x_test_index_fingtip = x_test[opt_idx, :, index_finger_idx, num]

        # Ground truth
        fig = plt.figure(figsize=(4, 3))
        ax_1 = plt.subplot(1, 1, 1)

        ax_1.plot(x, y_test_index_fingtip, 'r', label='Ground truth')

        ax_1.set_xlabel('Frame No.')
        ax_1.set_ylabel('$%s$-axis' % axis_name[num])
        ax_1.set_title('Trajectory of $%s$-axis' % axis_name[num])

        ax_1.legend()

        fig_name = os.path.join(fig_path, 'axis/%d-axis-%s-gt.pdf' % (opt_idx, axis_name[num]))
        fig.savefig(fig_name, bbox_inches='tight', transparent=True)
        log.logger.info('Save trajectory: %s' % fig_name)

        # Input
        fig = plt.figure(figsize=(4, 3))
        ax_1 = plt.subplot(1, 1, 1)

        ax_1.plot(x, x_test_index_fingtip, 'g', label='Input')
        ax_1.plot(x, y_test_index_fingtip, 'r', label='Ground truth')
        ax_1.fill_between(x, x_test_index_fingtip, y_test_index_fingtip, facecolor='g', alpha=0.25)

        ax_1.set_xlabel('Frame No.')
        ax_1.set_ylabel('$%s$-axis' % axis_name[num])
        ax_1.set_title('Trajectory of $%s$-axis' % axis_name[num])

        ax_1.legend()

        fig_name = os.path.join(fig_path, 'axis/%d-axis-%s-input.pdf' % (opt_idx, axis_name[num]))
        fig.savefig(fig_name, bbox_inches='tight', transparent=True)
        log.logger.info('Save trajectory: %s' % fig_name)

        # predict
        fig = plt.figure(figsize=(4, 3))
        ax_1 = plt.subplot(1, 1, 1)

        ax_1.plot(x, y_test_hat_index_fingtip, 'b', label='Output')
        ax_1.plot(x, y_test_index_fingtip, 'r', label='Ground truth')
        ax_1.fill_between(x, y_test_index_fingtip, y_test_hat_index_fingtip, facecolor='b', alpha=0.25)

        ax_1.set_xlabel('Frame No.')
        ax_1.set_ylabel('$%s$-axis' % axis_name[num])
        ax_1.set_title('Trajectory of $%s$-axis' % axis_name[num])

        ax_1.legend()

        fig_name = os.path.join(fig_path, 'axis/%d-axis-%s-predict.pdf' % (opt_idx, axis_name[num]))
        fig.savefig(fig_name, bbox_inches='tight', transparent=True)

        log.logger.info('Save trajectory: %s' % fig_name)

        plt.close()

############################################################
# plot and save the learnable adjacency matrix
###########################################################
def plot_adj(adj, fig_path, num=0):
    fig = plt.figure(figsize=(6, 4.5))
    ax = plt.subplot(111)

    sns.heatmap(adj, fmt="d", cmap=plt.cm.bwr, ax=ax)

    label_y = ax.get_yticklabels()
    plt.setp(label_y, rotation=360, horizontalalignment='right')
    label_x = ax.get_xticklabels()
    plt.setp(label_x, rotation=45, horizontalalignment='right')

    ax.set_xlabel('Hand joint index')
    ax.set_ylabel('Hand joint index')
    if num != 0:
        ax.set_title('$\mathbf{A}_%d \odot \mathbf{M}_%d$' % (num, num))
        fig_name = os.path.join(fig_path, 'adj_heatmap_%d.pdf' % num)
    else:
        ax.set_title('$\sum_k^K (\mathbf{A}_k \odot \mathbf{M}_k)$')
        fig_name = os.path.join(fig_path, 'adj_heatmap.pdf')


    fig.savefig(fig_name, bbox_inches='tight', transparent=True)

    log.logger.info('Save heatmap: %s' % fig_name)

############################################################
# plot and save mse error curve
###########################################################
def plot_mse_error(x_test, y_test, y_test_hat, fig_path, opt_idx):
    y_test_opt = y_test[opt_idx, :, :, :]
    y_test_hat_opt = y_test_hat[opt_idx, :, :, :]
    x_test_opt = x_test[opt_idx, :, :, :]

    mse_input = []
    mse_test = []
    for i  in range(y_test.shape[1]):
        mse_input.append(mean_squared_error(x_test_opt[i, :, :], y_test_opt[i, :, :]))
        mse_test.append(mean_squared_error(y_test_hat_opt[i, :, :], y_test_opt[i, :, :]))

    # plot
    x = [i + 1 for i in range(y_test.shape[1])]
    fig = plt.figure(figsize=(4, 3))
    ax = plt.subplot(111)
    ax.plot(x, mse_input, 'r', label='Input')
    ax.plot(x, mse_test, 'b', label='Output')

    ax.set_xlabel('Frame No.')
    ax.set_ylabel('MSE')
    ax.set_title('MSE curve')

    ax.legend()

    fig_name = os.path.join(fig_path, '%d-mse_error_curve.pdf' % (opt_idx))
    plt.savefig(fig_name, bbox_inches='tight', transparent=True)

    log.logger.info('Save MSE curve plot: %s' % fig_name)
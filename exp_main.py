from tool.tools import *
from model.stgae import STGAE

def train(log_dir=OUTPUT_LOG_PATH, t_size=T_SIZE,
          validation_rate=VALIDATION_RATE,
          sample_interval=50, start_pos=0):

    # get nyu data
    x_train, y_train, x_valid, y_valid = load_train_data(t_size=t_size,
                                                       validation_rate=validation_rate,
                                                       sample_interval=sample_interval,
                                                       start_pos=start_pos)

    # create model
    model = STGAE(filters=3)

    # print summary
    model(x_train[:1, :, :, :])
    print(model.summary())

    # load pre-trained model weights.
    pre_trained_model = OPT_MODEL_PATH
    if os.path.exists(pre_trained_model):
        model.load_weights(pre_trained_model)
        log.logger.info('Load pre-trained model %s' % pre_trained_model)

    # compile
    sgd = tf.keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    adam = tf.keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer='adam', loss='mse')

    # callbacks
    EarlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                 patience=PATIENCE, verbose=1, mode='auto')

    Reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                  factor=0.80, patience=1,
                                                  verbose=1, mode='auto',
                                                  epsilon=0.0001,
                                                  cooldown=0, min_lr=MIN_LR)

    TensorBoard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True)

    # Tem best weights
    best_weight = TEMP_BEST_MODEL
    checkpoint = tf.keras.callbacks.ModelCheckpoint(best_weight,
                                 monitor="val_loss",
                                 mode='min',
                                 save_weights_only=True,
                                 save_best_only=True,
                                 verbose=1,
                                 period=1)

    # model fit
    history = model.fit(x_train, y_train,
                        epochs=MAX_EPOCH,
                        batch_size=32,
                        verbose=1,
                        validation_data=(x_valid, y_valid),
                        callbacks=[TensorBoard, Reduce, EarlyStop, checkpoint])

    # save training weight and history
    save_training_phase(history)
    model_name = best_weight

    return model_name

def test(model_path, t_size=T_SIZE, search_opt=False, opt_model=OPT_MODEL_PATH):
    # judge
    if os.path.exists(model_path) == False:
        log.logger.info('%s is not existing!!!' % model_path)
        return

    # get nyu data
    x_test, y_test = load_corrupt_data(mode='test', t_size=t_size, sample_interval=100)

    # model
    model = STGAE(filters=3)

    # print summary
    model(x_test[:1, :, :, :])
    print(model.summary())

    # model compile
    model.compile(optimizer='adam', loss="mse")
    log.logger.info('Model compile...')

    # load weight
    model.load_weights(model_path)
    log.logger.info('Load model %s' % model_path)

    # predict
    y_test_hat = model.predict(x_test)
    loss = model.evaluate(x_test, y_test, verbose=1)
    log.logger.info('Test loss: %f' % loss)

    # deal with test phase: search optimal model and clean the others, save inference result
    deal_test_phase(model, model_path, opt_model, x_test, y_test, y_test_hat, search_opt, loss)

def main():
    t_size = T_SIZE # Temporal size
    validation_rate = 0.15 # validation rate
    search_opt = True  # select the optimal model as the pre-trained model

    # train or test
    option = OPTION
    if option == True:
        # delete the existing training log before training
        log_dir = OUTPUT_LOG_PATH
        delete_file_folder(log_dir)

        # train
        train_model = train(log_dir=log_dir, t_size=t_size,
                            validation_rate=validation_rate,
                            sample_interval=50)

        # test train model
        test(model_path=train_model, t_size=t_size, search_opt=search_opt)

    else:
        # test
        test_model_path = OPT_MODEL_PATH  # test model path
        test(model_path=test_model_path, t_size=t_size, search_opt=search_opt)

if __name__ == '__main__':
    main()
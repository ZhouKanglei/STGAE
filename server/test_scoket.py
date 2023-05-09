
import logging
import traceback
from threading import Thread
import socket
import threading
import time
import struct
import numpy as np
np.random.seed(1024)

from kito import reduce_keras_model

logger = logging

float_num = 26 * 3
type_bytes_num = 4
batch = 35
bytes_num = float_num * type_bytes_num * batch

temporal_size = 36
mrtk_joint_num = 26
scale_factor = 5000.0

buffer_mrtk = np.zeros(shape=(1, temporal_size + batch - 1, mrtk_joint_num, 3))
buffer_input = np.zeros(shape=(1, temporal_size + batch - 1, 20, 3))
buffer_input_model = np.zeros(shape=(batch, temporal_size, 20, 3))
buffer_cnt = 0
ground_truth = []

MRTK_selected_joints = [1, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16,
                        18, 19, 20, 21, 23, 24, 25, 26]

output_frame_idx = 17
average_size = 1

from model.stgae import STGAE

# model
model = STGAE(filters=3)
# model = reduce_keras_model(model)

# print summary
model(buffer_input_model)
print(model.summary())

# model compile
model.compile(optimizer='adam', loss="mse")
print('Model compile...')

# load weight
model_path = './output/weight_mrtk/best_weights-Uniform_full-spatial-A+B+C.h5'
model.load_weights(model_path)
print('Load model %s' % model_path)

def denoising(msg_batch):
    global buffer_cnt
    buffer_cnt += batch

    # slide forward one frame
    buffer_mrtk[:, 0:-batch, :, :] = buffer_mrtk[:, batch:, :, :]
    buffer_input[:, 0:-batch, :, :] = buffer_input[:, batch:, :, :]

    # fill buffers
    input = np.array(msg_batch).reshape((1, batch, mrtk_joint_num, 3))
    buffer_mrtk[:, -batch:, :, :] = input

    # select mrtk joints from HoloLens
    for i in range(mrtk_joint_num):
        if i + 1 in MRTK_selected_joints:
            idx = MRTK_selected_joints.index(i + 1)
            buffer_input[:, -batch:, idx, :] = input[:, :, i, :]

    send_frame = buffer_mrtk[:, output_frame_idx:(output_frame_idx + batch), :, :]

    global ground_truth
    ground_truth = np.array(send_frame.reshape((-1,)), dtype='float').tolist()

    for i in range(batch):
        buffer_input_model[i:i + 1, :, :, :] = buffer_input[:, i:i + temporal_size, :, :]
    # print(buffer_input_model.shape)

    # predict
    if buffer_cnt >= temporal_size:
        # print('Predict', end=' ')
        start_inference = time.time()
        outputs = model.predict(buffer_input_model * scale_factor)
        end_inference = time.time()
        print("Inference: %.4fs" % (end_inference - start_inference))

        outputs /= scale_factor

        # print('-> processing', end=' ')
        output_frame = outputs[:, output_frame_idx:(output_frame_idx + 1), :, :]

        for average_idx in range(1, average_size + 1):
            output_frame += outputs[:, output_frame_idx - average_idx:(output_frame_idx + 1 - average_idx), :, :]
            output_frame += outputs[:, output_frame_idx + average_idx:(output_frame_idx + 1 + average_idx), :, :]

        output_frame /= (average_size * 2 + 1)

        output_frame = output_frame.reshape((1, batch, len(MRTK_selected_joints), 3))

        for i in range(mrtk_joint_num):
            if i + 1 in MRTK_selected_joints and i != 0:
                idx = MRTK_selected_joints.index(i + 1)
                # print(idx, end=' ')
                send_frame[:, :, i, :] = output_frame[:, :, idx, :]
        # print('-> done!')

    send_frame = send_frame.reshape((-1,))
    send_frame = np.array(send_frame, dtype='float')

    return send_frame.tolist()

def float_to_bin(num):
    return struct.pack('f', num)

def bin_to_float(binary):
    return struct.unpack('f', binary)[0]

class ElevStatusWsServer(Thread):
    def __init__(self, host, port):
        Thread.__init__(self, name="ElevStatusServer")
        self.host = host
        self.port = port
        self.logger = logger
        self.seqNo = 1
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        print(self.host)
        print(self.server)

    def run(self):
        self.doConnect()
        while True:
            global buffer_cnt
            buffer_cnt = 0
            try:
                client, addr = self.server.accept()
                #threading.Thread(target=self.send_msg, args=(client, addr, self.data)).start()
                threading.Thread(target=self.recv_msg, args=(client, addr)).start()
                # print(threading.enumerate())
            except socket.error:
                traceback.print_exc()
                print('socket connect error, doing connect 2s host/port:{}/{}'.format(self.host, self.port))
                time.sleep(2)
            except Exception as e:
                print('other error occur:{}'.format(e))
                time.sleep(2)

    def doConnect(self):
        while True:
            try:
                self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.server.bind((self.host, self.port))
                self.server.listen(5)
                print('-----------------------------------------------------------')
                print("esWsServer host: {}/port: {} started listen...".format(self.host, self.port))
                print('-----------------------------------------------------------')
                break
            except Exception as e:
                time.sleep(1)
                print('start ws server error:{}'.format(str(e)))
                traceback.print_exc()

    def recv_msg(self, client, addr):
        try:
            print('Accept new connection from {0}'.format(addr))
            global buffer_cnt
            buffer_cnt = 0
            cnt = 0
            while True:
                start_time = time.time()
                # Recieve data
                # data = client.recv(bytes_num)
                data = b''
                data += client.recv(bytes_num)

                while len(data) != bytes_num:
                    len_recv = bytes_num - len(data)
                    data += client.recv(len_recv)

                # deal & send
                if len(data) == bytes_num:
                    msg = []
                    cnt += 1
                    print('Recv msg - {}: '.format(buffer_cnt))
                    for i in range(len(data) // type_bytes_num):
                        byte_item = data[i * type_bytes_num:(i + 1) * type_bytes_num]
                        msg.append(bin_to_float(byte_item))
                        # print('%7.4f ' % (msg[-1]), end='')
                        # if (i + 1) % 18 == 0:
                        #     print()
                        # if (i + 1) % 3 == 0:
                        #     print('\t', end='')
                    print('\nRecv length: {}'.format(len(msg)))

                    # de-noising
                    start_denoising_time = time.time()
                    msg = denoising(msg)
                    end_denoising_time = time.time()

                    # buffer_cnt = 36
                    # send
                    if buffer_cnt >= temporal_size:
                        self.send_msg(client, addr, msg)
                        # for batch_idx in range(batch):
                        #     self.send_msg(client, addr, msg[float_num * batch_idx : float_num * (batch_idx + 1)])

                    end_time = time.time()

                    print("Once total: %.4fs\tDenoising: %.4fs\tOthers: %.4fs" %
                          ((end_time - start_time), (end_denoising_time - start_denoising_time),
                           ((end_time - start_time) - (end_denoising_time - start_denoising_time))))
                else:
                    print("-----", len(data))

        except Exception as e:
            print('Recv msg: {}'.format(e))
            client.close()

    def send_msg(self, client, addr, data):

        try:
            if True:
                msg2Elev = b''
                for i in range(len(data)):

                    msg2Elev += float_to_bin(float(data[i]))
                #     print('%7.4f ' % (data[i]), end='')
                #     if (i + 1) % 18 == 0:
                #         print()
                #     if (i + 1) % 3 == 0:
                #         print('\t', end='')
                # print('\nouput<----------[%d-th]--------->input' % (buffer_cnt + output_frame_idx - temporal_size))
                # for i in range(len(data)):
                #     print('%7.4f ' % (ground_truth[i]), end='')
                #     if (i + 1) % 18 == 0:
                #         print()
                #     if (i + 1) % 3 == 0:
                #         print('\t', end='')
                # print()
                client.sendto(msg2Elev, addr)
                print('Send msg to client [{}]: {}'.format(len(data), addr))

        except Exception as e:
            print('Send msg: {}'.format(e))
            client.close()

if __name__ == '__main__':
    es = ElevStatusWsServer('219.224.168.98', 5000)
    # es = ElevStatusWsServer('192.168.1.105', 8989)
    es.start()


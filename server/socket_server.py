# Server program
import logging
import socket
import struct

def float_to_bin(num):
    return struct.pack('f', num)

def bin_to_float(binary):
    return struct.unpack('f', binary)[0]

def server_program():
    # get the hostname
    host = socket.gethostname()
    print('Host name: ', host)
    port = 5000  # initiate port no above 1024

    server_socket = socket.socket()  # get instance
    # look closely. The bind() function takes tuple as argument
    server_socket.bind(("192.168.1.34", port))  # bind host address and port together

    # configure how many client the server can listen simultaneously
    server_socket.listen(2)
    conn, address = server_socket.accept()  # accept new connection
    print("Connection from: " + str(address))

    cnt = 0.1314159262
    while True:
        cnt = cnt + 1
        send(conn, cnt)
        recieive(conn)

    conn.close()  # close the connection


def recieive(conn):
    # receive data stream. it won't accept data packet greater than 1024 bytes
    data = bin_to_float(conn.recv(4))
    print("Server recieive: ", data)

    return True


def send(conn, cnt):
    # send data
    data = float_to_bin(cnt)
    conn.send(data)  # send data to the client

    return False


if __name__ == '__main__':

    while True:
        server_program()

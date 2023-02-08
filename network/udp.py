'''
For UDP Communication
'''


import socket
import json


class UDPServer:
    def __init__(self):
        self.host = ""
        self.port = 0000
        self.isOpened = False

    def open(self, host='127.0.0.1', port=5555):
        self.host = host
        self.port = port
        self.client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.isOpened = True

    def send(self, data):
        if not self.isOpened:
            return False
        try:
            data_json = json.dumps(data)
            self.client.sendto(data_json.encode('utf-8'), (self.host, self.port))
            return True
        except:
            return False

    def close(self):
        self.client.close()


class UDPClient:
    def __init__(self, host, port, timeout=1.0, buffersize=4096):
        self.host = host
        self.port = port
        self.buffersize = buffersize
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(timeout)
        self.sock.bind((host,port))
        print('listening...')

    def listen(self):
        try:
            data, addr = self.sock.recvfrom(self.buffersize)
        except socket.timeout:
            data = None
        return data
    
    def close(self):
        self.sock.close()



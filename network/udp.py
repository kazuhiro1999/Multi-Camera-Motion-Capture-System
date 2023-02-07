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



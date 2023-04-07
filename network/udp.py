'''
For UDP Communication
'''


import socket
import json


class UDPClient:
    def __init__(self):
        self.host = ""
        self.port = 50000
        self.isOpened = False

    def open(self, host='127.0.0.1', port=50000):
        if self.isOpened:
            return 
        self.host = host
        self.port = port
        self.client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.isOpened = True

    def send(self, data):
        if not self.isOpened:
            return False
        try:
            data_json = json.dumps(data)
            self.client.sendto(data_json.encode('utf-8'), (self.host,self.port))
        except:
            return False
        return True


    def close(self):
        self.client.close()


class UDPServer:
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


if __name__ == '__main__':

    client = UDPClient()
    client.open(host='192.168.11.3', port=50000)

    dummy_data = {'Type': 'MediapipePose', 'TimeStamp': 1680856256809, 'Bones': [{'Name': 'nose', 'Position': {'x': -0.011116458103060722, 'y': 1.6044437885284424, 'z': -0.6124200224876404}}, {'Name': 'left_inner_eye', 'Position': {'x': 0.012894166633486748, 'y': 1.6376038789749146, 'z': -0.6147984266281128}}, {'Name': 'left_eye', 'Position': {'x': 0.025778356939554214, 'y': 1.6366912126541138, 'z': -0.6150716543197632}}, {'Name': 'left_outer_eye', 'Position': {'x': 0.03784007206559181, 'y': 1.6350712776184082, 'z': -0.6158614754676819}}, {'Name': 'right_inner_eye', 'Position': {'x': -0.012577802874147892, 'y': 1.6392228603363037, 'z': -0.6196502447128296}}, {'Name': 'right_eye', 'Position': {'x': -0.01937343180179596, 'y': 1.6395364999771118, 'z': -0.6234374642372131}}, {'Name': 'right_outer_eye', 'Position': {'x': -0.026479944586753845, 'y': 1.6393934488296509, 'z': -0.6270210146903992}}, {'Name': 'left_ear', 'Position': {'x': 0.056409548968076706, 'y': 1.6126108169555664, 'z': -0.6294527649879456}}, {'Name': 'right_ear', 'Position': {'x': -0.03773632273077965, 'y': 1.6210640668869019, 'z': -0.6469115018844604}}, {'Name': 'left_mouth', 'Position': {'x': 0.006846425123512745, 'y': 1.5700281858444214, 'z': -0.6189956665039062}}, {'Name': 'right_mouth', 'Position': {'x': -0.02959875576198101, 'y': 1.5734084844589233, 'z': -0.6218287944793701}}, {'Name': 'left_shoulder', 'Position': {'x': 0.20361730456352234, 'y': 1.3948017358779907, 'z': -0.6855918765068054}}, {'Name': 'right_shoulder', 'Position': {'x': -0.22030293941497803, 'y': 1.4148046970367432, 'z': -0.6968747973442078}}, {'Name': 'left_elbow', 'Position': {'x': 0.2758201062679291, 'y': 1.0843483209609985, 'z': -0.7275527119636536}}, {'Name': 'right_elbow', 'Position': {'x': -0.2792728841304779, 'y': 1.1580239534378052, 'z': -0.6920138001441956}}, {'Name': 'left_wrist', 'Position': {'x': 0.36156657338142395, 'y': 0.8565720915794373, 'z': -0.7765185236930847}}, {'Name': 'right_wrist', 'Position': {'x': -0.34756919741630554, 'y': 0.941004753112793, 'z': -0.7106412649154663}}, {'Name': 'left_outer_hand', 'Position': {'x': 0.3984526991844177, 'y': 0.8010297417640686, 'z': -0.7828853726387024}}, {'Name': 'right_outer_hand', 'Position': {'x': -0.3688482940196991, 'y': 0.8782175779342651, 'z': -0.7340568900108337}}, {'Name': 'left_hand_tip', 'Position': {'x': 0.3786296546459198, 'y': 0.8086351752281189, 'z': -0.7745900750160217}}, {'Name': 'right_hand_tip', 'Position': {'x': -0.3542378544807434, 'y': 0.8883799910545349, 'z': -0.7280455231666565}}, {'Name': 'left_inner_hand', 'Position': {'x': 0.35440170764923096, 'y': 0.8296211361885071, 'z': -0.7798119187355042}}, {'Name': 'right_inner_hand', 'Position': {'x': -0.3298224210739136, 'y': 0.9106754660606384, 'z': -0.727232038974762}}, {'Name': 'left_hip', 'Position': {'x': 0.06427066773176193, 'y': 0.8169257640838623, 'z': -0.7106735110282898}}, {'Name': 'right_hip', 'Position': {'x': -0.16514137387275696, 'y': 0.8317490816116333, 'z': -0.7077386975288391}}, {'Name': 'left_knee', 'Position': {'x': 0.03577866032719612, 'y': 0.44051605463027954, 'z': -0.7292656302452087}}, {'Name': 'right_knee', 'Position': {'x': -0.2030211240053177, 'y': 0.46733152866363525, 'z': -0.618664562702179}}, {'Name': 'left_ankle', 'Position': {'x': 0.008038154803216457, 'y': 0.0741526260972023, 'z': -0.760811448097229}}, {'Name': 'right_ankle', 'Position': {'x': -0.2440274953842163, 'y': 0.08789956569671631, 'z': -0.5662990212440491}}, {'Name': 'left_heel', 'Position': {'x': -0.016598012298345566, 'y': 0.010091052390635014, 'z': -0.7752141356468201}}, {'Name': 'right_heel', 'Position': {'x': -0.22844943404197693, 'y': 0.00946690235286951, 'z': -0.5915697813034058}}, {'Name': 'left_toe', 'Position': {'x': 0.07943502068519592, 'y': 0.001009325380437076, 'z': -0.6337029337882996}}, {'Name': 'right_toe', 'Position': {'x': -0.3163985013961792, 'y': 0.043365150690078735, 'z': -0.40068718791007996}}]}
    
    ret = client.send(dummy_data)
    print(ret)

    client.close()
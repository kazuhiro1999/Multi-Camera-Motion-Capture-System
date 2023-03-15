'''
カメラパラメータの設定用サーバ
'''

from flask import *
import subprocess

PYTHON_PATH = r"C:/Users/xr/anaconda3/envs/motion-capture/python.exe"
SCRIPT_PATH = r"C:/Users/xr/esaki/MotionCapture/Multi-Camera-Motion-Capture-System/multi_camera.py"

app = Flask(__name__)

@app.route("/", methods=["GET"])
def root():
    return ""

# 接続テスト
@app.route("/test", methods=["GET"])
def test():
    return "Connection Established"

# カメラ設定
@app.route("/settings", methods=["POST"])
def settings():
    if request.headers['Content-Type'] != 'application/json':
        return jsonify(res='error'), 400
    try:
        data = request.json
        config_path = save_config(data)
        process_start(SCRIPT_PATH, config_path)
        return f"Capture Started at Port:{data['udp_port']}"
    except:
        return "Configuration Failed"
    

def save_config(data):
    path = f"{data['name']}.json"
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"config saved at {path}")
    return path

def process_start(script_path, config_path):
    command = [PYTHON_PATH, script_path, "--config_path", config_path]
    proc = subprocess.Popen(command)
    return proc


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=8888)
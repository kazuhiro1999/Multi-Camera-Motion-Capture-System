'''
カメラパラメータの設定用サーバ
'''

from flask import *
import subprocess


app = Flask(__name__)

@app.route("/", methods=["GET"])
def root():
    return ""

# 接続テスト
@app.route("/test", methods=["GET"])
def test():
    return "Welcome to Server"

# カメラ設定
@app.route("/settings", methods=["POST"])
def settings():
    if request.headers['Content-Type'] != 'application/json':
        return jsonify(res='error'), 400
    try:
        data = request.json
        path = save_config(data)
        process_start("main.py", path)
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
    command = [r"C:/Users/esaki/anaconda3/python.exe", script_path, "--config_path", config_path]
    proc = subprocess.Popen(command)
    return proc


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=8888)
import base64
import cv2
import requests
import json
import os
import subprocess
import sys
import time
from threading import Thread
from kube_resources.pods import create_pod, get_pod, delete_pod


namespace = "mehran"

POD_NAME = "video-classifier"

PORT = 8000

EXPORTER_IP = os.environ["NODE_IP"]
EXPORTER_PORT = 8081
SOURCE_NAME = "Classifier"
IMAGE_NAME = "mehransi/main:pelastic-video-classifier"

directory = os.path.dirname(__file__)

def get_data():
    im = cv2.imread(f"{directory}/ancelotti.jpg")
    # im = cv2.resize(im, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    return base64.b64encode(cv2.imencode(".jpeg",im)[1].tobytes()).decode("utf-8")


def deploy_classifier(next_target_endpoint, inter, intra, cpu, i):
    create_pod(
        f"{POD_NAME}-inter{inter}-intra{intra}-{i}",
        [
            {
                "name": f"{POD_NAME}-container",
                "image": IMAGE_NAME,
                "request_mem": "1G",
                "request_cpu": f"{cpu}",
                "limit_mem": "1G",
                "limit_cpu": f"{cpu}",
                "env_vars": {
                    "NEXT_TARGET_ENDPOINT": next_target_endpoint, 
                    "PORT": f"{PORT}", 
                    "INTEROP_THREADS": f"{inter}",
                    "PYTHONUNBUFFERED": "1",
                },
                "container_ports": [PORT],
            }
        ],
        namespace=namespace,
        labels={"pipeline": "video", "stage": POD_NAME}
    )
    while True:
            time.sleep(0.2)
            pod = get_pod(f"{POD_NAME}-inter{inter}-intra{intra}-{i}", namespace=namespace)
            if pod["pod_ip"] and pod["pod_ip"].lower() != "none":
                break    
    
    while True:
        time.sleep(0.5)
        try:
            response = requests.post(f"http://{pod['pod_ip']}:{PORT}/initialize", 
                data=json.dumps({"threads": intra}),
                headers={
                    'Content-type':'application/json', 
                    'Accept':'application/json'
                }
            )
            break
        except Exception as e:
            print("initialize exception")
            print(e)
            print()
    return pod["pod_ip"]


def send(data, pod_ip, batch, cpu):
    t = time.perf_counter()
    response = requests.post(f"http://{pod_ip}:{PORT}/infer", data=data)
    t = time.perf_counter() - t
    print(t, response.text, "cpu:", cpu, "batch:", batch)


if __name__ == "__main__":
    log_file_path = os.path.dirname(__file__)
    filename = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) + "/exporter.py"
  
    exporter = subprocess.Popen(["python", filename, f"{EXPORTER_PORT}", SOURCE_NAME])
    
    replicas = 8
    cpu = 4
    for inter in [1, cpu]:
        for intra in [1, cpu]:
            input_data = get_data()
            pod_ips = []
            for r in range(replicas):
                pod_ips.append(deploy_classifier(f"{EXPORTER_IP}:{EXPORTER_PORT}", inter, intra, cpu, r))
                requests.post(
                    f"http://{pod_ips[r]}:{PORT}/infer", data=json.dumps([{"data": input_data}])
                )
            time.sleep(0.2)
            for batch in [1, 2, 4]:
                batch_input = []
                for _ in range(batch):
                    batch_input.append({"data": input_data})
                repeat = 0
                data = json.dumps(batch_input)
                while repeat < 8 * 128 + 2 * batch:
                    for r in range(replicas):
                        repeat += 1
                        thread = Thread(target=send, args=(data, pod_ips[r], batch, cpu))
                        thread.start()
                    thread.join()
                    time.sleep(0.1)
                    print()
     
                time.sleep(3)
                requests.post(
                    f"http://{EXPORTER_IP}:{EXPORTER_PORT}/write",
                    data=json.dumps({"inter": inter, "intra": intra, "batch": batch})
                )
                time.sleep(3)
            
            for r in range(replicas):
                delete_pod(f"{POD_NAME}-inter{inter}-intra{intra}-{r}", namespace)
    
    
    os.system(f"kill -9 {exporter.pid}")

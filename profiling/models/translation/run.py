import requests
import json
import os
import subprocess
import time
from threading import Thread
from kube_resources.pods import create_pod, get_pod, update_pod, delete_pod


namespace = "mehran"
os.system(f"mk create ns {namespace}")

POD_NAME = "translation"

PORT = 8000

EXPORTER_IP = os.environ["NODE_IP"]
EXPORTER_PORT = 8086
SOURCE_NAME = "Translation"
IMAGE_NAME = "mehransi/main:pelastic-translation"


def get_data():
    return "こんにちは。私はAIです。"


def deploy_translation(next_target_endpoint, i):
    create_pod(
        POD_NAME + f"-{i}",
        [
            {
                "name": f"{POD_NAME}-container",
                "image": IMAGE_NAME,
                "request_mem": "2G",
                "request_cpu": "1",
                "limit_mem": "2G",
                "limit_cpu": "1",
                "image_pull_policy": "Always",
                "env_vars": {"NEXT_TARGET_ENDPOINT": next_target_endpoint, "PORT": f"{PORT}", "PYTHONUNBUFFERED": "1",},
                "container_ports": [PORT],
            }
        ],
        namespace=namespace,
        labels={"pipeline": "nlp", "stage": POD_NAME}
    )
    while True:
            time.sleep(0.2)
            pod = get_pod(POD_NAME + f"-{i}", namespace=namespace)
            if pod["pod_ip"] and pod["pod_ip"].lower() != "none":
                break    
    
    while True:
        time.sleep(0.5)
        try:
            response = requests.post(f"http://{pod['pod_ip']}:{PORT}/initialize", 
                data=json.dumps({"threads": 1}),
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
    pod_ips = []
    input_data = get_data()
    for r in range(replicas):
        pod_ips.append(deploy_translation(f"{EXPORTER_IP}:{EXPORTER_PORT}", r))
        requests.post(
            f"http://{pod_ips[r]}:{PORT}/infer", data=json.dumps([{"data": input_data}])
        )
    
    
    for cpu in range(1, 7):
        for r in range(replicas):
            update_pod(
                POD_NAME + f"-{r}",
                [
                    {
                        "name": f"{POD_NAME}-container",
                        "image": IMAGE_NAME,
                        "request_cpu": f"{cpu}",
                        "limit_cpu": f"{cpu}",
                    }
                ],
                namespace=namespace,
            )
            response = requests.post(f"http://{pod_ips[r]}:{PORT}/update-threads", data=json.dumps({"threads": cpu}))
            assert json.loads(response.text).get("success") == True
        time.sleep(0.2)
        for batch in range(1, 5):
            batch_input = []
            for _ in range(batch):
                batch_input.append({"data": input_data})
            repeat = 0
            while repeat < 128 * 8 + 2 * batch:
                data = json.dumps(batch_input)
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
                data=json.dumps({"cpu": cpu, "batch": batch})
            )
            time.sleep(3)
    
    for r in range(replicas):
        delete_pod(POD_NAME + f"-{r}", namespace)
    os.system(f"kill -9 {exporter.pid}")

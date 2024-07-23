import requests
import json
import os
import subprocess
import time
from kube_resources.pods import create_pod, get_pod, update_pod, delete_pod


namespace = "mehran"

POD_NAME = "summarizer"

PORT = 8000

EXPORTER_IP = os.environ["NODE_IP"]
EXPORTER_PORT = 8087
SOURCE_NAME = "Summarizer"
IMAGE_NAME = "mehransi/main:pelastic-summarizer"


def get_data():
    return "Hello. I'm an AI."


def deploy_sentiment(next_target_endpoint):
    create_pod(
        POD_NAME,
        [
            {
                "name": f"{POD_NAME}-container",
                "image": IMAGE_NAME,
                "request_mem": "2G",
                "request_cpu": "1",
                "limit_mem": "2G",
                "limit_cpu": "1",
                "env_vars": {"NEXT_TARGET_ENDPOINT": next_target_endpoint, "PORT": f"{PORT}", "PYTHONUNBUFFERED": "1",},
                "container_ports": [PORT],
            }
        ],
        namespace=namespace,
        labels={"pipeline": "nlp", "stage": POD_NAME}
    )
    while True:
            time.sleep(0.2)
            pod = get_pod(POD_NAME, namespace=namespace)
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


if __name__ == "__main__":
    log_file_path = os.path.dirname(__file__)
    filename = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) + "/exporter.py"
  
    exporter = subprocess.Popen(["python", filename, f"{EXPORTER_PORT}", SOURCE_NAME])
    pod_ip = deploy_sentiment(f"{EXPORTER_IP}:{EXPORTER_PORT}")
    input_data = get_data()
    
    requests.post(
        f"http://{pod_ip}:{PORT}/infer", data=json.dumps([{"data": input_data}])
    )
    
    for cpu in range(1, 9):
        update_pod(
            POD_NAME,
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
        response = requests.post(f"http://{pod_ip}:{PORT}/update-threads", data=json.dumps({"threads": cpu}))
        assert json.loads(response.text) == {"success": True}
        time.sleep(0.2)
        for batch in range(1, 9):
            batch_input = []
            for _ in range(batch):
                batch_input.append({"data": input_data})
            for repeat in range(512 // batch + 2 * batch):
                t = time.perf_counter()
                response = requests.post(f"http://{pod_ip}:{PORT}/infer", data=json.dumps(batch_input))
                t = time.perf_counter() - t
            print(t, response.text, "cpu:", cpu, "batch:", batch)
            time.sleep(3)
            requests.post(
                f"http://{EXPORTER_IP}:{EXPORTER_PORT}/write",
                data=json.dumps({"cpu": cpu, "batch": batch})
            )
            time.sleep(3)
    
    
    delete_pod(POD_NAME, namespace)
    os.system(f"kill -9 {exporter.pid}")
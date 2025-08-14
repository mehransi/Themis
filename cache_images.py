from kube_resources.deployments import create_deployment, delete_deployment
from utils import wait_till_pod_is_ready
import time

namespace = "mehran"
images = ["mehransi/main:pelastic-video-classifier", "mehransi/main:pelastic-video-detector", "mehransi/main:pelastic-adapter", "mehransi/main:pelastic-dispatcher"]
if __name__ == "__main__":
    for image in images:
        name = image.split(":")[1]
        create_deployment(
            name,
            replicas=16,
            namespace=namespace,
            labels={"component": "model-server"},
            containers=[
                {
                    "name": f"{name}-container",
                    "image": image,
                    "limit_cpu": 2, 
                    "limit_mem": "2G",
                    "request_cpu": 1, 
                    "container_ports": [8000],
                    "image_pull_policy": "Always",
                    "env_vars": {
                        "NEXT_TARGET_ENDPOINT": "localhost:8000",
                        "PORT": "8000",
                        "URL_PATH": "/predict",
                        "YOLO_OFFLINE": "true",
                        "PYTHONUNBUFFERED": "1"
                    },
                }
            ]
        )

        time.sleep(2)
        wait_till_pod_is_ready(name, namespace, ready_replicas=16)
        delete_deployment(name, namespace=namespace)
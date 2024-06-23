import time
import sys
from kube_resources.deployments import get_deployment


def wait_till_pod_is_ready(deploy_name: str, namespace: str):
    while True:
        time.sleep(0.5)
        try:
            deploy = get_deployment(deploy_name, namespace=namespace)
            if deploy["status"]["ready_replicas"] > 0:
                break
        except Exception as e:
            print("Unexpected error:", e)
            
    return

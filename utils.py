import time
import sys
from kube_resources.pods import get_pod


def wait_till_pod_is_ready(pod_name: str, namespace: str):
    while True:
        time.sleep(0.5)
        try:
            pod = get_pod(pod_name, namespace=namespace)
            if pod.get("pod_ip") and pod["pod_ip"].lower() != "none":
                break
        except:
            print("Unexpected error:", sys.exc_info()[0])
        
            
    return pod["pod_ip"]

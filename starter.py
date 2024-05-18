import requests
import json
import time

from kube_resources.configmaps import create_configmap, get_configmap
from kube_resources.pods import create_pod, get_pod
from kube_resources.services import create_service, get_service

from utils import wait_till_pod_is_ready


namespace = "mehran"

VIDEO_DETECTOR = "video-detector"
VIDEO_CLASSIFIER = "video-classifier"

DISPATCHER_PORT = 8000
DETECTOR_PORT = 8000
CLASSIFIER_PORT = 8000


def deploy_dispatchers():
    create_pod(
        f"{VIDEO_DETECTOR}-dispatcher",
        [
            {
                "name": f"{VIDEO_DETECTOR}-dispatcher-container",
                "image": "mehransi/main:pelastic-dispatcher",
                "request_mem": "1G",
                "request_cpu": "1",
                "limit_mem": "1G",
                "limit_cpu": "1",
                "env_vars": {"DISPATCHER_PORT": f"{DISPATCHER_PORT}"},
                "container_ports": [DISPATCHER_PORT],
            }
        ],
        namespace=namespace,
        labels={"pipeline": "video", "stage": f"{VIDEO_DETECTOR}-dispatcher"}
    )

    create_pod(
        f"{VIDEO_CLASSIFIER}-dispatcher",
        [
            {
                "name": f"{VIDEO_CLASSIFIER}-dispatcher-container",
                "image": "mehransi/main:pelastic-dispatcher",
                "request_mem": "1G",
                "request_cpu": "1",
                "limit_mem": "1G",
                "limit_cpu": "1",
                "env_vars": {"DISPATCHER_PORT": f"{DISPATCHER_PORT}"},
                "container_ports": [DISPATCHER_PORT],
            }
        ],
        namespace=namespace,
        labels={"pipeline": "video", "stage": f"{VIDEO_CLASSIFIER}-dispatcher"}
    )


def deploy_detector(next_target_ip, next_target_port):
    create_pod(
        VIDEO_DETECTOR,
        [
            {
                "name": f"{VIDEO_DETECTOR}-container",
                "image": "mehransi/main:pelastic-video-detector",
                "request_mem": "1G",
                "request_cpu": "1",
                "limit_mem": "1G",
                "limit_cpu": "1",
                "env_vars": {"NEXT_TARGET_IP": next_target_ip, "NEXT_TARGET_PORT": f"{next_target_port}", "PORT": f"{DETECTOR_PORT}"},
                "container_ports": [DETECTOR_PORT],
            }
        ],
        namespace=namespace,
        labels={"pipeline": "video", "stage": VIDEO_DETECTOR}
    )

def deploy_classifier(next_target_ip, next_target_port):
    create_pod(
        VIDEO_CLASSIFIER,
        [
            {
                "name": f"{VIDEO_CLASSIFIER}-container",
                "image": "mehransi/main:pelastic-video-classifier",
                "request_mem": "1G",
                "request_cpu": "1",
                "limit_mem": "1G",
                "limit_cpu": "1",
                "env_vars": {"NEXT_TARGET_IP": next_target_ip, "NEXT_TARGET_PORT": f"{next_target_port}", "PORT": f"{CLASSIFIER_PORT}"},
                "container_ports": [CLASSIFIER_PORT],
            }
        ],
        namespace=namespace,
        labels={"pipeline": "video", "stage": VIDEO_CLASSIFIER}
    )
    

def initialize_dispatchers(detector_dispatcher_ip, detector_ip, classifier_dispatcher_ip, classifier_ip):
    response = requests.post(f"http://{detector_dispatcher_ip}:{DISPATCHER_PORT}/initialize", data=json.dumps({
        "dispatcher_name": f"{VIDEO_DETECTOR}-dispatcher",
        "backends_port": DETECTOR_PORT,
        "batch_size": 4,
        "backends": [
            {"name": "detector1", "ip": detector_ip}
        ]
    }), headers={
        'Content-type':'application/json', 
        'Accept':'application/json'
    })

    response = requests.post(f"http://{classifier_dispatcher_ip}:{DISPATCHER_PORT}/initialize", data=json.dumps({
        "dispatcher_name": f"{VIDEO_CLASSIFIER}-dispatcher",
        "backends_port": CLASSIFIER_PORT,
        "batch_size": 2,
        "backends": [
            {"name": "classifier1", "ip": classifier_ip}
        ]
    }), headers={
        'Content-type':'application/json', 
        'Accept':'application/json'
    })


def initialize_detector(detector_ip):
    while True:
        time.sleep(0.1)
        try:
            response = requests.post(f"http://{detector_ip}:{DETECTOR_PORT}/initialize", headers={
                'Content-type':'application/json', 
                'Accept':'application/json'
            })
            break
        except Exception as e:
            print("initialize detector exception")
            print(e)
            print()


def initialize_classifier(classifier_ip):
    while True:
        time.sleep(0.1)
        try:
            response = requests.post(f"http://{classifier_ip}:{CLASSIFIER_PORT}/initialize", headers={
                'Content-type':'application/json', 
                'Accept':'application/json'
            })
        except Exception as e:
            print("initialize classifier exception")
            print(e)
            print()
            

if __name__ == "__main__":
    deploy_dispatchers()

    detector_dispatcher_ip = wait_till_pod_is_ready(f"{VIDEO_DETECTOR}-dispatcher", namespace)
    classifier_dispatcher_ip = wait_till_pod_is_ready(f"{VIDEO_CLASSIFIER}-dispatcher", namespace)

    deploy_detector(classifier_dispatcher_ip, DISPATCHER_PORT)
    exporter_ip = "localhost"  # FIXME
    deploy_classifier(exporter_ip, 8008)
    detector_ip = wait_till_pod_is_ready(VIDEO_DETECTOR, namespace)
    classifier_ip = wait_till_pod_is_ready(VIDEO_CLASSIFIER, namespace)

    
    initialize_dispatchers(detector_dispatcher_ip, detector_ip, classifier_dispatcher_ip, classifier_ip)
    
    initialize_detector(detector_ip)
    initialize_classifier(classifier_ip)

    

    

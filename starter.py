import requests
import json
import os
import time

from kube_resources.deployments import create_deployment
from kube_resources.services import create_service, get_service

from utils import wait_till_pod_is_ready


namespace = "mehran"

VIDEO_DETECTOR = "video-detector"
VIDEO_CLASSIFIER = "video-classifier"
ADAPTER_DEPLOY_NAME = "pelastic-adapter"

DISPATCHER_PORT = 8000
DETECTOR_PORT = 8000
CLASSIFIER_PORT = 8000
ADAPTER_PORT = 8000

EXPORTER_IP = os.getenv("NODE_IP")
EXPORTER_PORT = 8008

DETECTOR_POD_LABELS = {"pipeline": "video", "component": "model-server", "stage": VIDEO_DETECTOR}
CLASSIFIER_POD_LABELS = {"pipeline": "video", "component": "model-server", "stage": VIDEO_CLASSIFIER}


def deploy_dispatchers():
    detector_dispatcher_labels = {"pipeline": "video", "component": "dispatcher", "stage": f"{VIDEO_DETECTOR}-dispatcher"}
    create_deployment(
        f"{VIDEO_DETECTOR}-dispatcher",
        [
            {
                "name": f"{VIDEO_DETECTOR}-dispatcher-container",
                "image": "mehransi/main:pelastic-dispatcher",
                "request_mem": "1G",
                "request_cpu": "1",
                "limit_mem": "1G",
                "limit_cpu": "1",
                "env_vars": {"DISPATCHER_PORT": f"{DISPATCHER_PORT}", "EXPORT_REQUESTS_TOTAL": 1},
                "container_ports": [DISPATCHER_PORT],
            }
        ],
        replicas=1,
        namespace=namespace,
        labels=detector_dispatcher_labels
    )
    create_service(
        f"{VIDEO_DETECTOR}-dispatcher-svc",
        target_port=DISPATCHER_PORT,
        port=DISPATCHER_PORT,
        selector=detector_dispatcher_labels,
        namespace=namespace
    )

    classifier_dispatcher_labels = {"pipeline": "video", "component": "dispatcher", "stage": f"{VIDEO_CLASSIFIER}-dispatcher"}
    create_deployment(
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
        replicas=1,
        namespace=namespace,
        labels=classifier_dispatcher_labels
    )
    create_service(
        f"{VIDEO_CLASSIFIER}-dispatcher-svc",
        target_port=DISPATCHER_PORT,
        port=DETECTOR_PORT,
        selector=classifier_dispatcher_labels,
        namespace=namespace
    )

            
def deploy_adapter(classifier_dispatcher_ip):
    adapter_labels = {"pipeline": "video", "component": "adapter"}
    create_deployment(
        ADAPTER_DEPLOY_NAME,
        [
            {
                "name": f"{ADAPTER_DEPLOY_NAME}-container",
                "image": "mehransi/main:pelastic-adapter",
                "request_mem": "1G",
                "request_cpu": "1",
                "limit_mem": "1G",
                "limit_cpu": "1",
                "env_vars": {
                    "FIRST_DECIDE_DELAY_MINUTES": "1",
                    "DECISION_INTERVAL": "5",
                    "K8S_IN_CLUSTER_CLIENT": "true",
                    "K8S_NAMESPACE": namespace, 
                    "BASE_POD_NAMES": json.dumps({0: VIDEO_DETECTOR, 1: VIDEO_CLASSIFIER}),
                    "POD_LABELS": json.dumps({
                        0: DETECTOR_POD_LABELS,
                        1: CLASSIFIER_POD_LABELS
                    }),
                    "POD_PORTS": json.dumps({0: DETECTOR_PORT, 1: CLASSIFIER_PORT}),
                    "CONTAINER_CONFIGS": json.dumps({
                        0: {
                            "name": f"{VIDEO_DETECTOR}-container",
                            "image": "mehransi/main:pelastic-video-detector",
                            "request_mem": "1G",
                            "limit_mem": "1G",
                            "env_vars": {"NEXT_TARGET_ENDPOINT": f"{classifier_dispatcher_ip}:{DISPATCHER_PORT}", "PORT": f"{DETECTOR_PORT}"},
                            "container_ports": [DETECTOR_PORT],
                        },
                        1: {
                            "name": f"{VIDEO_CLASSIFIER}-container",
                            "image": "mehransi/main:pelastic-video-classifier",
                            "request_mem": "1G",
                            "limit_mem": "1G",
                            "env_vars": {"NEXT_TARGET_ENDPOINT": f"{EXPORTER_IP}:{EXPORTER_PORT}", "PORT": f"{CLASSIFIER_PORT}"},
                            "container_ports": [CLASSIFIER_PORT],
                        }
                    })
                    
                },
                "container_ports": [ADAPTER_PORT],
            }
        ],
        replicas=1,
        namespace=namespace,
        labels=adapter_labels
    )
    create_service(
        f"{ADAPTER_DEPLOY_NAME}-svc",
        target_port=ADAPTER_PORT,
        port=ADAPTER_PORT,
        selector=adapter_labels,
        namespace=namespace
    )

def initialize_adapter(adapter_ip, prometheus_endpoint, dispatcher_endpoints):
    response = requests.post(
        f"http://{adapter_ip}:{ADAPTER_PORT}/initialize",
        data=json.dumps({
            "prometheus_endpoint": prometheus_endpoint,
            "dispatcher_endpoints": dispatcher_endpoints,
            "initial_pod_cpus": {0: 4, 1: 4}
        }),
        headers={'Content-type':'application/json', 'Accept':'application/json'}
    )


if __name__ == "__main__":
    deploy_dispatchers()

    wait_till_pod_is_ready(f"{VIDEO_DETECTOR}-dispatcher", namespace)
    wait_till_pod_is_ready(f"{VIDEO_CLASSIFIER}-dispatcher", namespace)
    detector_dispatcher_ip = get_service(f"{VIDEO_DETECTOR}-dispatcher-svc", namespace=namespace)["cluster_ip"]
    classifier_dispatcher_ip = get_service(f"{VIDEO_CLASSIFIER}-dispatcher-svc", namespace=namespace)["cluster_ip"]
    
    deploy_adapter(classifier_dispatcher_ip)
    wait_till_pod_is_ready(ADAPTER_DEPLOY_NAME, namespace)
    adapter_ip = get_service(f"{ADAPTER_DEPLOY_NAME}-svc", namespace=namespace)["cluster_ip"]
    
    prometheus_service = get_service("kube-prom-stack-kube-prome-prometheus", namespace="observability")
    prometheus_endpoint = f"{prometheus_service['cluster_ip']}:{prometheus_service['port']}"
    
    time.sleep(5)
    initialize_adapter(
        adapter_ip,
        prometheus_endpoint,
        {
            0: f"{detector_dispatcher_ip}:{DISPATCHER_PORT}",
            1: f"{classifier_dispatcher_ip}:{DISPATCHER_PORT}"
        }
    )
    
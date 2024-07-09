import requests
import json
import os
import sys
import time

os.system(f"mkdir -p {os.environ['HOME']}/.kube")
os.system(f"microk8s config > {os.environ['HOME']}/.kube/config")
    
    
from kube_resources.deployments import create_deployment
from kube_resources.services import create_service, get_service

from utils import wait_till_pod_is_ready


namespace = "mehran"

pipeline = sys.argv[1]

with open(f"experiment_parameters/{pipeline}.json") as f:
    pipeline_config = json.load(f)
    

ADAPTER_DEPLOY_NAME = "pelastic-adapter"

DISPATCHER_PORT = 8000
ADAPTER_PORT = 8000

EXPORTER_IP = os.environ["NODE_IP"]
EXPORTER_PORT = 8008


def deploy_dispatchers():
   
    for i in range(len(pipeline_config["stages"])):
        dispatcher_labels = {
            "pipeline": pipeline,
            "component": "dispatcher",
            "stage": f"{pipeline_config['stages'][i]['stage_name']}-dispatcher"
        }
        env_vars = {"DISPATCHER_PORT": f"{DISPATCHER_PORT}", "PYTHONUNBUFFERED": "1", "LATENCY_SLO": pipeline_config["SLO"]}
        if i == 0:
            dispatcher_labels["project"] = "pelastic"
            env_vars["EXPORT_REQUESTS_TOTAL"] = 1
        
        create_deployment(
            f"{pipeline_config['stages'][i]['stage_name']}-dispatcher",
            [
                {
                    "name": f"{pipeline_config['stages'][i]['stage_name']}-dispatcher-container",
                    "image": "mehransi/main:pelastic-dispatcher",
                    "request_mem": "1G",
                    "request_cpu": "1",
                    "limit_mem": "1G",
                    "limit_cpu": "1",
                    "env_vars": env_vars,
                    "container_ports": [DISPATCHER_PORT],
                }
            ],
            replicas=1,
            namespace=namespace,
            labels=dispatcher_labels
        )
        create_service(
            f"{pipeline_config['stages'][i]['stage_name']}-dispatcher-svc",
            target_port=DISPATCHER_PORT,
            port=DISPATCHER_PORT,
            selector=dispatcher_labels,
            namespace=namespace
        )

            
def deploy_adapter(next_target_endpoints: dict):
    adapter_labels = {"project": "pelastic", "pipeline": pipeline, "component": "adapter"}
    for i in range(len(pipeline_config["stages"])):
        pipeline_config["stages"][i]["container_configs"]["env_vars"]["NEXT_TARGET_ENDPOINT"] = next_target_endpoints[i]
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
                    "DECISION_INTERVAL": "1",
                    "HORIZONTAL_STABILIZATION": "10",
                    "K8S_IN_CLUSTER_CLIENT": "true",
                    "PYTHONUNBUFFERED": "1",
                    "K8S_NAMESPACE": namespace,
                    "MAX_BATCH_SIZE": 8,
                    "MAX_CPU_CORES": 8,
                    "LATENCY_SLO": pipeline_config["SLO"],
                    "BASE_POD_NAMES": json.dumps({i: pipeline_config["stages"][i]["stage_name"] for i in range(len(pipeline_config["stages"]))}),
                    "LATENCY_MODELS": json.dumps({
                        i: pipeline_config["stages"][i]["latency_model"] for i in range(len(pipeline_config["stages"]))
                    }),
                    "POD_LABELS": json.dumps({
                        i: pipeline_config["stages"][i]["pod_labels"] for i in range(len(pipeline_config["stages"]))
                    }),
                    "POD_PORTS": json.dumps({i: pipeline_config["stages"][i]["port"] for i in range(len(pipeline_config["stages"]))}),
                    "CONTAINER_CONFIGS": json.dumps({
                        i: pipeline_config['stages'][i]['container_configs'] for i in range(len(pipeline_config['stages']))
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
    prometheus_port = 32000
    prometheus_container_name = "pelastic_prometheus"
    # os.system(f"microk8s kubectl apply -f podmonitor.yaml")
    os.system(f"microk8s config > ./prom/kube.config")
    os.system(f"docker stop {prometheus_container_name}")
    time.sleep(1)
    os.system(f"docker rm {prometheus_container_name}")
    time.sleep(1)
    os.system(f"docker run --name {prometheus_container_name} -d -p {prometheus_port}:9090 -v ./prom:/etc/prometheus prom/prometheus")
    
    deploy_dispatchers()
    num_stages = len(pipeline_config["stages"])
    
    for i in range(num_stages):
        wait_till_pod_is_ready(f"{pipeline_config['stages'][i]['stage_name']}-dispatcher", namespace)
    
    dispatcher_endpoints = {}
    for i in range(num_stages):
        dispatcher_endpoints[i] = get_service(f"{pipeline_config['stages'][i]['stage_name']}-dispatcher-svc", namespace=namespace)['cluster_ip'] + f":{DISPATCHER_PORT}"
    
    next_target_endpoints = {}
    for i in range(num_stages - 1):
        next_target_endpoints[i] = dispatcher_endpoints[i+1]
    next_target_endpoints[num_stages - 1] = f"{EXPORTER_IP}:{EXPORTER_PORT}"
    
    
    
    deploy_adapter(next_target_endpoints)
    wait_till_pod_is_ready(ADAPTER_DEPLOY_NAME, namespace)
    adapter_ip = get_service(f"{ADAPTER_DEPLOY_NAME}-svc", namespace=namespace)["cluster_ip"]
    
    # prometheus_service = get_service("kube-prom-stack-kube-prome-prometheus", namespace="observability")
    # prometheus_endpoint = f"{prometheus_service['cluster_ip']}:{prometheus_service['port']}"
    
    prometheus_endpoint = f"{os.environ['NODE_IP']}:{prometheus_port}"
    
    time.sleep(10)
    initialize_adapter(
        adapter_ip,
        prometheus_endpoint,
        dispatcher_endpoints
    )
    
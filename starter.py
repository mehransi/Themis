import asyncio
import base64
import csv
import cv2
import requests
import subprocess
import json
import os
import sys
import time

from barazmoon import BarAzmoon
from datetime import datetime
from multiprocessing import Manager
from threading import Thread, Event

from utils import PrometheusClient



os.system(f"mkdir -p {os.environ['HOME']}/.kube")
os.system(f"microk8s config > {os.environ['HOME']}/.kube/config")
    
    
from kube_resources.deployments import create_deployment
from kube_resources.services import create_service, get_service

from utils import wait_till_pod_is_ready


namespace = "mehran"
GET_METRICS_INTERVAL = 1
FIRST_DECIDE_DELAY_MINUTES = 1

SLO_MULTIPLIER = 1
DROP_MULTIPLIER = 1  # zero means no drop

pipeline = sys.argv[1]
assert pipeline in ["video", "sentiment", "nlp"]
adapter_type = sys.argv[2]
assert adapter_type in ["hv", "ho", "vo", "vomax"], "Adapter type must be one of {hv, ho, vo, vomax}"

with open(f"experiment_parameters/{pipeline}.json") as f:
    pipeline_config = json.load(f)

SLO = pipeline_config["SLO"]
num_stages = len(pipeline_config["stages"])

if adapter_type == "vomax":
    if pipeline == "video":
        initial_replicas = [3, 2]
    elif pipeline == "sentiment":
        initial_replicas = [4, 2]
    elif pipeline == "nlp":
        initial_replicas = [1, 3, 3]
else:
    initial_replicas = [1 for _ in range(num_stages)]

drop_after = SLO * DROP_MULTIPLIER

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
        env_vars = {"DISPATCHER_PORT": f"{DISPATCHER_PORT}", "PYTHONUNBUFFERED": "1", "DROP_AFTER": drop_after}
        # if i == 0:
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
                    "FIRST_DECIDE_DELAY_MINUTES": f"{FIRST_DECIDE_DELAY_MINUTES}",
                    "ADAPTER_TYPE": adapter_type,
                    "DECISION_INTERVAL": "1",
                    "HORIZONTAL_STABILIZATION": pipeline_config["HORIZONTAL_STABILIZATION"],
                    "K8S_IN_CLUSTER_CLIENT": "true",
                    "PYTHONUNBUFFERED": "1",
                    "K8S_NAMESPACE": namespace,
                    "MAX_BATCH_SIZE": pipeline_config["MAX_BATCH_SIZE"],
                    "MAX_CPU_CORES": 8,
                    "LATENCY_SLO": int(SLO * SLO_MULTIPLIER),
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
            "initial_pod_cpus": {i: 1 for i in range(num_stages)},
            "initial_replicas": {i: initial_replicas[i] for i in range(num_stages)}
        }),
        headers={'Content-type':'application/json', 'Accept':'application/json'}
    )


def _get_value(prom_res, divide_by=1, should_round=True):
        for tup in prom_res:
            if tup[1] != "NaN":
                v = float(tup[1]) / divide_by
                if should_round:
                    return round(v, 2)
                return v
        

def query_metrics(prom_endpoint, event: Event):
    async def get_metrics(prom: PrometheusClient):
        loop = asyncio.get_event_loop()
        percentiles = {
            99: None, 98: None, 97: None, 96: None, 95: None, 90: None, 50: None,
        }
        models = {}
        
        for pl in percentiles.keys():
            percentiles[pl] = loop.run_in_executor(None, lambda: prom.get_instant(
                f'histogram_quantile(0.{pl}, sum(rate(pelastic_requests_latency_bucket[{2}s])) by (le))')
            )
        
        
        dispatcher_stage0 = loop.run_in_executor(
            None, lambda: prom.get_instant(f'histogram_quantile(0.99, sum(rate(dispatcher_stage0_latency_bucket[{2}s])) by (le))')
        )
        
        if pipeline == "video":
            models["detector_latency"] = loop.run_in_executor(
                None, lambda: prom.get_instant(f'histogram_quantile(0.99, sum(rate(detector_latency_bucket[{2}s])) by (le))')
            )
            models["classifier_latency"] = loop.run_in_executor(
                None, lambda: prom.get_instant(f'histogram_quantile(0.99, sum(rate(classifier_latency_bucket[{2}s])) by (le))')
            )
        elif pipeline == "sentiment":
            models["audio_latency"] = loop.run_in_executor(
                None, lambda: prom.get_instant(f'histogram_quantile(0.99, sum(rate(audio_latency_bucket[{2}s])) by (le))')
            )
            models["sentiment_latency"] = loop.run_in_executor(
                None, lambda: prom.get_instant(f'histogram_quantile(0.99, sum(rate(sentiment_latency_bucket[{2}s])) by (le))')
            )
        else:
            models["identification_latency"] = loop.run_in_executor(
                None, lambda: prom.get_instant(f'histogram_quantile(0.99, sum(rate(identification_latency_bucket[{2}s])) by (le))')
            )
            models["translation_latency"] = loop.run_in_executor(
                None, lambda: prom.get_instant(f'histogram_quantile(0.99, sum(rate(translation_latency_bucket[{2}s])) by (le))')
            )
            models["summarizer_latency"] = loop.run_in_executor(
                None, lambda: prom.get_instant(f'histogram_quantile(0.99, sum(rate(summarizer_latency_bucket[{2}s])) by (le))')
            )
            models["dispatcher_stage2"] = loop.run_in_executor(
                None, lambda: prom.get_instant(f'histogram_quantile(0.99, sum(rate(dispatcher_stage2_latency_bucket[{2}s])) by (le))')
            )
        
        dispatcher_stage1 = loop.run_in_executor(
            None, lambda: prom.get_instant(f'histogram_quantile(0.99, sum(rate(dispatcher_stage1_latency_bucket[{2}s])) by (le))')
        )

        cost_stage0 = loop.run_in_executor(
            None, lambda: prom.get_instant('sum(last_over_time(pelastic_cost{stage="0"}[2s]))')
        )
        cost_stage1 = loop.run_in_executor(
            None, lambda: prom.get_instant('sum(last_over_time(pelastic_cost{stage="1"}[2s]))')
        )
        cost = loop.run_in_executor(
            None, lambda: prom.get_instant(f"sum(last_over_time(pelastic_cost[{2}s]))")
        )

        rate = loop.run_in_executor(
            None, lambda: prom.get_instant(f'sum(rate(dispatcher_requests_total{{stage="stage-0"}}[{2}s]))')
        )
        
        total_requests = loop.run_in_executor(
            None, lambda: prom.get_instant(f'dispatcher_requests_total{{stage="stage-0"}}')
        )
        
        within_slo = loop.run_in_executor(
            None, lambda: prom.get_instant(f'sum(rate(pelastic_requests_latency_bucket{{le="{SLO / 1000}"}}[2s])) / sum(rate(pelastic_requests_latency_count[2s]))')
        )
        
        drop_rate = loop.run_in_executor(
            None, lambda: prom.get_instant(f"sum(rate(dispatcher_dropped_total[{2}s]))")
        )
        drop_total_stage0 = loop.run_in_executor(
            None, lambda: prom.get_instant(f'dispatcher_dropped_total{{stage="stage-0"}}')
        )
        drop_total_stage1 = loop.run_in_executor(
            None, lambda: prom.get_instant(f'dispatcher_dropped_total{{stage="stage-1"}}')
        )
                
        for pl in percentiles.keys():
            percentiles[pl] = _get_value(await percentiles[pl])
        for m in models.keys():
            models[m] = _get_value(await models[m])
        
        return {
            **percentiles,
            "dispatcher_stage0": _get_value(await dispatcher_stage0),
            **models,
            "dispatcher_stage1": _get_value(await dispatcher_stage1),
            "cost": _get_value(await cost),
            "cost_stage0": _get_value(await cost_stage0),
            "cost_stage1": _get_value(await cost_stage1),
            "rate": _get_value(await rate),
            "total_requests": _get_value(await total_requests),
            "within_slo": _get_value(await within_slo, should_round=False),
            "drop_rate": _get_value(await drop_rate),
            "drop_total_stage0": _get_value(await drop_total_stage0),
            "drop_total_stage1": _get_value(await drop_total_stage1),
            "timestamp": datetime.now().isoformat()
        }
        
    prom = PrometheusClient(prom_endpoint)
    time.sleep(1)
    while True:
        if event.is_set():
            break
        time.sleep(GET_METRICS_INTERVAL)
        metrics = asyncio.run(get_metrics(prom))
        filepath = f"./series-{pipeline}-{adapter_type}-{SLO}-{drop_after}.csv"
        file_exists = os.path.exists(filepath)
        with open(filepath, "a") as f:
            field_names = [
                *list(metrics.keys())
            ]
            writer = csv.DictWriter(f, fieldnames=field_names)
            if not file_exists:
                writer.writeheader()

            writer.writerow(
                metrics
            )


if __name__ == "__main__":
    os.system(f"microk8s kubectl create ns {namespace}")
    prometheus_port = 32000
    prometheus_container_name = "pelastic_prometheus"
    # os.system(f"microk8s kubectl apply -f podmonitor.yaml")
    os.system(f"microk8s config > ./prom/kube.config")
    os.system(
        f"docker run --name {prometheus_container_name} -d --net=host -v ./prom:/etc prom/prometheus --config.file=/etc/prometheus.yml --web.listen-address=0.0.0.0:{prometheus_port}"
    )
    
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
    time.sleep(2)
    
    if pipeline == "video":
        im = cv2.imread(f"./zidane.jpg")
        im = cv2.resize(im, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
        encoded = base64.b64encode(cv2.imencode(".jpg",im)[1].tobytes()).decode("utf-8")
    elif pipeline == "sentiment":
        encoded = base64.b64encode(open('./audio.flac', 'rb').read()).decode("utf-8")
    else:
        encoded = "nor is mister cultar's manner less interesting than his matter"

    counter = Manager().Value("i", value=0)
    class MyLoadTester(BarAzmoon):
        def get_request_data(self):
            global counter
            counter.value += 1
            return counter.value, json.dumps({"data": encoded, "id": counter.value})

        def process_response(self, sent_data_id: str, response: json):
            pass
            # print(str(datetime.now()), sent_data_id, response)

    exporter = subprocess.Popen(["python", f"pipelines/exporter-{pipeline}.py", str(SLO / 1000)])
    time.sleep(FIRST_DECIDE_DELAY_MINUTES * 60)
    event = Event()
    query_task = Thread(target=query_metrics, args=(prometheus_endpoint, event))
    query_task.start()
    
    svc = get_service(f"{pipeline_config['stages'][0]['stage_name']}-dispatcher-svc", "mehran")
    with open("workload.txt") as f:
        wl = f.read()
    if pipeline == "video":
        wl_divider = 4.384
    elif pipeline == "sentiment":
        wl_divider = 8.5625
    else:
        wl_divider = 21
    wl = list(map(lambda x: round(max(1, int(x) / wl_divider)), wl.split()))
    day = 60 * 60 * 24
    workload = wl[15 * day + 84 * 60 + 450: 15 * day + 95 * 60 + 450]
    tester = MyLoadTester(workload=workload, endpoint=F"http://{svc['cluster_ip']}:{svc['port']}/predict", http_method="post")
    count, success = tester.start()
    print("Load tester finished", count, success)
    event.set()
    query_task.join()
    os.system(f"microk8s kubectl delete ns {namespace}")
    os.system(f"docker stop {prometheus_container_name}")
    time.sleep(1)
    os.system(f"docker rm {prometheus_container_name}")
    time.sleep(1)
    os.system(f"kill -9 {exporter.pid}")

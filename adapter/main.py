import asyncio
import json
import logging
import numpy as np
import os
import random
import string
import time
import tensorflow as tf
from aiohttp import ClientSession, web
from datetime import datetime
from tensorflow.keras import saving
from typing import Dict, List
from kube_resources.pods import create_pod, get_pod, update_pod, delete_pod

from optimizer import horizontal_2d, vertical_2d


def load_pipeline_data(d: dict):
    data = {}
    for k, v in json.loads(d).items():
        data[int(k)] = v
        
    return data
    
class Adapter:
    def __init__(self) -> None:
        self.current_state = {}  # {stage: [cores, replicas, batch_size]}
        self.stage_replicas = {}  # {stage: [{name: replica1_pod_name, ip: replica1_pod_ip}, ...]}
        self.latency_models = load_pipeline_data(os.environ["LATENCY_MODELS"])
        self.dispatcher_sessions: Dict[int, ClientSession] = {} 
        self.k8s_namespace = os.environ["K8S_NAMESPACE"]
        self.base_pod_names = load_pipeline_data(os.environ["BASE_POD_NAMES"])
        self.pod_labels = load_pipeline_data(os.environ["POD_LABELS"])
        self.pod_ports = load_pipeline_data(os.environ["POD_PORTS"])
        self.container_configs = load_pipeline_data(os.environ["CONTAINER_CONFIGS"])
        for i in range(len(self.container_configs)):
            self.stage_replicas[i] = []
        self.max_batch_size = int(os.environ["MAX_BATCH_SIZE"])
        self.max_cores = int(os.environ["MAX_CPU_CORES"])
        self.latency_slo = int(os.environ["LATENCY_SLO"])
        self.lstm_model = None
        self.prometheus_session = None
        self.logger = logging.getLogger()
    
    async def initialize(self, data: dict):
        self.lstm_model = saving.load_model(os.environ["LSTM_MODEL"])
        self.prometheus_session = ClientSession(
            base_url=f"http://{data['prometheus_endpoint']}"
        )
        tasks = []
        for idx, endpoint in data["dispatcher_endpoints"].items():
            self.dispatcher_sessions[int(idx)] = ClientSession(base_url=f"http://{endpoint}")
            c = int(data["initial_pod_cpus"][idx])
            tasks.append(asyncio.create_task(self.create_pod(int(idx), c)))
            self.current_state[int(idx)] = [c, 1, 4]
        
        await asyncio.gather(*tasks)
        
        tasks = []
        for i in range(len(self.dispatcher_sessions)):
            tasks.append(asyncio.create_task(self.initialize_dispatcher(i)))
        
        await asyncio.gather(*tasks)
            
    async def initialize_dispatcher(self, stage_idx):
        async with self.dispatcher_sessions[stage_idx].post("/initialize", json={
            "dispatcher_name": f"stage-{stage_idx}",
            "backends_port": self.pod_ports[stage_idx],
            "batch_size": 4,
            "backends": self.stage_replicas[stage_idx]
        }) as response:
            response = await response.json()

    async def adapt(self):
        async with self.prometheus_session.post(
            f"/api/v1/query",
            params={
                "query": f"sum(rate(dispatcher_requests_total[2s]))",
            }
        ) as response:
            response = await response.json()
            try:
                current_rps = int(response["data"]["result"]["value"][1])
            except TypeError:
                current_rps = int(response["data"]["result"][0]["value"][1])
            
        now = datetime.now().timestamp()
        async with self.prometheus_session.post(
            f"/api/v1/query_range",
            params={
                "query": "sum(dispatcher_requests_total)",
                "start": now - 300,
                "end": now,
                "step": 1
            }
        ) as response:
            response = await response.json()
            try:
                history_rps = response["data"]["result"].get("values")
            except AttributeError:
                try:
                    history_rps = response["data"]["result"][0].get("values")
                except (KeyError, IndexError, AttributeError):
                    history_rps = response["data"]["result"]
        history_rps = list(map(lambda x: int(x[1]), history_rps))
        history_rps = [history_rps[i] - history_rps[i-1] for i in range(1, len(history_rps))]
        inp = []
        for i in range(0, len(history_rps), 10):
            inp.append(max(history_rps[i:i+10]))
        history_rps = inp
        if len(history_rps) < 30:
            next_10s_rps = current_rps
            # FIXME
        else:
            next_10s_rps = self.predict(history_rps)
            self.logger.info(f"adapter LSTM next_10s_rps without error considered: {next_10s_rps}")
        
        new_horizontal_config = horizontal_2d(self.max_batch_size, self.latency_slo, self.latency_models, current_rps)
        future_horizontal_config = horizontal_2d(self.max_batch_size, self.latency_slo, self.latency_models, next_10s_rps)
        should_apply_horizontal = True
        for i in range(len(new_horizontal_config)):
            if new_horizontal_config[i][0] < future_horizontal_config[i][0]:
                should_apply_horizontal = False
                break
        
        
        loop = asyncio.get_event_loop()
        
        if should_apply_horizontal is False:
            tasks = []
            new_vertical_config = vertical_2d(self.max_batch_size, self.max_cores, self.latency_slo, self.latency_models, self.current_state, current_rps)
            new_state = {}
            for i in range(len(new_vertical_config)):
                new_state[i] = [new_vertical_config[i][0], self.current_state[i][1], new_vertical_config[i][1]]
                if new_vertical_config[i][0] != self.current_state[i][0]:
                    for r in self.stage_replicas[i]:
                        tasks.append(
                            self.update_pod(r, i, new_vertical_config[i][0])
                        )
            await asyncio.gather(*tasks)
        else:
            tasks = []
            new_state = {}
            for i in range(len(new_horizontal_config)):
                new_state[i] = [1, new_horizontal_config[i][0], new_horizontal_config[i][1]]
                
                if new_horizontal_config[i][0] >= self.current_state[i][1]:
                    for r in self.stage_replicas[i]:
                        tasks.append(
                            asyncio.create_task(self.update_pod(r, i, 1))
                        )
                    for _ in range(new_horizontal_config[i][0] - self.current_state[i][1]):
                        tasks.append(
                            asyncio.create_task(self.create_pod(i, 1))
                        )
                else:
                    extra_pods_count = self.current_state[i][1] - new_horizontal_config[i][0]
                    extra_pods = self.stage_replicas[i][:extra_pods_count]
                    self.stage_replicas[i] = self.stage_replicas[i][extra_pods_count:]
                    for r in extra_pods:
                        tasks.append(
                            loop.run_in_executor(
                                None,
                                lambda: delete_pod(r["name"], namespace=self.k8s_namespace)
                            )
                        )                    
            await asyncio.gather(*tasks)
        
        
        update_dispatchers_tasks = []
        for i in range(len(self.current_state)):
            if should_apply_horizontal:
                update_dispatchers_tasks.append(self.reset_backends(i, self.stage_replicas[i]))
            _, _, prev_stage_batch_size = self.current_state[i]
            _, _, new_stage_batch_size = new_state[i]
            if new_stage_batch_size != prev_stage_batch_size:
                update_dispatchers_tasks.append(self.update_batch(i, new_stage_batch_size))
        
        self.current_state = new_state
        await asyncio.gather(*update_dispatchers_tasks)
                
    async def create_pod(self, stage_idx, cpu: int):
        new_pod_name = f"{self.base_pod_names[stage_idx]}-{''.join(random.choices(string.ascii_lowercase + string.digits, k=5))}"
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: create_pod(
                new_pod_name,
                [{
                    **self.container_configs[stage_idx],
                    "request_cpu": cpu,
                    "limit_cpu": cpu
                }],
                namespace=self.k8s_namespace,
                labels=self.pod_labels[stage_idx]
            )
        )
        while True:
            await asyncio.sleep(0.05)
            pod = await loop.run_in_executor(None, lambda: get_pod(new_pod_name, namespace=self.k8s_namespace))
            if pod["pod_ip"] and pod["pod_ip"].lower() != "none":
                break
            
        while True:
            await asyncio.sleep(0.1)
            try:
                await self.initialize_pod(f"{pod['pod_ip']}:{self.pod_ports[stage_idx]}", cpu)
                break
            except:
                pass
        self.logger.info(f"New pod for stage {stage_idx} created. pod name: {pod['name']}, cpu: {cpu}")
        self.stage_replicas[stage_idx].append({"name": pod["name"], "ip": pod["pod_ip"]})
        return pod
        
    
    async def initialize_pod(self, pod_endpoint, cpu):
        async with ClientSession() as session:
            async with session.post(
                f"http://{pod_endpoint}/initialize", json={"threads": cpu}
            ) as response:
                return await response.json()

    async def update_pod(self, pod_info: dict, stage_idx, new_cpu):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: update_pod(
                pod_info["name"],
                [{
                    **self.container_configs[stage_idx],
                    "request_cpu": new_cpu,
                    "limit_cpu": new_cpu
                }],
                namespace=self.k8s_namespace
            )
        )
        await self.update_threading(f"{pod_info['ip']}:{self.pod_ports[stage_idx]}", new_cpu)
        
    async def update_threading(self, pod_endpoint, threads: int):
        async with ClientSession() as session:
            async with session.post(
                f"http://{pod_endpoint}/update-threads", json={"threads": threads}
            ) as response:
                return await response.json()
            
     
    async def update_batch(self, stage_idx, new_batch: int):
        async with self.dispatcher_sessions[stage_idx].post(
            "/reset-batch", json={"batch_size": new_batch}
        ) as response:
            response = await response.json()
    
    async def reset_backends(self, stage_idx, backends: List[dict]):
        async with self.dispatcher_sessions[stage_idx].post(
            "/reset-backends", json={"backends": backends}
        ) as response:
            response = await response.json()
                
        
    def predict(self, history_10m_rps):
        t = time.perf_counter()
        
        history_5m = tf.convert_to_tensor(np.array(history_10m_rps).reshape((-1, 30, 1)), dtype=tf.float32)
        next_10s = self.lstm_model.predict(history_5m)
        self.logger.info(f"LSTM predict took {time.perf_counter() - t} seconds")
        return int(next_10s)


adapter = Adapter()


async def initialize(request):
    data = await request.json()
    if adapter.prometheus_session is None:
        await adapter.initialize(data)
        return web.json_response({"success": True})
    return web.json_response({"success": False, "message": "Already initialized."})



async def export_cost(request):
    content = "# HELP pelastic_cost Number of cpu cores * replicas.\n"
    content += "# TYPE pelastic_cost gauge\n"
    for stage_idx in range(len(adapter.current_state)):
        cpu, replicas, _ = adapter.current_state[stage_idx]
        content += f'pelastic_cost{{stage="{stage_idx}"}} {cpu*replicas}\n'
    return web.Response(body=content)


async def decide(request):
    return web.json_response(await adapter.adapt())


app = web.Application()
app.add_routes(
    [
        web.post("/initialize", initialize),
        web.get("/metrics", export_cost),
        web.post("/decide", decide)
    ]
)
if __name__ == '__main__':
    web.run_app(app, host="0.0.0.0", port=8000, access_log=None)

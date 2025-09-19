import asyncio
import json
import logging
import math
import numpy as np
import os
import random
import string
import time
import tensorflow as tf
from aiohttp import ClientSession, web
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from tensorflow.keras import saving
from typing import Dict, List
from kube_resources.pods import create_pod, get_pod, update_pod, delete_pod

from optimizer import horizontal_2d, vertical_2d, latency as get_latency, get_throughput, core_to_G_RAM


def load_pipeline_data(d: dict):
    data = {}
    for k, v in json.loads(d).items():
        data[int(k)] = v
        
    return data
    
class Adapter:
    def __init__(self) -> None:
        self.current_state = {}  # {stage: [cores, replicas, batch_size]}
        self.stage_replicas = {}  # {stage: [{name: replica1_pod_name, ip: replica1_pod_ip}, ...]}
        self.horizontal_stabilization = int(os.getenv("HORIZONTAL_STABILIZATION", 10))
        self.binary_threshold = float(os.getenv("BINARY_THRESHOLD", 0.5))
        self.horizontal_stabilization_counter = 0
        self.latency_models = load_pipeline_data(os.environ["LATENCY_MODELS"])
        self.dispatcher_sessions: Dict[int, ClientSession] = {}
        self.k8s_namespace = os.environ["K8S_NAMESPACE"]
        self.base_pod_names = load_pipeline_data(os.environ["BASE_POD_NAMES"])
        self.pod_labels = load_pipeline_data(os.environ["POD_LABELS"])
        self.pod_ports = load_pipeline_data(os.environ["POD_PORTS"])
        self.container_configs = load_pipeline_data(os.environ["CONTAINER_CONFIGS"])
        self.memory_per_stage_replica = {}
        for i in range(len(self.container_configs)):
            self.stage_replicas[i] = []
            mem = self.container_configs[i]["request_mem"]
            self.memory_per_stage_replica[i] = int(mem[:mem.index("G")]) # Fixme: Other units
        self.max_batch_sizes = load_pipeline_data(os.environ["MAX_BATCH_SIZE"])
        self.max_cores = load_pipeline_data(os.environ["MAX_CPU_CORES"])
        self.latency_slo = int(os.environ["LATENCY_SLO"])
        self.lstm_model = None
        self.prometheus_session = None
        self.__thread_executor = ThreadPoolExecutor(max_workers=5 * len(self.latency_models))
        self.logger = logging.getLogger()
        logging.basicConfig(level = logging.INFO)
    
    async def initialize(self, data: dict):
        self.lstm_model = saving.load_model(os.environ["LSTM_MODEL"])
        self.prometheus_session = ClientSession(
            base_url=f"http://{data['prometheus_endpoint']}"
        )
        tasks = []
        self.initial_cpus = {}
        self.initial_batches = {}
        self.batch_timeout = {}
        self.ro_list = {}
        self.tp_list = {}
        for idx, endpoint in data["dispatcher_endpoints"].items():
            self.dispatcher_sessions[int(idx)] = ClientSession(base_url=f"http://{endpoint}")
            c = int(data["initial_pod_cpus"][idx])
            self.initial_cpus[idx] = c
            r = int(data["initial_replicas"][idx])
            self.batch_timeout[int(idx)] = int(data["batch_timeout"][idx])
            for _ in range(r):
                tasks.append(asyncio.create_task(self.create_pod(int(idx), c)))
            b = int(data["initial_batches"][idx])
            self.initial_batches[int(idx)] = b
            self.tp_list[int(idx)] = int(1000 * b / get_latency(c, b, *self.latency_models[int(idx)]))
            self.ro_list[int(idx)] = data["inferline_base_arrival"] / (r * self.tp_list[int(idx)])
            
            self.current_state[int(idx)] = [c, r, b]
        
        self.logger.info(f"datetime={str(datetime.now())}, inferline_base_arrival={data['inferline_base_arrival']}, ro_list={json.dumps(self.ro_list)}")
        await asyncio.gather(*tasks)

        tasks = []
        
        for i in range(len(self.dispatcher_sessions)):
            tasks.append(asyncio.create_task(self.initialize_dispatcher(i, self.initial_batches[i], self.batch_timeout[i])))
        
        await asyncio.gather(*tasks)
            
    async def initialize_dispatcher(self, stage_idx, batch_size, batch_timeout):
        async with self.dispatcher_sessions[stage_idx].post("/initialize", json={
            "dispatcher_name": f"stage-{stage_idx}",
            "backends_port": self.pod_ports[stage_idx],
            "batch_size": batch_size,
            "batch_timeout": batch_timeout,
            "backends": self.stage_replicas[stage_idx]
        }) as response:
            response = await response.json()

    async def adapt(self):
        starting_time = time.perf_counter()
        current_rps = await self.get_current_rps()
        current_throughput = get_throughput(self.current_state, self.latency_models)
        
        is_wl_increasing = await self.is_wl_increasing_next_10(current_throughput)
        should_apply_horizontal = not is_wl_increasing and (current_rps < current_throughput)
        
        no_room_for_vertical = False
        for i in range(len(self.current_state)):
            if self.current_state[i][0] == self.max_cores[i]:
                no_room_for_vertical = True
                break
        
        should_apply_horizontal = should_apply_horizontal or no_room_for_vertical
            
        self.logger.info(
            f"datetime={str(datetime.now())}, {current_rps=}, {current_throughput=}, stabilization_counter={self.horizontal_stabilization_counter}, {should_apply_horizontal=}, {is_wl_increasing=}", 
        )
        if should_apply_horizontal:
            new_horizontal_config = horizontal_2d(
                self.max_batch_sizes,
                [1] * len(self.current_state), 
                self.latency_slo, 
                self.latency_models,
                self.memory_per_stage_replica,
                current_rps
            )
            scaling_in = True
            for i in range(len(new_horizontal_config)):
                if new_horizontal_config[i][0] > self.current_state[i][1]:
                    scaling_in = False
            if scaling_in and self.horizontal_stabilization_counter < self.horizontal_stabilization:
                self.horizontal_stabilization_counter += 1
                return
            
        self.horizontal_stabilization_counter = 0
        
        loop = asyncio.get_event_loop()
        pod_status_check_tasks = []
        futures = []
        if should_apply_horizontal is False:
            update_tasks = []
            create_tasks = []
            
            t_dp = time.perf_counter()
            new_vertical_config, more_instances = vertical_2d(
                self.max_batch_sizes, 
                self.max_cores, 
                self.latency_slo, self.latency_models, self.current_state, current_rps
            )
            self.logger.info(
                f"datetime={str(datetime.now())}, vertical_2d_took={time.perf_counter() - t_dp:.4f}, {current_rps=}"
            )
            before_vertical_apply_time = time.perf_counter()
            new_state = {}
            for i in range(len(new_vertical_config)):
                if os.getenv("VERTICAL_SCALE_DOWN", "false").lower() == "true":
                    new_cpu = new_vertical_config[i][0]
                    must_update = new_cpu != self.current_state[i][0]
                else:
                    new_cpu = max(new_vertical_config[i][0], self.current_state[i][0])
                    must_update = new_vertical_config[i][0] > self.current_state[i][0]
                new_state[i] = [new_cpu, self.current_state[i][1] + more_instances[i], new_vertical_config[i][1]]
                if must_update:
                    for r in self.stage_replicas[i]:
                        update_tasks.append(
                            asyncio.create_task(self.update_pod(r, i, new_vertical_config[i][0]))
                        )
                for _ in range(more_instances[i]):
                    create_tasks.append(
                        asyncio.create_task(self.create_pod(i, new_vertical_config[i][0]))
                    )
            t_ = time.perf_counter()
            pod_status_check_tasks = await asyncio.gather(*update_tasks)
            update_in_vertical_took = time.perf_counter() - t_
            t_ = time.perf_counter()
            await asyncio.gather(*create_tasks)
            self.logger.info(
                f"datetime={str(datetime.now())}, vertical_config={json.dumps(new_vertical_config)}, more_instances={more_instances} vertical_crud_took={time.perf_counter() - before_vertical_apply_time:.3f}, updating_pods_took={update_in_vertical_took:.3f}, creating_pods_took={time.perf_counter() - t_:.3f}"
            )
        else:
            t_dp = time.perf_counter()            
            self.logger.info(
                f"datetime={str(datetime.now())}, horizontal_2d_took={time.perf_counter() - t_dp:.4f}, {current_rps=}"
            )
            tasks = []
            new_state = {}
            before_horizontal_apply_time = time.perf_counter()
            for i in range(len(new_horizontal_config)):
                if self.current_state[i][0] > new_horizontal_config[i][1]:
                    # Add one more instance for stablity
                    new_horizontal_config[i][0] += 1
                    
                new_state[i] = [new_horizontal_config[i][1], new_horizontal_config[i][0], new_horizontal_config[i][2]]
                if new_state[i] == self.current_state[i]:
                    continue
                
                if new_horizontal_config[i][0] >= self.current_state[i][1]:
                    for r in self.stage_replicas[i]:
                        futures.append(self.update_pod(r, i, new_horizontal_config[i][1], check_status=False))
                    for _ in range(new_horizontal_config[i][0] - self.current_state[i][1]):
                        tasks.append(
                            asyncio.create_task(self.create_pod(i, new_horizontal_config[i][1]))
                        )
                else:
                    extra_pods_count = self.current_state[i][1] - new_horizontal_config[i][0]
                    extra_pods = self.stage_replicas[i][:extra_pods_count]
                    self.stage_replicas[i] = self.stage_replicas[i][extra_pods_count:]
                    for r in self.stage_replicas[i]:
                        tasks.append(asyncio.create_task(self.update_pod(r, i, new_horizontal_config[i][1], check_status=False)))
                    self.logger.info(f"datetime={str(datetime.now())}, pods_to_delete_stage{i}={json.dumps(extra_pods)}")
                    if extra_pods:
                        await self.reset_backends(i, self.stage_replicas[i])
                        for r in extra_pods:
                            delete_response = await loop.run_in_executor(
                                self.__thread_executor,
                                lambda: delete_pod(r["name"], namespace=self.k8s_namespace)
                            )
                            self.logger.info(f"datetime={str(datetime.now())}, deleted_replica={r['name']}")
                        
            await asyncio.gather(*tasks)
            self.logger.info(f"datetime={str(datetime.now())}, horizontal_crud_took={time.perf_counter() - before_horizontal_apply_time:.3f}")
        
        
        update_dispatchers_tasks = []
        before_updating_dispatchers_time = time.perf_counter()
        for i in range(len(self.current_state)):
            _, prev_replicas, prev_batch_size = self.current_state[i]
            _, new_replicas, new_batch_size = new_state[i]
            if prev_replicas < new_replicas:  # for deleted replicas, we already resetted the backends
                update_dispatchers_tasks.append(self.reset_backends(i, self.stage_replicas[i]))
            if new_batch_size != prev_batch_size:
                update_dispatchers_tasks.append(self.update_batch(i, new_batch_size))
        
        self.logger.info(f"datetime={str(datetime.now())}, Prev state={json.dumps(self.current_state)}, New state={json.dumps(new_state)}, stage_replicas={json.dumps(self.stage_replicas)}")
        self.current_state = new_state
        await asyncio.gather(*update_dispatchers_tasks)
        await asyncio.gather(*[asyncio.create_task(f) for f in futures])
        await asyncio.gather(*[asyncio.create_task(f) for f in pod_status_check_tasks])
        self.logger.info(f"datetime={str(datetime.now())}, reconfiguration_took {time.perf_counter() - starting_time:.3f}, update_dispatchers_took={time.perf_counter() - before_updating_dispatchers_time:.3f}")
        self.logger.info(f"")
                
    
    async def adapt_ho(self):
        starting_time = time.perf_counter()
        current_rps = await self.get_current_rps()
        horizontal_config = horizontal_2d(
            self.max_batch_sizes,
            [1] * len(self.current_state),
            self.latency_slo,
            self.latency_models, 
            self.memory_per_stage_replica,
            current_rps
        )
        is_scaling_in = True
        for i in range(len(self.current_state)):
            if self.current_state[i][1] < horizontal_config[i][0]:
                is_scaling_in = False
                break
        if is_scaling_in:
            if self.horizontal_stabilization_counter < self.horizontal_stabilization:
                    self.horizontal_stabilization_counter += 1
                    return
        new_state = {}
        self.horizontal_stabilization_counter = 0
        tasks = []
        loop = asyncio.get_event_loop()
        for i in range(len(horizontal_config)):
            new_state[i] = [1, horizontal_config[i][0], horizontal_config[i][2]]
            self.logger.info(f"datetime={str(datetime.now())}, Prev state {i}={json.dumps(self.current_state[i])} | New state {i}={json.dumps(new_state[i])}")
            if new_state[i] == self.current_state[i]:
                continue
            
            if horizontal_config[i][0] > self.current_state[i][1]:
                for _ in range(horizontal_config[i][0] - self.current_state[i][1]):
                    tasks.append(
                        asyncio.create_task(self.create_pod(i, 1))
                    )
            else:
                extra_pods_count = self.current_state[i][1] - horizontal_config[i][0]
                extra_pods = self.stage_replicas[i][:extra_pods_count]
                self.stage_replicas[i] = self.stage_replicas[i][extra_pods_count:]
                self.logger.info(f"datetime={str(datetime.now())}, pods_to_delete_stage{i}={json.dumps(extra_pods)}")
                self.logger.info(f"datetime={str(datetime.now())}, stage_replicas={json.dumps(self.stage_replicas)}, current_state={json.dumps(self.current_state)}")
                if extra_pods:
                    await self.reset_backends(i, self.stage_replicas[i])
                    for r in extra_pods:
                        delete_response = await loop.run_in_executor(
                            self.__thread_executor,
                            lambda: delete_pod(r["name"], namespace=self.k8s_namespace)
                        )
                        self.logger.info(f"datetime={str(datetime.now())}, deleted_replica={r['name']}")
        await asyncio.gather(*tasks)
        
        update_dispatchers_tasks = []
        before_updating_dispatchers_time = time.perf_counter()
        for i in range(len(self.current_state)):
            _, prev_replicas, prev_batch_size = self.current_state[i]
            _, new_replicas, new_batch_size = new_state[i]
            if prev_replicas < new_replicas:  # for deleted replicas, we already resetted the backends
                update_dispatchers_tasks.append(self.reset_backends(i, self.stage_replicas[i]))
            if new_batch_size != prev_batch_size:
                update_dispatchers_tasks.append(self.update_batch(i, new_batch_size))
        
        self.logger.info(f"datetime={str(datetime.now())}, Prev state={json.dumps(self.current_state)} | New state={json.dumps(new_state)}")
        self.current_state = new_state
        await asyncio.gather(*update_dispatchers_tasks)
        self.logger.info(f"datetime={str(datetime.now())}, reconfiguration_took {time.perf_counter() - starting_time:.3f}, update_dispatchers_took={time.perf_counter() - before_updating_dispatchers_time:.3f}")
    
    
    async def adapt_vo(self):
        starting_time = time.perf_counter()
        current_rps = await self.get_current_rps()
        new_vertical_config, more_instances = vertical_2d(self.max_batch_sizes, self.max_cores, self.latency_slo, self.latency_models, self.current_state, current_rps)
        
        is_scaling_down = True
        for i in range(len(self.current_state)):
            if self.current_state[i][0] < new_vertical_config[i][0]:
                is_scaling_down = False
                break
        if is_scaling_down:
            if self.horizontal_stabilization_counter < self.horizontal_stabilization:
                self.horizontal_stabilization_counter += 1
                return
        
        self.horizontal_stabilization_counter = 0
        new_state = {}
        update_tasks = []
        
        for i in range(len(new_vertical_config)):
            new_state[i] = [new_vertical_config[i][0], self.current_state[i][1], new_vertical_config[i][1]]
            if new_vertical_config[i][0] != self.current_state[i][0]:
                for r in self.stage_replicas[i]:
                    update_tasks.append(
                        asyncio.create_task(self.update_pod(r, i, new_vertical_config[i][0]))
                    )
        
        await asyncio.gather(*update_tasks)
        update_tasks = []
        for i in range(len(self.current_state)):
            _, _, prev_batch_size = self.current_state[i]
            _, _, new_batch_size = new_state[i]
            if new_batch_size != prev_batch_size:
                update_tasks.append(self.update_batch(i, new_batch_size))
                
        await asyncio.gather(*update_tasks)
        self.logger.info(f"datetime={str(datetime.now())}, Prev state={json.dumps(self.current_state)} | New state={json.dumps(new_state)}, reconfiguration_took {time.perf_counter() - starting_time:.3f}")
        self.current_state = new_state
    
    async def adapt_il(self):
        starting_time = time.perf_counter()
        current_rps = await self.get_rps_history(30)
        current_rps = max(current_rps)

        replicas = []
        for i in range(len(self.current_state)):
            stage_replicas = max(1, math.ceil(current_rps / (self.tp_list[i] * self.ro_list[i])))
            if stage_replicas < self.current_state[i][1]:
                stage_replicas = max(1, math.ceil(current_rps / (self.tp_list[i] * min(self.ro_list.values()))))
                if stage_replicas > self.current_state[i][1]:
                    stage_replicas = self.current_state[i][1]
            replicas.append(stage_replicas)
            
        is_scaling_in = True
        for i in range(len(self.current_state)):
            if self.current_state[i][1] < replicas[i]:
                is_scaling_in = False
                break
        self.logger.info(
            f"datetime={str(datetime.now())}, {current_rps=},  stabilization_counter={self.horizontal_stabilization_counter}, {is_scaling_in=}, Replicas={json.dumps(replicas)}"
        )
        if is_scaling_in:
            if self.horizontal_stabilization_counter < self.horizontal_stabilization:
                self.horizontal_stabilization_counter += 1
                return
        new_state = {}
        self.horizontal_stabilization_counter = 0
        tasks = []
        loop = asyncio.get_event_loop()
        for i in range(len(replicas)):
            new_state[i] = [self.current_state[i][0], replicas[i], self.current_state[i][2]]
            self.logger.info(f"datetime={str(datetime.now())}, Prev state {i}={json.dumps(self.current_state[i])} | New state {i}={json.dumps(new_state[i])}")
            if new_state[i] == self.current_state[i]:
                continue
            
            if replicas[i] > self.current_state[i][1]:
                for _ in range(replicas[i] - self.current_state[i][1]):
                    tasks.append(
                        asyncio.create_task(self.create_pod(i, self.current_state[i][0]))
                    )
            else:
                extra_pods_count = self.current_state[i][1] - replicas[i]
                extra_pods = self.stage_replicas[i][:extra_pods_count]
                self.stage_replicas[i] = self.stage_replicas[i][extra_pods_count:]
                self.logger.info(f"datetime={str(datetime.now())}, pods_to_delete_stage{i}={json.dumps(extra_pods)}")
                self.logger.info(f"datetime={str(datetime.now())}, stage_replicas={json.dumps(self.stage_replicas)}, current_state={json.dumps(self.current_state)}")
                if extra_pods:
                    await self.reset_backends(i, self.stage_replicas[i])
                    for r in extra_pods:
                        delete_response = await loop.run_in_executor(
                            self.__thread_executor,
                            lambda: delete_pod(r["name"], namespace=self.k8s_namespace)
                        )
                        self.logger.info(f"datetime={str(datetime.now())}, deleted_replica={r['name']}")
        await asyncio.gather(*tasks)
        
        update_dispatchers_tasks = []
        before_updating_dispatchers_time = time.perf_counter()
        for i in range(len(self.current_state)):
            _, prev_replicas, _ = self.current_state[i]
            _, new_replicas, _ = new_state[i]
            if prev_replicas < new_replicas:  # for deleted replicas, we already resetted the backends
                update_dispatchers_tasks.append(self.reset_backends(i, self.stage_replicas[i]))
        
        self.logger.info(f"datetime={str(datetime.now())}, Inferline, {current_rps=}, Prev state={json.dumps(self.current_state)} | New state={json.dumps(new_state)}")
        self.current_state = new_state
        await asyncio.gather(*update_dispatchers_tasks)
        self.logger.info(f"datetime={str(datetime.now())}, reconfiguration_took {time.perf_counter() - starting_time:.3f}, update_dispatchers_took={time.perf_counter() - before_updating_dispatchers_time:.3f}")
        
    
    async def get_current_rps(self):
        t = time.perf_counter()
        
        current_rps = await self.get_rps_history(5)
        current_rps = max(current_rps)
        
        return math.ceil(current_rps)
        # async with self.prometheus_session.post(
        #     f"/api/v1/query",
        #     params={
        #         "query": 'rate(dispatcher_requests_total{stage="stage-0"}[2s])',
        #     }
        # ) as response:
        #     response = await response.json()
        #     self.logger.info(f"datetime={str(datetime.now())}, get_current_rps_took: {time.perf_counter() - t:.3f}")
        #     rps = float(response["data"]["result"][0]["value"][1])
        #     return math.ceil(rps)
    
    async def is_wl_increasing_next_10(self, target: int):
        t = time.perf_counter()
        history_rps = await self.get_rps_history(60)
        get_history_rps_took = round(time.perf_counter() - t, 3)
        t = time.perf_counter()
        inp = []
        for i in range(0, len(history_rps), 10):
            inp.append(max(history_rps[i:i+10]))
        history_rps = inp
        if len(history_rps) < 6:
            history_rps = history_rps + (6 - len(history_rps)) * [history_rps[-1]]

        pred = self.predict(history_rps, target)
        self.logger.info(f"datetime={str(datetime.now())}, {get_history_rps_took=}, LSTM_prediction_took: {time.perf_counter() - t:.3f}")
        return pred
    
    async def get_rps_history(self, seconds):
        now = datetime.now().timestamp()
        async with self.prometheus_session.post(
            f"/api/v1/query_range",
            params={
                "query": 'dispatcher_requests_total{stage="stage-0"}',
                "start": now - seconds,
                "end": now,
                "step": 1
            }
        ) as response:
            response = await response.json()
            history_rps = response["data"]["result"][0].get("values")
       
        history_rps = list(map(lambda x: int(x[1]), history_rps))
        history_rps = [history_rps[i] - history_rps[i-1] for i in range(1, len(history_rps))]
        return history_rps
        
        
    async def create_pod(self, stage_idx, cpu: int):
        new_pod_name = f"{self.base_pod_names[stage_idx]}-{''.join(random.choices(string.ascii_lowercase + string.digits, k=5))}"
        loop = asyncio.get_event_loop()
        t = time.perf_counter()
        await loop.run_in_executor(
            self.__thread_executor,
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
        pod_creation_time = time.perf_counter() - t
        t = time.perf_counter()
        while True:
            await asyncio.sleep(0.05)
            pod = await loop.run_in_executor(self.__thread_executor, lambda: get_pod(new_pod_name, namespace=self.k8s_namespace))
            if pod["pod_ip"] and pod["pod_ip"].lower() != "none":
                break
        pod_wait_for_ip_time = time.perf_counter() - t
        t = time.perf_counter()
        while True:
            await asyncio.sleep(0.05)
            try:
                await self.initialize_pod(f"{pod['pod_ip']}:{self.pod_ports[stage_idx]}", cpu)
                break
            except:
                pass
        pod_initialization_time = time.perf_counter() - t
        self.logger.info(f"datetime={str(datetime.now())}, New pod for stage {stage_idx} created. pod name: {pod['name']}, cpu: {cpu}, creation={pod_creation_time:.3f}, ip={pod_wait_for_ip_time:.3f}, init={pod_initialization_time:.3f}")
        self.stage_replicas[stage_idx].append({"name": pod["name"], "ip": pod["pod_ip"]})
        return pod
        
    
    async def initialize_pod(self, pod_endpoint, cpu):
        async with ClientSession() as session:
            async with session.post(
                f"http://{pod_endpoint}/initialize", json={"threads": cpu}
            ) as response:
                return await response.json()

    async def update_pod(self, pod_info: dict, stage_idx, new_cpu, check_status=True):
        loop = asyncio.get_event_loop()
        current_cpu = self.current_state[stage_idx][0]
        if new_cpu == 1:
            await self.update_threading(f"{pod_info['ip']}:{self.pod_ports[stage_idx]}", new_cpu)
        await loop.run_in_executor(
            self.__thread_executor,
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
        if new_cpu > 1:
            await self.update_threading(f"{pod_info['ip']}:{self.pod_ports[stage_idx]}", new_cpu)
        
        if check_status:
            return self.react_to_update_status(pod_info, stage_idx, current_cpu, new_cpu)
    
    async def react_to_update_status(self, pod_info, stage_idx, current_cpu, new_cpu):
        loop = asyncio.get_event_loop()
        status = await self.check_update_status(pod_info["name"])
        
        if status == "PodResizePending":
            self.logger.info(f"datetime={str(datetime.now())}, pod {pod_info['name']} update deferred")
            await self.update_threading(f"{pod_info['ip']}:{self.pod_ports[stage_idx]}", current_cpu)
            await self.create_pod(stage_idx, new_cpu)
            await loop.run_in_executor(
                self.__thread_executor,
                lambda: delete_pod(pod_info['name'], namespace=self.k8s_namespace)
            )
            self.stage_replicas[stage_idx] = list(filter(lambda x: x["name"] != pod_info["name"], self.stage_replicas[stage_idx]))
            await self.reset_backends(stage_idx, self.stage_replicas[stage_idx])
        
        
    
    async def check_update_status(self, pod_name, rep=0):
        response = get_pod(pod_name, namespace=self.k8s_namespace)
        status = response["conditions"][0]["type"]
        if status not in ["PodResizeInProgress", "PodResizePending"]:
            await asyncio.sleep(0.02)
            if rep < 5:
                return await self.check_update_status(pod_name, rep+1)
        self.logger.info(f"datetime={str(datetime.now())}, pod {pod_name} update status {status} rep={rep}")
        return status
            

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
                
        
    def predict(self, history_rps, target):
        t = time.perf_counter()
        
        history = tf.convert_to_tensor(np.array(history_rps).reshape((-1, 6, 1)), dtype=tf.float32)
        preds = self.lstm_model.predict([history, tf.convert_to_tensor([target])], verbose=0)
        y_pred_classes = (preds >= self.binary_threshold).astype(int)
        return bool(y_pred_classes)


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
        content += f'pelastic_cost{{stage="{stage_idx}"}} {replicas * (adapter.memory_per_stage_replica[stage_idx] + cpu * core_to_G_RAM)}\n'
    return web.Response(body=content)


async def decide(request):
    return web.json_response(await adapter.adapt())

async def decide_ho(request):
    return web.json_response(await adapter.adapt_ho())

async def decide_vo(request):
    return web.json_response(await adapter.adapt_vo())

async def decide_il(request):
    return web.json_response(await adapter.adapt_il())


app = web.Application()
app.add_routes(
    [
        web.post("/initialize", initialize),
        web.get("/metrics", export_cost),
        web.post("/decide", decide),
        web.post("/decide-ho", decide_ho),
        web.post("/decide-vo", decide_vo),
        web.post("/decide-vomax", decide_vo),
        web.post("/decide-il", decide_il),
    ]
)
if __name__ == '__main__':
    web.run_app(app, host="0.0.0.0", port=8000, access_log=None)

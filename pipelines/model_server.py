import asyncio
import json
import os
import time
import torch

from aiohttp import ClientSession, ClientTimeout, TCPConnector, web


class ModelServer:

    def __init__(self) -> None:
        self.next_target_endpoint = os.getenv("NEXT_TARGET_ENDPOINT", "localhost")
        self.url_path = os.getenv("URL_PATH", "/receive")
        self.session = None
        self.model = None
    
    async def initialize(self):
        t = time.perf_counter()
        self.model = self.load_model()
        print("model loading time", self.__class__.__name__, time.perf_counter() - t)
        self.session = ClientSession(
            base_url=f"http://{self.next_target_endpoint}",
            timeout=ClientTimeout(total=int(os.getenv("TIMEOUT", 30))),
            connector=TCPConnector(limit=0)
        )
        t = time.perf_counter()
        self.warmup()
        print("warmup took:", self.__class__.__name__, time.perf_counter() - t)

    def load_model(self):
        raise NotImplementedError
    
    def warmup(self):
        pass
    
    def preprocess(self, data):
        return data
    
    def batch_preprocess(self, batch: list):
        return batch
    
    def inference(self, batch):
        return self.model(batch)
    
    def convert_batch_result(self, preds):
        return preds
    
    def get_next_target_data(self, pred):
        return pred

    async def infer(self, req) -> dict:
        batch = []
        arrival_time = time.time()
        c = 0
        for query in req:
            data = query["data"]
            query[f"arrival-{self.__class__.__name__}"] = arrival_time
            t = time.perf_counter()
            inp = self.preprocess(data)
            query[f"{self.__class__.__name__}-prep-time"] = round(time.perf_counter() - t, 3)
            batch.append(inp)
            c += 1

        t = time.perf_counter()
        batch = self.batch_preprocess(batch)
        batch_prep_time = round(time.perf_counter() - t, 3)
        t = time.perf_counter()
        preds = self.inference(batch)
        infer_time = round(time.perf_counter() - t, 3)
        t = time.perf_counter()
        preds = self.convert_batch_result(preds)
        convert_batch_time = round(time.perf_counter() - t, 3)
        
        
        tasks = []
        for i in range(c):
            to_send = req[i]
            to_send["data"] = self.get_next_target_data(preds[i])
            to_send[f"{self.__class__.__name__}-batch-prep-time-{c}"] = batch_prep_time
            to_send[f"{self.__class__.__name__}-batch-inference-time-{c}"] = infer_time
            to_send[f"{self.__class__.__name__}-batch-convert-time-{c}"] = convert_batch_time
            to_send[f"leaving-{self.__class__.__name__}"] = time.time()

            tasks.append(asyncio.create_task(self.send(to_send)))
        return {"received": True, "node": os.uname()[1]}
    
    async def send(self, data):
        async with self.session.post(self.url_path, data=json.dumps(data)) as response:
            await response.text()
            return


def add_base_routes(app: web.Application, model_server: ModelServer):
    async def initialize(request):
        if model_server.model is None:
            req = await request.json()
            torch.set_num_interop_threads(int(os.getenv("INTEROP_THREADS", 1)))
            torch.set_num_threads(int(req["threads"]))
            print(f"Torch parallelism threads: inter={torch.get_num_interop_threads()}, intra={torch.get_num_threads()}")
            await model_server.initialize()
            return web.json_response({"success": True})
        return web.json_response({"success": False, "message": "Already initialized."})

    async def update_threads(request):
        req = await request.json()
        t = time.perf_counter()
        torch.set_num_threads(int(req["threads"]))
        torch_thread_update_took = round(time.perf_counter() - t, 3)
        t = time.perf_counter()
        if os.getenv("WARMUP_AFTER_THREAD_UPDATE", "false").lower() == "true" and int(req["threads"] > 1):
            model_server.warmup()
        warmup_time = round(time.perf_counter() - t, 3)
        print(
            "Torch updating threads:",
            req["threads"], torch.get_num_threads(),
            "torch_update_took", torch_thread_update_took,
            "warmup_took", warmup_time
        )

        return web.json_response({"success": True, "torch_update_took": torch_thread_update_took, "warmup_took": warmup_time})

    async def infer(request):
        req = await request.json()
        return web.json_response(await model_server.infer(req))
    
    app.add_routes(
        [
            web.post("/initialize", initialize),
            web.post("/update-threads", update_threads),
            web.post("/infer", infer),
        ]
    )


import asyncio
import base64
import io
import json
import numpy as np
import os
import time
import torch

from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
from aiohttp import ClientSession, ClientTimeout, TCPConnector, web
from PIL import Image


class Classifier:

    def __init__(self) -> None:
        self.model = self.load_model()
        self.next_target_ip = os.getenv("NEXT_TARGET_IP", "localhost")
        self.next_target_port = os.getenv("NEXT_TARGET_PORT", "8080")
        self.url_path = os.getenv("URL_PATH", "/receive")
        self.session = None
    
    async def initialize(self):
        self.session = ClientSession(
            base_url=f"http://{self.next_target_ip}:{self.next_target_port}",
            timeout=ClientTimeout(total=int(os.getenv("TIMEOUT", 30))),
            connector=TCPConnector(limit=0)
        )

    def load_model(self):
        torch.set_num_interop_threads(1)
        self.weights = ResNet50_Weights.IMAGENET1K_V1
        model = resnet50(weights=self.weights)
        model.eval()
        self.preprocessor = self.weights.transforms()
        return model

    
    async def infer(self, req) -> dict:
        batch = []
        arrival_time = time.time()
        for query in req:
            data = query["data"]
            query["arrival-classifier"] = arrival_time
            decoded = base64.b64decode(data)
            inp = Image.open(io.BytesIO(decoded))
            inp = self.preprocessor(inp)
            batch.append(inp)

        batch = torch.from_numpy(np.array(batch))
        t = time.perf_counter()
        preds = self.model(batch)
        t = time.perf_counter() - t
        
        tasks = []
        for i in range(len(batch)):
            labels = []
            for idx in list(preds[i].sort()[1])[-1:-6:-1]:
                labels.append(self.weights.meta["categories"][idx])
            to_send = req[i]
            to_send["data"] = labels
            to_send[f"classifier-batch-inference-time-{len(batch)}"] = t
            to_send["leaving-classifier"] = time.time()

            tasks.append(asyncio.create_task(self.send(to_send)))
        return {"received": True}
    
    async def send(self, data):
        async with self.session.post(self.url_path, data=json.dumps(data)) as response:
            await response.text()
            return
        

model_server = Classifier()

async def initialize(request):
    await model_server.initialize()
    return web.json_response({"success": True})

async def update_threads(request):
    req = await request.json()
    torch.set_num_threads(int(req["threads"]))
    return web.json_response({"success": True})

async def infer(request):
    req = await request.json()
    return web.json_response(await model_server.infer(req))


app = web.Application()
app.add_routes(
    [
        web.post("/initialize", initialize),
        web.post("/update-threads", update_threads),
        web.post("/infer", infer),
    ]
)

if __name__ == '__main__':
    web.run_app(app, host="0.0.0.0", port=int(os.getenv("PORT", 8003)), access_log=None)

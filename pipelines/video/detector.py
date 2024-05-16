import asyncio
import base64
import io
import json
import numpy as np
import os
import torch
from aiohttp import ClientSession, ClientTimeout, TCPConnector, web
from PIL import Image


class Detector:

    def __init__(self) -> None:
        self.model = self.load_model()
        self.next_target_ip = os.getenv("NEXT_TARGET_IP", "localhost")
        self.next_target_port = os.getenv("NEXT_TARGET_PORT", "8002")
        self.url_path = os.getenv("URL_PATH", "/predict")
        self.session = None
    
    async def initialize(self):
        self.session = ClientSession(
            base_url=f"http://{self.next_target_ip}:{self.next_target_port}",
            timeout=ClientTimeout(total=int(os.getenv("TIMEOUT", 30))),
            connector=TCPConnector(limit=0)
        )
    
    def load_model(self):
        return torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path="./yolov5s.pt",
        )
    
    async def infer(self, req: dict):
        batch = []
        for query in req:
            data = query["data"]
            decoded = base64.b64decode(data)
            image = Image.open(io.BytesIO(decoded))
            image = np.array(image)
            batch.append(image)
        batch_result = self.model(batch)
        tasks = []
        for i in range(len(batch)):
            res = batch_result.tolist()[i]
            best_detect = None
            best_detect_score = -1
            for cropped in res.crop(save=False):
                conf = round(float(cropped["conf"]), 2)
                if conf > best_detect_score:
                    best_detect = cropped
                    best_detect_score = conf
            buffered = io.BytesIO()
            im_base64 = Image.fromarray(best_detect["im"])
            im_base64.save(buffered, format="JPEG")
            im_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            tasks.append(asyncio.create_task(self.send(im_base64)))

            print("best detect", i, best_detect)

        # await asyncio.gather(*tasks)
        
        return {"received": True}
    
    async def send(self, data):
        async with self.session.post(self.url_path, data=data) as response:
            await response.text()
            return
            


model_server = Detector()

async def infer(request):
    req = await request.json()
    return web.json_response(await model_server.infer(req))

async def initialize(request):
    await model_server.initialize()
    return web.json_response({"success": True})


app = web.Application()
app.add_routes(
    [
        web.post("/initialize", initialize),
        web.post("/infer", infer),
    ]
)
if __name__ == '__main__':
    web.run_app(app, host="0.0.0.0", port=8001, access_log=None)



import base64
import io
import json
import numpy as np
import os
import torch

from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
from aiohttp import web
from PIL import Image


class Classifier:

    def __init__(self) -> None:
        self.model = self.load_model()
        self.next_target_ip = os.getenv("NEXT_TARGET_IP", "localhost")
        self.next_target_port = os.getenv("NEXT_TARGET_PORT", "8004")
        self.url_path = os.getenv("URL_PATH", "/predict")
        self.session = None

    def load_model(self):
        self.weights = ResNet50_Weights.IMAGENET1K_V1
        model = resnet50(weights=self.weights)
        model.eval()
        self.preprocessor = self.weights.transforms()
        return model

    
    async def infer(self, req: dict) -> dict:
        batch = []
        for query in req:
            data = query["data"]
            decoded = base64.b64decode(data)
            inp = Image.open(io.BytesIO(decoded))
            inp = self.preprocessor(inp)
            batch.append(inp)

        batch = torch.from_numpy(np.array(batch))
        preds = self.model(batch)

        for i in range(len(batch)):
            for idx in list(preds[i].sort()[1])[-1:-6:-1]:
                print(self.weights.meta["categories"][idx])
            print()
        return {"received": True}
        

model_server = Classifier()

async def initialize(request):
    await model_server.initialize()
    return web.json_response({"success": True})

async def infer(request):
    req = await request.json()
    return web.json_response(await model_server.infer(req))



app = web.Application()
app.add_routes(
    [
        web.post("/initialize", initialize),
        web.post("/infer", infer),
    ]
)

if __name__ == '__main__':
    web.run_app(app, host="0.0.0.0", port=8003, access_log=None)

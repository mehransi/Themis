import asyncio
import base64
import io
import json
import numpy as np
import os
import time
import torch
from aiohttp import web
from PIL import Image

from model_server import ModelServer, add_base_routes

class Detector(ModelServer):
    
    def load_model(self):
        torch.set_num_interop_threads(1)
        return torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path="./yolov5n.pt",
        )
    
    def preprocess(self, data):
        decoded = base64.b64decode(data)
        image = Image.open(io.BytesIO(decoded))
        return np.array(image)
    
    def convert_batch_result(self, preds):
        return preds.tolist()
    
    def get_next_target_data(self, pred):
        best_detect = None
        best_detect_score = -1
        for cropped in pred.crop(save=False):
            conf = round(float(cropped["conf"]), 2)
            if conf > best_detect_score:
                best_detect = cropped
                best_detect_score = conf
        buffered = io.BytesIO()
        im_base64 = Image.fromarray(best_detect["im"])
        im_base64.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
            

model_server = Detector()
app = web.Application()

add_base_routes(app, model_server)

if __name__ == '__main__':
    web.run_app(app, host="0.0.0.0", port=int(os.getenv("PORT", 8001)), access_log=None)

import base64
import io
import numpy as np
import os
import torch
from aiohttp import web
from PIL import Image
from ultralytics import YOLO

from model_server import ModelServer, add_base_routes

class Detector(ModelServer):
    
    def load_model(self):
        model = YOLO("./yolov10n")
        model.eval()
        self.__example_image = Image.open("./zidane.jpg")
        return model
        
    def warmup(self):
        im = np.array(self.__example_image)
        self.model.predict(im, device=torch.device("cpu"), verbose=False)
    
    def preprocess(self, data):
        decoded = base64.b64decode(data)
        image = Image.open(io.BytesIO(decoded))
        return np.array(image)
    
    def get_next_target_data(self, pred):
        best_detect = None
        best_detect_score = -1
        for cropped in pred.boxes:
            conf = round(float(cropped.conf), 2)
            if conf > best_detect_score:
                best_detect = cropped
                best_detect_score = conf
        buffered = io.BytesIO()
        img = Image.fromarray(pred.orig_img).crop(best_detect.xyxy.tolist()[0])
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
            

model_server = Detector()
app = web.Application()

add_base_routes(app, model_server)

if __name__ == '__main__':
    web.run_app(app, host="0.0.0.0", port=int(os.getenv("PORT", 8001)), access_log=None)

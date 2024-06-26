import base64
import io
import numpy as np
import os
import torch

from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights
from aiohttp import web
from PIL import Image

from model_server import ModelServer, add_base_routes

class Classifier(ModelServer):

    def load_model(self):
        torch.set_num_interop_threads(1)
        self.categories = ResNet18_Weights.IMAGENET1K_V1.meta["categories"]
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.eval()
        self.preprocessor = ResNet18_Weights.IMAGENET1K_V1.transforms()
        return model
    
    def preprocess(self, data):
        decoded = base64.b64decode(data)
        inp = Image.open(io.BytesIO(decoded))
        return self.preprocessor(inp)
    
    def batch_preprocess(self, batch: list):
        return torch.from_numpy(np.array(batch))
    
    def inference(self, batch):
        return self.model(batch)
    
    def get_next_target_data(self, pred):
        return self.categories[torch.argmax(pred).item()]
        

model_server = Classifier()
app = web.Application()

add_base_routes(app, model_server)

if __name__ == '__main__':
    web.run_app(app, host="0.0.0.0", port=int(os.getenv("PORT", 8003)), access_log=None)

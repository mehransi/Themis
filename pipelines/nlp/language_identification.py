import os

from aiohttp import web
from transformers import pipeline

from model_server import ModelServer, add_base_routes

class LanguageIdentification(ModelServer):

    def load_model(self):
        model = pipeline(task="text-classification", model="./dinalzein/xlm-roberta-base-finetuned-language-identification", batch_size=8)
        return model
    
    def warmup(self):
        text = "Intégration des échelles horizontale et verticale pour l'inférence Servir les systèmes"
        self.model([text])
        
    def inference(self, batch):
        identification = super().inference(batch)
        return batch


model_server = LanguageIdentification()
app = web.Application()

add_base_routes(app, model_server)

if __name__ == '__main__':
    web.run_app(app, host="0.0.0.0", port=int(os.getenv("PORT", 8001)), access_log=None)

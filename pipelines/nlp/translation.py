import os

from aiohttp import web
from transformers import pipeline

from model_server import ModelServer, add_base_routes

class Translation(ModelServer):

    def load_model(self):
        return pipeline(task="translation_fr_to_en", model="./Helsinki-NLP/opus-mt-fr-en")
    
    def warmup(self):
        text = "Intégration des échelles horizontale et verticale pour l'inférence Servir les systèmes"
        self.model([text])
        
    def get_next_target_data(self, pred):
        return pred["translation_text"]


model_server = Translation()
app = web.Application()

add_base_routes(app, model_server)

if __name__ == '__main__':
    web.run_app(app, host="0.0.0.0", port=int(os.getenv("PORT", 8001)), access_log=None)

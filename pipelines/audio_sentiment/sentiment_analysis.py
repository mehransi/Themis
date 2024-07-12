import os

from aiohttp import web
from transformers import pipeline

from model_server import ModelServer, add_base_routes


class SentimentAnalysis(ModelServer):
    def load_model(self):
        model = pipeline(
            task="sentiment-analysis", model="./Souvikcmsa/SentimentAnalysisDistillBERT"
        )
        return model
    
    def warmup(self):
        self.model(["nor is mister cultar's manner less interesting than his matter"])


model_server = SentimentAnalysis()
app = web.Application()

add_base_routes(app, model_server)

if __name__ == '__main__':
    web.run_app(app, host="0.0.0.0", port=int(os.getenv("PORT", 8003)), access_log=None)

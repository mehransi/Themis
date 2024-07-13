import os

from aiohttp import web
from transformers import pipeline


from model_server import ModelServer, add_base_routes

class Summarizer(ModelServer):

    def load_model(self):
        return pipeline("summarization", "./stevhliu/my_awesome_billsum_model", max_length=11, min_length=11)
    
    def warmup(self):
        text = "Hello. I'm an AI."
        self.model([text])
    
    def get_next_target_data(self, pred):
        return pred["summary_text"]


model_server = Summarizer()
app = web.Application()

add_base_routes(app, model_server)

if __name__ == '__main__':
    web.run_app(app, host="0.0.0.0", port=int(os.getenv("PORT", 8001)), access_log=None)

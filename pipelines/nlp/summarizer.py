import os

from aiohttp import web
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


from model_server import ModelServer, add_base_routes

class Summarizer(ModelServer):

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained("./google/roberta2roberta_L-24_bbc_tokenizer")
        summarizer = AutoModelForSeq2SeqLM.from_pretrained("./google/roberta2roberta_L-24_bbc_model")
        return summarizer
    
    def warmup(self):
        text = "Integration of horizontal and vertical scales for inference Serving systems"
        input_ids = self.tokenizer([text], return_tensors="pt").input_ids
        output_ids = self.model.generate(input_ids)
        self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    
    def batch_preprocess(self, batch: list):
        return self.tokenizer(batch, return_tensors="pt").input_ids
        
    def inference(self, batch):
        output_ids = self.model.generate(batch)
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)


model_server = Summarizer()
app = web.Application()

add_base_routes(app, model_server)

if __name__ == '__main__':
    web.run_app(app, host="0.0.0.0", port=int(os.getenv("PORT", 8001)), access_log=None)

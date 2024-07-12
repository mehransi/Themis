import base64
import io
import os
import soundfile as sf

from aiohttp import web
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration

from model_server import ModelServer, add_base_routes

class AudioToText(ModelServer):

    def load_model(self):
        model = Speech2TextForConditionalGeneration.from_pretrained("./facebook/s2t-small-librispeech-asr-model")
        self.processor = Speech2TextProcessor.from_pretrained("./facebook/s2t-small-librispeech-asr-processor")
        self.__sample, _ = sf.read(io.BytesIO(open("audio.flac", "rb").read()), dtype="float32")
        return model
    
    def warmup(self):
        input_features = self.processor(
            self.__sample,
            sampling_rate=16_000,
            return_tensors="pt"
        ).input_features
        generated_ids = self.model.generate(input_features=input_features)
        self.processor.batch_decode(generated_ids)
        
    
    def preprocess(self, data):
        decoded = base64.b64decode(data)
        audio, _ = sf.read(io.BytesIO(decoded), dtype='float32')
        return audio
    
    def batch_preprocess(self, batch: list):
        return self.processor(batch, sampling_rate=16000, return_tensors="pt")
    
    def inference(self, batch):
        generated_ids = self.model.generate(batch["input_features"], attention_mask=batch["attention_mask"])
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return transcription


model_server = AudioToText()
app = web.Application()

add_base_routes(app, model_server)

if __name__ == '__main__':
    web.run_app(app, host="0.0.0.0", port=int(os.getenv("PORT", 8001)), access_log=None)

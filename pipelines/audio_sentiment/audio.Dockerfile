FROM python:3.12.3-slim

WORKDIR /app

RUN apt update -y && apt install -y libsndfile1-dev
RUN pip install --upgrade pip
RUN pip install torch==2.3.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install transformers==4.41.1
RUN pip install sentencepiece==0.2.0
RUN pip install datasets==2.19.1
RUN pip install soundfile==0.12.1
RUN pip install librosa==0.10.2.post1

COPY model_server.py model_server.py
COPY audio_to_text.py audio_to_text.py

ENTRYPOINT ["python", "audio_to_text.py"]
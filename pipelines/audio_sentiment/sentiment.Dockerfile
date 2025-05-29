FROM python:3.12.3-slim

WORKDIR /app

RUN pip install --upgrade pip
RUN pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install transformers==4.41.1
RUN pip install sentencepiece==0.2.0
RUN pip install datasets==2.19.1
RUN pip install aiohttp==3.9.5

COPY Souvikcmsa Souvikcmsa
COPY model_server.py model_server.py
COPY sentiment_analysis.py sentiment_analysis.py


ENTRYPOINT ["taskset", "-c", "0-7", "python", "sentiment_analysis.py"]
FROM python:3.12.3-slim

WORKDIR /app

RUN pip install --upgrade pip
RUN pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install transformers==4.41.1
RUN pip install aiohttp==3.9.5

COPY Helsinki-NLP Helsinki-NLP
COPY model_server.py model_server.py
COPY translation.py translation.py


ENTRYPOINT ["python", "translation.py"]
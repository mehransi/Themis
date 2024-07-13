FROM python:3.12.3-slim

WORKDIR /app

RUN pip install --upgrade pip
RUN pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install transformers==4.41.1
RUN pip install aiohttp==3.9.5

COPY dinalzein dinalzein
COPY model_server.py model_server.py
COPY language_identification.py language_identification.py


ENTRYPOINT ["python", "language_identification.py"]
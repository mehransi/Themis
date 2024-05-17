FROM python:3.12.3-slim

WORKDIR /app
RUN pip install --upgrade pip

RUN pip install aiohttp==3.9.5

COPY main.py main.py

ENTRYPOINT ["python", "main.py"]
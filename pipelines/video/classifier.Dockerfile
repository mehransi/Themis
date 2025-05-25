FROM python:3.12.3-slim

WORKDIR /app
RUN pip install --upgrade pip
RUN pip install numpy==1.26.3
RUN pip install pillow==10.3.0
RUN pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install aiohttp==3.9.5

COPY resnet18-f37072fd.pth /root/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
COPY model_server.py model_server.py
COPY classifier.py classifier.py
COPY zidane.jpg zidane.jpg

ENTRYPOINT ["taskset", "-c", "0-7", "python", "classifier.py"]

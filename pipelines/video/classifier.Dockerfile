FROM python:3.12.3-slim

WORKDIR /app
RUN pip install --upgrade pip
RUN pip install numpy==1.26.3
RUN pip install pillow==10.3.0
RUN pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install aiohttp==3.9.5

COPY resnet50-0676ba61.pth /root/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth
COPY classifier.py classifier.py

ENTRYPOINT ["python", "classifier.py"]
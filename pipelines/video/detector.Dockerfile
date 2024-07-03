FROM python:3.10-slim

WORKDIR /app
RUN pip install --upgrade pip
RUN pip install numpy==1.26.3
RUN pip install pillow==10.3.0
RUN pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install ultralytics==8.2.10
RUN pip install aiohttp==3.9.5
RUN pip install opencv-python-headless
RUN pip install gitpython==3.1.43

COPY yolov5n.pt yolov5n.pt
COPY model_server.py model_server.py
COPY detector.py detector.py
COPY zidane.jpg zidane.jpg
COPY ultralytics_yolov5_master ultralytics_yolov5_master

ENTRYPOINT ["python", "detector.py"]

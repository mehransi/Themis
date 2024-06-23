FROM python:3.10-slim

WORKDIR /app

RUN apt-get -y update && apt-get -y install git

RUN pip install --upgrade pip

RUN pip install tensorflow==2.16.1
RUN pip install keras==3.3.3
RUN pip install git+https://github.com/reconfigurable-ml-pipeline/kubernetes-python-client.git

COPY LSTM/lstm.keras /app/lstm.keras
ENV LSTM_MODEL=/app/lstm.keras

COPY optimizer.py optimizer.py
COPY main.py main.py
COPY runner.sh runner.sh
COPY signal_sender.py signal_sender.py

ENTRYPOINT ["bash", "runner.sh"]
import requests
import time
import logging
import os
from threading import Thread


logger = logging.getLogger()


def signal():
    try:
        requests.post("http://localhost:8000/decide", timeout=10)
    except (TimeoutError, requests.exceptions.Timeout, requests.exceptions.ReadTimeout):
        logger.info(f"The reconfiguration timed out")
        
    
    
if __name__ == "__main__":
    time.sleep(60 * int(os.environ["FIRST_DECIDE_DELAY_MINUTES"]))
    decision_interval = int(os.environ["DECISION_INTERVAL"])
    while True:
        thread = Thread(target=signal)
        thread.run()
        time.sleep(decision_interval)

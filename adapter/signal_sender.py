import requests
import time
import logging
import os
from threading import Thread


logger = logging.getLogger()
adapter_type = os.getenv("ADAPTER_TYPE")
assert adapter_type is not None, "Pass ADAPTER_TYPE as env var. It gets one of hv, ho,vo, and vomax values"

def signal():
    path = "decide"
    if adapter_type in ["ho", "vo", "vomax", "inferline"]:
        path = path + f"-{adapter_type}"
    
    try:
        requests.post(f"http://localhost:8000/{path}", timeout=10)
    except (TimeoutError, requests.exceptions.Timeout, requests.exceptions.ReadTimeout):
        logger.info(f"The reconfiguration timed out")
        
    
    
if __name__ == "__main__":
    time.sleep(int(60 * float(os.environ["FIRST_DECIDE_DELAY_MINUTES"])))
    decision_interval = int(os.environ["DECISION_INTERVAL"])
    logger.info(f"Adapter type is {adapter_type}")
    while True:
        thread = Thread(target=signal)
        thread.run()
        time.sleep(decision_interval)

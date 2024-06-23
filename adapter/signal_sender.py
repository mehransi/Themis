import requests
import time
import os


if __name__ == "__main__":
    time.sleep(60 * int(os.environ["FIRST_DECIDE_DELAY_MINUTES"]))
    decision_interval = int(os.environ["DECISION_INTERVAL"])
    while True:
        t = time.perf_counter()
        requests.post("http://localhost:8000/decide", timeout=10)
        print(f"reconfiguration took {time.perf_counter() - t}s")
        time.sleep(decision_interval)
import sys
import os
import json
import numpy as np

SOURCE = sys.argv[1]
PERCENTILE = int(sys.argv[2])

if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    data_dir = f"{current_dir}/models/{SOURCE.lower()}"
    
    profiling_data = []
    for cpu in range(1, 9):
        for batch in range(1, 9):
            try:
                with open(f"{data_dir}/data/{SOURCE}_latencies_cpu{cpu}_batch{batch}.json") as f:
                    latencies = json.load(f)
                    latencies = list(map(lambda x: x["e2e"], latencies))
                    latency = np.percentile(latencies, PERCENTILE)
                    profiling_data.append({"cpu": cpu, "batch": batch, "latency": latency})
            except:
                pass
    with open(f"{data_dir}/profiling-{PERCENTILE}.json", "w") as f2:
        json.dump(profiling_data, f2, indent=2)

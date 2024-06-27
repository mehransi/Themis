import sys
import glob
import os
import re
import json
import numpy as np

SOURCE = sys.argv[1]


if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    data_dir = f"{current_dir}/models/{SOURCE.lower()}"
    
    profiling_data = []
    for cpu in range(1, 11):
        for batch in range(1, 11):
            with open(f"{data_dir}/{SOURCE}_latencies_core{cpu}_batch{batch}.json") as f:
                latencies = json.load(f)
                latencies = list(map(lambda x: x["e2e"], latencies))
                latency = np.percentile(latencies, 95)
                profiling_data.append({"cpu": cpu, "batch": batch, "latency": latency})
    with open(f"{data_dir}/profiling.json", "w") as f2:
        json.dump(profiling_data, f2, indent=2)

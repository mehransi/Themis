import sys
import os
import json
import numpy as np

SOURCE = sys.argv[1]


if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    data_dir = f"{current_dir}/models/{SOURCE.lower()}"
    
    profiling_data = []
    for inter in [1, 4]:
        for intra in [1, 4]:
            for batch in [1, 2, 4, 8]:
                with open(f"{data_dir}/data-parallelism/{SOURCE}_latencies_inter{inter}_intra{intra}_batch{batch}.json") as f:
                    latencies = json.load(f)
                    latencies = list(map(lambda x: x["e2e"], latencies))
                    latency = np.percentile(latencies, 95)
                    profiling_data.append({"cpu": 4, "inter": inter, "intra": intra, "batch": batch, "latency": latency})
    with open(f"{data_dir}/profiling-parallelism.json", "w") as f2:
        json.dump(profiling_data, f2, indent=2)

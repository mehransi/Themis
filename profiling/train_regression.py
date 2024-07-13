import sys
import os
import json
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

    
def batch_cost_latency_model(cpu_batch_tuple, alpha, beta, gamma, zeta):
        cpu, batch = cpu_batch_tuple
        return alpha * batch / cpu + beta * batch + gamma / cpu + zeta


model = sys.argv[1]
with open(f"{os.path.dirname(__file__)}/models/{model}/profiling.json") as f:
        profiling_data = json.load(f)


training_data = list(filter(lambda x: x["cpu"] in [1, 2, 4, 8], profiling_data))
training_data = list(filter(lambda x: x["batch"] in [1, 2, 4, 8], training_data))
print("len profiling_data", len(profiling_data))
cpu_sizes = list(map(lambda x: x["cpu"], training_data))
batch_sizes = list(map(lambda x: x["batch"], training_data))
latencies = list(map(lambda x: x["latency"] * 1000, training_data)) # ms

params, _ = curve_fit(batch_cost_latency_model, (cpu_sizes, batch_sizes), latencies)
alpha, beta, gamma, zeta = params


with open(f"{os.path.dirname(__file__)}/models/{model}/parameters.json", "w") as f:
        json.dump({"alpha": alpha, "beta": beta, "gamma": gamma, "zeta": zeta}, f, indent=2)

b = []
l = []
rl = []
CPU = 6
for d in profiling_data:
    if d["cpu"] == CPU:
        l.append(d["latency"] * 1000)
        b.append(d["batch"])
        rl.append(batch_cost_latency_model((d["cpu"], d["batch"]), alpha, beta, gamma, zeta))

plt.scatter(b, l, label="p99")
plt.scatter(b, rl, label="regression")
plt.title(f"{model} profiling for cpu={CPU}")
plt.xlabel("batch")
plt.ylabel("latency (ms)")
plt.legend()
plt.show()

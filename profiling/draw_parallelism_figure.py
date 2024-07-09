import json
import matplotlib.pyplot as plt
import sys
import os


def batch_cost_latency_model(cpu_batch_tuple, alpha, beta, gamma, zeta):
        cpu, batch = cpu_batch_tuple
        return alpha * batch / cpu + beta * batch + gamma / cpu + zeta


x_data = batches = [1, 2, 4, 8]
colors = ["#5e3c99", "#b2abd2", "#fdb863", "#e66101"]
    
    
with open(f"{os.path.dirname(__file__)}/models/detector/profiling-parallelism.json") as f:
    profiling_data_detector = json.load(f)
    
detector_data = {}
for data in profiling_data_detector:
    if detector_data.get((data["inter"], data["intra"])) is None:
        detector_data[(data["inter"], data["intra"])] = []
    detector_data[(data["inter"], data["intra"])].append(data["latency"] * 1000)


with open(f"{os.path.dirname(__file__)}/models/classifier/profiling-parallelism.json") as f:
        profiling_data_classifier = json.load(f)

classifier_data = {}
for data in profiling_data_classifier:
    if classifier_data.get((data["inter"], data["intra"])) is None:
        classifier_data[(data["inter"], data["intra"])] = []
    classifier_data[(data["inter"], data["intra"])].append(data["latency"] * 1000)


def draw():
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(6.4, 4.4))
    idx = 0
    for (inter, intra), latencies in detector_data.items():
        axs[0].plot(x_data, latencies, label=f"inter, intra={inter}, {intra}", color=colors[idx])
        idx += 1
    axs[0].set_title("Yolov5n", fontweight="bold")
    axs[0].set_ylabel("Latency (ms)", fontsize=12)
    axs[0].set_xticks([])
    idx = 0
    for (inter, intra), latencies in classifier_data.items():
        axs[1].plot(x_data, latencies, label=f"inter, intra={inter}, {intra}", color=colors[idx])
        idx += 1
    axs[1].set_xticks(x_data)
    axs[1].set_title("Resnet18", fontweight="bold")
    axs[1].set_xlabel("batch", fontsize=12)
    axs[1].set_ylabel("Latency (ms)", fontsize=12)

    legend_fontsize = 10
    plt.rc('legend', fontsize=legend_fontsize)
    fig.tight_layout()
    plt.legend()
    fig.savefig("parallelism_batch_latency.pdf", dpi=600, format="pdf", pad_inches=0, bbox_inches="tight")
    

draw()
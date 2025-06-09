import json
import matplotlib.pyplot as plt
import sys
import os


legend_fontsize = 12
plt.rc('legend', fontsize=legend_fontsize)
font_small_size = 18
plt.rcParams.update({'font.size': font_small_size})
canvas_size = 4
plt.rcParams['figure.figsize'] = 2 * canvas_size, 12
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams['hatch.linewidth'] = 0.7
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

x_data = batches = [1, 2, 4]
colors = ['#2c7bb6', '#abd9e9', '#fdae61', '#d7191c', '#ffffbf']
markers = ["D", "D", "", ""]
    
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
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))
    idx = 0
    for (inter, intra), latencies in detector_data.items():
        axs[0].plot(x_data, latencies, label=f"inter, intra={inter}, {intra}", color=colors[idx], marker=markers[idx])
        idx += 1
    axs[0].set_title("Object Detection")
    axs[0].set_ylabel("Latency (ms)")
    axs[0].set_xticks([])
    idx = 0
    for (inter, intra), latencies in classifier_data.items():
        axs[1].plot(x_data, latencies, label=f"inter, intra={inter}, {intra}", color=colors[idx], marker=markers[idx])
        idx += 1
    axs[1].set_xticks(x_data)
    axs[1].set_title("Object Classification")
    axs[1].set_xlabel("batch")
    axs[1].set_ylabel("Latency (ms)")

    fig.tight_layout()
    plt.legend()
    fig.savefig("inter_intra.pdf", dpi=600, format="pdf", pad_inches=0, bbox_inches="tight")
    

draw()
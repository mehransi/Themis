import json
import matplotlib.pyplot as plt
import sys
import os


def batch_cost_latency_model(cpu_batch_tuple, alpha, beta, gamma, zeta):
        cpu, batch = cpu_batch_tuple
        return alpha * batch / cpu + beta * batch + gamma / cpu + zeta

x_data = [1, 2, 4, 6]
batches = [1, 2, 4, 6]
color_by_batch = {1: "#5e3c99", 2: "#b2abd2", 4: "#fdb863", 8: "#e66101"}
percentile = int(sys.argv[1])

with open(f"{os.path.dirname(__file__)}/models/detector/parameters-{percentile}.json") as f:
    detector_parameters = json.load(f)

with open(f"{os.path.dirname(__file__)}/models/classifier/parameters-{percentile}.json") as f:
    classifier_parameters = json.load(f)
    
    
with open(f"{os.path.dirname(__file__)}/models/detector/profiling-{percentile}.json") as f:
    profiling_data_detector = json.load(f)
    profiling_data_detector = list(filter(lambda x: x["cpu"] in x_data, profiling_data_detector))
    profiling_data_detector = list(filter(lambda x: x["batch"] in batches, profiling_data_detector))
    
detector_data_by_batch = {}
for data in profiling_data_detector:
    if detector_data_by_batch.get(data["batch"]) is None:
        detector_data_by_batch[data["batch"]] = []
    detector_data_by_batch[data["batch"]].append(data["latency"] * 1000)


with open(f"{os.path.dirname(__file__)}/models/classifier/profiling-{percentile}.json") as f:
        profiling_data_classifier = json.load(f)
        profiling_data_classifier= list(filter(lambda x: x["cpu"] in x_data, profiling_data_classifier))
        profiling_data_classifier = list(filter(lambda x: x["batch"] in batches, profiling_data_classifier))

classifier_data_by_batch = {}
for data in profiling_data_classifier:
    if classifier_data_by_batch.get(data["batch"]) is None:
        classifier_data_by_batch[data["batch"]] = []
    classifier_data_by_batch[data["batch"]].append(data["latency"] * 1000)


def draw():
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(6.4, 4.4))
    for b, latencies in detector_data_by_batch.items():
        print(x_data, latencies)
        axs[0].plot(x_data, latencies, label=f"batch={b}", color=color_by_batch[b])
        axs[0].plot(
            x_data, list(map(lambda c: batch_cost_latency_model((c, b), **detector_parameters), x_data)),
            linestyle="dashed",
            color=color_by_batch[b]
        )
    axs[0].set_title("Yolov5n", fontweight="bold")
    axs[0].set_ylabel("Latency (ms)", fontsize=12)
    axs[0].set_xticks([])
    for b, latencies in classifier_data_by_batch.items():
        axs[1].plot(x_data, latencies, label=f"batch={b}", color=color_by_batch[b])
        axs[1].plot(
            x_data, list(map(lambda c: batch_cost_latency_model((c, b), **classifier_parameters), x_data)),
            linestyle="dashed",
            color=color_by_batch[b],
            label="predicted" if b == batches[-1] else None
        )
    axs[1].set_xticks(x_data)
    axs[1].set_title("Resnet18", fontweight="bold")
    axs[1].set_xlabel("CPU (cores)", fontsize=12)
    axs[1].set_ylabel("Latency (ms)", fontsize=12)

    legend_fontsize = 10
    plt.rc('legend', fontsize=legend_fontsize)
    # font_size = 18
    # plt.rc("font", size=font_small_size)
    # plt.rcParams.update({'font.size': font_size})
    fig.tight_layout()
    legend = plt.legend()
    legend.legend_handles[-1].set_color("gray")
    fig.savefig("cpu_batch_latency.pdf", dpi=600, format="pdf", pad_inches=0, bbox_inches="tight")
    

draw()
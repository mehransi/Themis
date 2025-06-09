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

percentile = int(sys.argv[1])
model1 = sys.argv[2]
model2 = sys.argv[3]

model_to_title = {"detector": "Object Detection", "classifier": "Object Classifier"}

def batch_cost_latency_model(cpu_batch_tuple, alpha, beta, gamma, zeta):
        cpu, batch = cpu_batch_tuple
        return alpha * batch / cpu + beta * batch + gamma / cpu + zeta

x_data = [1, 2, 4,]
batches = [1, 2, 3, 4,]
# colors = [ '#005906', '#f7cc3e', 'plum', 'chocolate', '#3894fc']
colors = ['#2c7bb6', '#abd9e9', '#fdae61', '#d7191c', '#ffffbf']


with open(f"{os.path.dirname(__file__)}/models/{model1}/parameters-{percentile}.json") as f:
    detector_parameters = json.load(f)

with open(f"{os.path.dirname(__file__)}/models/{model2}/parameters-{percentile}.json") as f:
    classifier_parameters = json.load(f)


with open(f"{os.path.dirname(__file__)}/models/{model1}/profiling-{percentile}.json") as f:
    profiling_data_detector = json.load(f)
    profiling_data_detector = list(filter(lambda x: x["cpu"] in x_data, profiling_data_detector))
    profiling_data_detector = list(filter(lambda x: x["batch"] in batches, profiling_data_detector))
    
detector_data_by_batch = {}
for data in profiling_data_detector:
    if detector_data_by_batch.get(data["batch"]) is None:
        detector_data_by_batch[data["batch"]] = []
    detector_data_by_batch[data["batch"]].append(data["latency"] * 1000)


with open(f"{os.path.dirname(__file__)}/models/{model2}/profiling-{percentile}.json") as f:
        profiling_data_classifier = json.load(f)
        profiling_data_classifier= list(filter(lambda x: x["cpu"] in x_data, profiling_data_classifier))
        profiling_data_classifier = list(filter(lambda x: x["batch"] in batches, profiling_data_classifier))

classifier_data_by_batch = {}
for data in profiling_data_classifier:
    if classifier_data_by_batch.get(data["batch"]) is None:
        classifier_data_by_batch[data["batch"]] = []
    classifier_data_by_batch[data["batch"]].append(data["latency"] * 1000)


def draw():
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))
    idx = 0
    for b, latencies in detector_data_by_batch.items():
        print(x_data, latencies)
        axs[0].plot(x_data, latencies, label=f"batch={b}", color=colors[idx])
        axs[0].plot(
            x_data, list(map(lambda c: batch_cost_latency_model((c, b), **detector_parameters), x_data)),
            linestyle="dashed",
            color=colors[idx]
        )
        idx += 1
    axs[0].set_title(model_to_title.get(model1, model1))
    axs[0].set_ylabel("Latency (ms)")
    axs[0].set_xticks([])
    
    idx = 0
    for b, latencies in classifier_data_by_batch.items():
        axs[1].plot(x_data, latencies, label=f"batch={b}", color=colors[idx])
        axs[1].plot(
            x_data, list(map(lambda c: batch_cost_latency_model((c, b), **classifier_parameters), x_data)),
            linestyle="dashed",
            color=colors[idx],
            label="predicted" if b == batches[-1] else None
        )
        idx += 1
    axs[1].set_xticks(x_data)
    axs[1].set_title(model_to_title.get(model2, model2))
    axs[1].set_xlabel("CPU (cores)",)
    axs[1].set_ylabel("Latency (ms)")
    

    fig.tight_layout()
    legend = plt.legend(ncols=3)
    legend.legend_handles[-1].set_color("gray")
    fig.savefig(f"cpu_batch_latency_{model1}_{model2}.pdf", dpi=600, format="pdf", pad_inches=0, bbox_inches="tight")
    

draw()
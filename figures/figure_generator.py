import random
import shutil
import json
import matplotlib.pyplot as plt
import pandas as pd
from pdfCropMargins import crop
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import csv
import numpy as np

# import math
# import random
# import matplotlib.patches as patches
# from copy import deepcopy
# import os
# import matplotlib.gridspec as gridspec
# from matplotlib.ticker import NullFormatter, PercentFormatter
# import statistics
# from scipy.optimize import curve_fit


def batch_cost_latency_model(bc, alpha, beta, gamma, zeta):
    b, c = bc
    return alpha * b / c + beta * b + gamma / c + zeta


def batch_cost_latency_calculation(b, c, alpha, beta, gamma, zeta):
    return int(alpha * b / c + beta * b + gamma / c + zeta)


def crop_pdf(filename):
    crop(["-p", "0", "-gui", filename])
    shutil.move(project_path + filename.split('.pdf')[0].split('/')[7] + '_cropped.pdf', filename)


def vertical_vs_horizontal_long():
    fig, (ax1, ax2) = plt.subplots(2, 1)
    time_range = range(1, 61)
    workload = [20] * 10
    workload.extend([120] * 40)
    workload.extend([20] * 10)

    vertical_throughput = [26] * 11
    vertical_throughput.extend([126] * 20)  # After 20 seconds, we go to horizontal (8 cores)
    vertical_throughput.extend([130] * 20)  # After 20 seconds, we go to horizontal (5 cores)
    vertical_throughput.extend([26] * 9)

    cpu_vertical = [1] * 11
    cpu_vertical.extend([8] * 20)
    cpu_vertical.extend([5] * 20)
    cpu_vertical.extend([1] * 9)
    # horizontal_throughput = [26] * 16
    # horizontal_throughput.extend([130] * 40)
    # horizontal_throughput.extend([26] * 4)

    seen = [False] * 60
    vertical_violation = 0
    # horizontal_violation = 0

    ax1.plot(time_range, vertical_throughput, '--', color=colors[0], label='Ideal Throughput')
    ax1.plot(time_range, workload, color='black', label='Workload')
    ax2.plot(time_range, cpu_vertical, '--', color=colors[1], label='CPU')
    # ax2.plot(time_range, workload, color='black', label='Workload')

    for i in range(len(time_range)):
        if vertical_throughput[i] < workload[i] and not seen[i]:
            j = i
            while j < len(time_range) and vertical_throughput[j] < workload[j]:
                vertical_violation += (workload[j] - vertical_throughput[j])
                j += 1
            # ax1.fill_between(time_range[i: j], workload[i: j], vertical_throughput[i: j],
            #                  facecolor="k", hatch='///', edgecolor='k', alpha=.5)
            for k in range(i, j):
                seen[k] = True
    ax1.fill_between(time_range, workload, facecolor="k", hatch='///', edgecolor='k', alpha=.5)
    ax1.fill_between(time_range, vertical_throughput, color='w')
    # ax2.fill_between(time_range, workload, facecolor="k", hatch='///', edgecolor='k', alpha=.5)
    # ax2.fill_between(time_range, horizontal_throughput, color='w')

    # seen = [False] * 60
    # for i in range(len(time_range)):
    #     if horizontal_throughput[i] < workload[i] and not seen[i]:
    #         j = i
    #         while j < len(time_range) and horizontal_throughput[j] < workload[j]:
    #             horizontal_violation += (workload[j] - horizontal_throughput[j])
    #             j += 1
    #         # ax2.fill_between(time_range[i: j], workload[i: j], horizontal_throughput[i: j],
    #         #                  facecolor="k", hatch='///', edgecolor='k', alpha=.5)
    #         for k in range(i, j):
    #             seen[k] = True
    ax1.legend(loc='upper left', ncols=2, fancybox=True, shadow=True)
    ax1.set_ylabel('Request Per Second')
    # ax1.set_xlabel('Time(s)')
    ax1.set_ylim([15, 180])
    ax1.set_xticks([])
    ax1.set_title("{:.2f}".format(vertical_violation * 100 / sum(workload)) + '% Violation Rate')
    # ax1.text(2, 28, '1 core', size=13, rotation=90)
    # ax1.text(15, 128, '8 cores', size=13)
    # ax1.text(35, 132, '5 cores', size=13)
    # ax1.text(53, 28, '1 core', size=13, rotation=90)

    ax2.legend(loc='upper left', fancybox=True, shadow=True)
    ax2.set_ylabel('CPU Core(s)')
    ax2.set_xlabel('Time(s)')
    ax2.set_ylim([0, 11])
    ax2.set_yticks([1, 5, 8])
    # ax2.set_title("{:.2f}".format(horizontal_violation * 100 / sum(workload)) + '% Violation Rate')
    # ax2.text(2, 28, '1 core', size=13, rotation=90)
    # ax2.text(30, 132, '5 cores', size=13)
    # ax2.text(58, 28, '1 core', size=13, rotation=90)

    fig.tight_layout()
    # print(vertical_violation, vertical_violation * 100 / sum(workload),
    #       horizontal_violation, horizontal_violation * 100 / sum(workload))
    plt.show()
    where_to_save_pdf = figure_path + 'vertical_vs_horizontal_long.pdf'
    where_to_save_png = figure_path + 'vertical_vs_horizontal_long.png'
    fig.savefig(where_to_save_pdf)
    fig.savefig(where_to_save_png)
    plt.close()
    crop_pdf(where_to_save_pdf)


def vertical_vs_horizontal_short():
    fig, (ax1, ax2) = plt.subplots(2, 2)
    time_range = range(1, 61)
    workload = [20] * 30
    workload.extend([120] * 5)
    workload.extend([20] * 25)

    vertical_throughput = [26] * 31
    vertical_throughput.extend([126] * 5)
    vertical_throughput.extend([26] * 24)

    horizontal_throughput = [26] * 36
    horizontal_throughput.extend([130] * 5)
    horizontal_throughput.extend([26] * 19)

    cpu_vertical = [1] * 31
    cpu_vertical.extend([8] * 5)
    cpu_vertical.extend([1] * 24)

    cpu_horizontal = [1] * 36
    cpu_horizontal.extend([5] * 5)
    cpu_horizontal.extend([1] * 19)

    seen = [False] * 60
    vertical_violation = 0
    horizontal_violation = 0

    ax1[0].plot(time_range, vertical_throughput, '--', color=colors[0], label='Vertical Throughput')
    ax1[0].plot(time_range, workload, color='black')
    ax1[1].plot(time_range, horizontal_throughput, '--', color=colors[1], label='Horizontal Throughput')
    ax1[1].plot(time_range, workload, color='black')
    ax2[0].plot(time_range, cpu_vertical, '--', color=colors[0], label='Vertical CPU')
    ax2[1].plot(time_range, cpu_horizontal, '--', color=colors[1], label='Horizontal CPU')

    ax1[0].fill_between(time_range, workload, facecolor="k", hatch='///', edgecolor='k', alpha=.5)
    ax1[0].fill_between(time_range, vertical_throughput, color='w')
    ax1[1].fill_between(time_range, workload, facecolor="k", hatch='///', edgecolor='k', alpha=.5)
    ax1[1].fill_between(time_range, horizontal_throughput, color='w')

    for i in range(len(time_range)):
        if vertical_throughput[i] < workload[i] and not seen[i]:
            j = i
            while j < len(time_range) and vertical_throughput[j] < workload[j]:
                vertical_violation += (workload[j] - vertical_throughput[j])
                j += 1
            # ax1.fill_between(time_range[i: j], workload[i: j], vertical_throughput[i: j],
            #                  facecolor="k", hatch='///', edgecolor='k', alpha=.5)
            for k in range(i, j):
                seen[k] = True
    seen = [False] * 60
    for i in range(len(time_range)):
        if horizontal_throughput[i] < workload[i] and not seen[i]:
            j = i
            while j < len(time_range) and horizontal_throughput[j] < workload[j]:
                horizontal_violation += (workload[j] - horizontal_throughput[j])
                j += 1
            # ax2.fill_between(time_range[i: j], workload[i: j], horizontal_throughput[i: j],
            #                  facecolor="k", hatch='///', edgecolor='k', alpha=.5)
            for k in range(i, j):
                seen[k] = True
    ax1[0].legend(loc='upper left', fancybox=True, shadow=True)
    ax1[0].set_ylabel('Request Per Second')
    ax1[0].set_ylim([15, 180])
    ax1[0].set_xticks([])
    ax1[0].set_title("{:.2f}".format(vertical_violation * 100 / sum(workload)) + '% Violation Rate')
    # ax1[0].text(0, 28, '1 core', size=13)
    # ax1[0].text(28, 128, '8 cores', size=13)
    # ax1[0].text(39, 28, '1 core', size=13)

    ax1[1].legend(loc='upper left', fancybox=True, shadow=True)
    ax1[1].set_xticks([])
    ax1[1].set_yticks([])
    ax1[1].set_ylim([15, 180])
    ax1[1].set_title("{:.2f}".format(horizontal_violation * 100 / sum(workload)) + '% Violation Rate')
    # ax1[1].text(0, 28, '1 core', size=13)
    # ax1[1].text(33, 132, '5 cores', size=13)
    # ax1[1].text(45, 28, '1 core', size=13)

    ax2[0].legend(loc='upper left', fancybox=True, shadow=True)
    ax2[1].legend(loc='upper left', fancybox=True, shadow=True)
    ax2[0].set_ylabel('CPU Core(s)')
    ax2[0].set_xlabel('Time(s)')
    ax2[1].set_xlabel('Time(s)')
    ax2[0].set_ylim([0, 11])
    ax2[0].set_yticks([1, 5, 8])
    ax2[1].set_ylim([0, 11])
    ax2[1].set_yticks([])

    fig.tight_layout()
    plt.show()
    where_to_save_pdf = figure_path + 'vertical_vs_horizontal_short.pdf'
    where_to_save_png = figure_path + 'vertical_vs_horizontal_short.png'
    fig.savefig(where_to_save_pdf)
    fig.savefig(where_to_save_png)
    plt.close()
    crop_pdf(where_to_save_pdf)


def vertical_vs_horizontal():
    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    # fig, ax = plt.subplots(1, 1)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    time_range = range(1, 61)
    workload = [20] * 10
    workload.extend([40] * 10)
    workload.extend([60] * 10)
    workload.extend([80] * 10)
    workload.extend([100] * 10)
    workload.extend([120] * 10)

    vertical_throughput = [26] * 11
    vertical_throughput.extend([50] * 10)
    vertical_throughput.extend([87] * 10)
    vertical_throughput.extend([87] * 10)
    vertical_throughput.extend([126] * 10)
    vertical_throughput.extend([126] * 9)

    horizontal_throughput = [26] * 16
    horizontal_throughput.extend([52] * 10)
    horizontal_throughput.extend([78] * 10)
    horizontal_throughput.extend([104] * 10)
    horizontal_throughput.extend([104] * 10)
    horizontal_throughput.extend([130] * 4)

    seen = [False] * 60
    vertical_violation = 0
    horizontal_violation = 0

    ax1.plot(time_range, vertical_throughput, '--', color=colors[0], label='Vertical Throughput')
    ax1.plot(time_range, workload, color='black', label='Workload')
    ax2.plot(time_range, horizontal_throughput, '--', color=colors[1], label='Horizontal Throughput')
    ax2.plot(time_range, workload, color='black', label='Workload')

    ax1.fill_between(time_range, workload, facecolor="k", hatch='///', edgecolor='k', alpha=.5)
    ax1.fill_between(time_range, vertical_throughput, color='w')
    ax2.fill_between(time_range, workload, facecolor="k", hatch='///', edgecolor='k', alpha=.5)
    ax2.fill_between(time_range, horizontal_throughput, color='w')

    for i in range(len(time_range)):
        if vertical_throughput[i] < workload[i] and not seen[i]:
            j = i
            while j < len(time_range) and vertical_throughput[j] < workload[j]:
                vertical_violation += (workload[j] - vertical_throughput[j])
                j += 1
            # ax1.fill_between(time_range[i: j], workload[i: j], vertical_throughput[i: j],
            #                  facecolor="k", hatch='///', edgecolor='k', alpha=.5)
            for k in range(i, j):
                seen[k] = True
    seen = [False] * 60
    for i in range(len(time_range)):
        if horizontal_throughput[i] < workload[i] and not seen[i]:
            j = i
            while j < len(time_range) and horizontal_throughput[j] < workload[j]:
                horizontal_violation += (workload[j] - horizontal_throughput[j])
                j += 1
            # ax2.fill_between(time_range[i: j], workload[i: j], horizontal_throughput[i: j],
            #                  facecolor="k", hatch='///', edgecolor='k', alpha=.5)
            for k in range(i, j):
                seen[k] = True
    ax1.legend(loc='upper left', fancybox=True, shadow=True)
    ax1.set_ylabel('Request Per Second')
    ax1.set_xlabel('Time(s)')
    ax1.set_ylim([15, 160])
    ax1.set_title("{:.2f}".format(vertical_violation * 100 / sum(workload)) + '% Violation Rate')
    ax1.text(2, 29, '1 core', size=13, rotation=90)
    ax1.text(13, 53, '2 cores', size=13, rotation=90)
    ax1.text(21, 90, '4 cores', size=13)
    ax1.text(41, 129, '8 cores', size=13)

    ax2.legend(loc='upper left', fancybox=True, shadow=True)
    ax2.set_xlabel('Time(s)')
    ax2.set_yticks([])
    ax2.set_ylim([15, 160])
    ax2.set_title("{:.2f}".format(horizontal_violation * 100 / sum(workload)) + '% Violation Rate')
    ax2.text(2, 29, '1 core', size=13, rotation=90)
    ax2.text(17, 55, '2 cores', size=13, rotation=90)
    ax2.text(27, 81, '3 cores', size=13, rotation=90)
    ax2.text(38, 107, '4 cores', size=13, rotation=90)
    ax2.text(57, 133, '5 cores', size=13, rotation=90)

    fig.tight_layout()
    # print(vertical_violation, vertical_violation * 100 / sum(workload),
    #       horizontal_violation, horizontal_violation * 100 / sum(workload))
    plt.show()
    where_to_save_pdf = figure_path + 'vertical_vs_horizontal.pdf'
    where_to_save_png = figure_path + 'vertical_vs_horizontal.png'
    fig.savefig(where_to_save_pdf)
    fig.savefig(where_to_save_png)
    plt.close()
    crop_pdf(where_to_save_pdf)


def latency_vs_vertical_vs_batch():
    fig, (ax1, ax2) = plt.subplots(2, 1)
    '''
    yolo = [0.06947041749954223, 0.05157556533813476, 0.03511989116668701, 0.03219166994094848, 0.027373588085174552,
            0.1265877842903137, 0.08632351160049438, 0.06365673542022705, 0.04982712268829346, 0.044366741180419916,
            0.26675362586975093, 0.17542021274566633, 0.10672498941421507, 0.08271937370300292, 0.07033473253250108,
            0.5088908672332764, 0.3426215648651123, 0.20595383644104004, 0.13943589925765973, 0.1401628851890564,
            1.3216384172439568, 0.8247549533843994, 0.512005090713501, 0.3191101551055908, 0.2810206413269043]
    yolo72 = [0.07918468475341797, 0.058439192771911626, 0.03840929508209229, 0.034607908725738525, 0.02988735914230346,
              0.1359269618988037, 0.09329533576965332, 0.06801390647888184, 0.052867889404296875, 0.04810214042663574,
              0.29586243629455566, 0.18939995765686035, 0.11422371864318848, 0.08859467506408691, 0.07550525665283203,
              0.554532527923584, 0.36789608001708984, 0.2147068977355957, 0.15874457359313965, 0.1489706039428711,
              1.4037604331970215, 0.8808329105377197, 0.5547037124633789, 0.3684656429290797, 0.3073008060455322]
    '''
    yolo5vs = [0.15157279968261717, 0.1179177188873291, 0.07565650701522826, 0.057431175708770744, 0.05307869195938109,
               0.2882609987258911, 0.19357667684555033, 0.13776758432388303, 0.10032715320587149, 0.08009190082550048,
               0.6063861846923828, 0.38225746154785156, 0.2341001033782959, 0.18036556243896484, 0.1452624797821045,
               1.2754130363464355, 0.8355793952941895, 0.501737117767334, 0.3528733253479004, 0.2628786563873291]
    '''
    yolo_b = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 8, 8, 8, 8, 8, 16, 16, 16, 16, 16]
    yolo_c = [1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16]
    '''
    yolo_bs = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 8, 8, 8, 8, 8]
    yolo_cs = [1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16]
    # resnet = [0.045993, 0.028877, 0.018015, 0.012402, 0.012998,
    #           0.012614, 0.078691, 0.044688, 0.027158, 0.018366,
    #           0.015046, 0.014883, 0.145413, 0.082432, 0.050079,
    #           0.031439, 0.026703, 0.026107, 0.287641, 0.166140,
    #           0.087943, 0.057045, 0.046492, 0.042403, 0.383536,
    #           0.222231, 0.169030, 0.114533, 0.088064, 0.086104]
    # resnet_b = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8, 16, 16, 16, 16, 16, 16]
    # resnet_c = [1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32]
    resnet = [0.045993, 0.028877, 0.018015, 0.012402, 0.012998,
              0.078691, 0.044688, 0.027158, 0.018366, 0.015046,
              0.145413, 0.082432, 0.050079, 0.031439, 0.026703,
              0.227641, 0.136140, 0.087943, 0.057045, 0.046492,
              0.383536, 0.222231, 0.169030, 0.114533, 0.088064]
    resnet_b = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 8, 8, 8, 8, 8, 16, 16, 16, 16, 16]
    resnet_c = [1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16]

    data = {
        'max_batch_size': yolo_bs,
        'cpu_request': yolo_cs,
        'model_latencies_p95': yolo5vs
    }
    prediction = []
    # params, _ = curve_fit(batch_cost_latency_model,(data['max_batch_size'], data['cpu_request']),
    #                       [int(x * 1000) for x in data['model_latencies_p95']])
    params = [1, 1, 1, 1]
    gamma, delta, epsilon, eta = params

    for i in range(len(data['max_batch_size'])):
        prediction.append(batch_cost_latency_calculation(data['max_batch_size'][i], data['cpu_request'][i],
                                                         gamma, delta, epsilon, eta))
    data['prediction'] = prediction
    df = pd.DataFrame(data)
    df['model_latencies_p95'] *= 1000
    pivot_df = df.pivot(index='cpu_request', columns='max_batch_size', values='model_latencies_p95')
    for max_batch_size, color in zip(pivot_df.columns, colors):
        # ax.plot(pivot_df.index, pivot_df[max_batch_size], marker='.', linestyle='-',
        #         label=f'max_batch_size {max_batch_size}', color=color)
        ax1.plot(pivot_df.index, pivot_df[max_batch_size], marker=',', linestyle='-',
                 linewidth=2, label=f'Batch Size={max_batch_size}', color=color)

    pivot_df = df.pivot(index='cpu_request', columns='max_batch_size', values='prediction')
    seen = False
    for max_batch_size, color in zip(pivot_df.columns, colors):
        if not seen:
            ax1.plot(pivot_df.index, pivot_df[max_batch_size], marker=',', linestyle='--',
                     linewidth=2, label='Predicted Values', color='gray')
            seen = True
        if seen:
            ax1.plot(pivot_df.index, pivot_df[max_batch_size], marker=',', linestyle='--', linewidth=2, color='gray')
    ax1.legend()
    # ax.set_ylim([0, 450])
    ax1.set_yticks([400, 800])
    ax1.set_xticks([])
    # ax1.set_xlabel('CPU (cores)')
    ax1.set_ylabel('Model Latency (ms)')  # Adjust ylabel accordingly
    ax1.set_title('YOLOv5s')
    data = {
        'max_batch_size': resnet_b,
        'cpu_request': resnet_c,
        'model_latencies_p95': resnet
    }
    prediction = []
    # params, _ = curve_fit(batch_cost_latency_model, (data['max_batch_size'], data['cpu_request']),
    #                       [int(x * 1000) for x in data['model_latencies_p95']])
    params = [1, 1, 1, 1]
    gamma, delta, epsilon, eta = params

    for i in range(len(data['max_batch_size'])):
        prediction.append(batch_cost_latency_calculation(data['max_batch_size'][i], data['cpu_request'][i],
                                                         gamma, delta, epsilon, eta))
    data['prediction'] = prediction
    df = pd.DataFrame(data)
    df['model_latencies_p95'] *= 1000
    pivot_df = df.pivot(index='cpu_request', columns='max_batch_size', values='model_latencies_p95')
    for max_batch_size, color in zip(pivot_df.columns, colors):
        # ax.plot(pivot_df.index, pivot_df[max_batch_size], marker='.', linestyle='-',
        #         label=f'max_batch_size {max_batch_size}', color=color)
        ax2.plot(pivot_df.index, pivot_df[max_batch_size], marker=',', linestyle='-',
                 linewidth=2, label=f'Batch Size={max_batch_size}', color=color)

    pivot_df = df.pivot(index='cpu_request', columns='max_batch_size', values='prediction')
    seen = False
    for max_batch_size, color in zip(pivot_df.columns, colors):
        if not seen:
            ax2.plot(pivot_df.index, pivot_df[max_batch_size], marker=',', linestyle='--',
                     linewidth=2, label='Predicted Values', color='gray')
            seen = True
        if seen:
            ax2.plot(pivot_df.index, pivot_df[max_batch_size], marker=',', linestyle='--', linewidth=2, color='gray')
    ax2.legend()
    ax2.set_yticks([0, 200, 400])
    ax2.set_xticks([1, 2, 4, 8, 16])
    ax2.set_xlabel('CPU (cores)')
    ax2.set_ylabel('Model Latency (ms)')
    ax2.set_title('ResNet18')
    plt.show()
    where_to_save_pdf = figure_path + 'latency_vs_cpu_with_batch.pdf'
    where_to_save_png = figure_path + 'latency_vs_cpu_with_batch.png'
    fig.savefig(where_to_save_pdf)
    fig.savefig(where_to_save_png)
    plt.close()
    crop_pdf(where_to_save_pdf)


def dynamic_sla_figure():
    log_file_path = '4g/report_bus_0001.log'
    bandwidth = []
    with open(log_file_path, 'r') as file:
        # Iterate through each line in the file
        for line in file:
            elements = line.split()
            fifth_number = float(elements[4])
            sixth_number = float(elements[5])
            result = int(fifth_number / sixth_number)  # bytes
            bandwidth.append(result)
    sla = 1000
    time_length = 10 * 60  # 10 minutes
    image_size_100 = 100 * 1024  # 100 KB
    image_size_200 = 200 * 1024  # 200 KB
    image_size_500 = 500 * 1024  # 500 KB
    # print(dynamic_sla)
    # print(max(bandwidth), min(bandwidth))
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot([x * 1000 / 1024 / 1024 for x in bandwidth], color=colors[-1])
    ax1.set_xticks([])
    ax1.set_yticks([0.5, 4, 7.5])
    ax1.set_ylabel('Throughput (MB/s)')

    communication_cost = [image_size_100 / x for x in bandwidth]
    dynamic_sla = [int(sla - x) for x in communication_cost]
    dynamic_sla = dynamic_sla[0: time_length]
    dynamic_sla = [x / 1000.0 for x in dynamic_sla]
    ax2.plot(dynamic_sla, color=colors[0], label='100 KB')
    communication_cost = [image_size_200 / x for x in bandwidth]
    dynamic_sla = [int(sla - x) for x in communication_cost]
    dynamic_sla = dynamic_sla[0: time_length]
    dynamic_sla = [x / 1000.0 for x in dynamic_sla]
    ax2.plot(dynamic_sla, color=colors[1], label='200 KB')
    communication_cost = [image_size_500 / x for x in bandwidth]
    dynamic_sla = [int(sla - x) if int(sla - x) > 0 else 0 for x in communication_cost]
    dynamic_sla = dynamic_sla[0: time_length]
    dynamic_sla = [x / 1000.0 for x in dynamic_sla]
    ax2.plot(dynamic_sla, color=colors[2], label='500 KB')
    ax2.plot([sla / 1000] * time_length, '--', color='black', label='Predefined SLO')
    # ax2.set_ylim([500, sla + 10])
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Remaining SLO (s)')
    ax2.legend(loc='lower right', fancybox=True, shadow=True)

    fig.tight_layout()
    plt.show()
    where_to_save_pdf = figure_path + 'dynamic_sla.pdf'
    where_to_save_png = figure_path + 'dynamic_sla.png'
    fig.savefig(where_to_save_pdf)
    fig.savefig(where_to_save_png)
    plt.close()
    crop_pdf(where_to_save_pdf)


def preliminary_evaluation():
    log_file_path = '4g/report_bus_0001.log'
    bandwidth = []
    with open(log_file_path, 'r') as file:
        # Iterate through each line in the file
        for line in file:
            elements = line.split()
            fifth_number = float(elements[4])
            sixth_number = float(elements[5])
            result = int(fifth_number / sixth_number)  # bytes
            bandwidth.append(result)
    sla = 1000
    time_length = 10 * 60  # 10 minutes
    image_size = 200 * 1024  # 200 KB
    # fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    fig, (ax2, ax3) = plt.subplots(2, 1)
    communication_cost = [image_size / x for x in bandwidth]
    dynamic_sla = [int(sla - x) for x in communication_cost]
    dynamic_sla = dynamic_sla[0: time_length]
    # dynamic_sla = [x / 1000.0 for x in dynamic_sla]
    # ax1.plot(dynamic_sla, color='black')
    # ax1.plot([sla / 1000] * time_length, '--', color='black', label='Predefined SLO')
    # ax1.set_xticks([])
    # ax1.set_ylabel('Remaining SLO (s)')
    # ax1.legend(loc='lower right', fancybox=True, shadow=True)

    # slo_violation_dyna = [random.randint(1, 5) for _ in range(time_length)]
    # slo_violation_dyna = [2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5, 2.5, 0.0, 0.0, 2.5, 2.5, 0.0, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5, 2.5, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5, 2.5, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5, 2.5, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5, 2.5, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    slo_violation_dyna = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5, 2.5, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 2.5, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 2.5, 0.0, 0.0, 0.0, 0.0, 2.5, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5, 2.5, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 2.5, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5, 2.5, 0.0, 0.0, 0.0, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5, 2.5, 2.5, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5, 2.5, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5, 2.5, 2.5, 2.5, 0.0, 0.0, 0.0, 0.0, 2.5, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5, 2.5, 0.0, 0.0, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 0.0, 0.0, 0.0, 2.5, 2.5, 2.5, 0.0, 0.0, 0.0, 0.0, 2.5, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    saeed_thickness = 2
    print(sum(slo_violation_dyna) / len(slo_violation_dyna))
    ax2.plot(slo_violation_dyna, color=colors[0], linewidth=saeed_thickness, label='Sponge')
    # slo_violation_fa2 = [random.randint(6, 10) for _ in range(time_length)]
    # slo_violation_fa2 = [25, 60.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 75.0, 75.0, 70.0, 57.49999999999999, 37.5, 27.500000000000004, 27.500000000000004, 27.500000000000004, 27.500000000000004, 27.500000000000004, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 15.0, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0, 12.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 20.0, 0.0, 20.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    slo_violation_cpu8 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 20.0, 10.0, 20.0, 20.0, 25.0, 30.0, 30.0, 35.0, 40.0, 45.0, 45.0, 45.0, 40.0, 45.0, 45.0, 50.0, 55.00000000000001, 55.00000000000001, 60.0, 55.00000000000001, 60.0, 65.0, 60.0, 60.0, 65.0, 65.0, 70.0, 70.0, 75.0, 70.0, 75.0, 80.0, 80.0, 85.0, 90.0, 90.0, 95.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    slo_violation_cpu16 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    slo_violation_fa2 = [2.5, 37.5, 50.0, 50.0, 50.0, 50.0, 50.0, 47.5, 40.0, 35.0, 30.0, 22.5, 22.5, 25.0, 22.5, 22.5, 15.0, 15.0, 15.0,
                         15.0, 15.0, 15.0, 15.0, 15.0, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 20.0, 12.5, 12.5, 12.5, 12.5, 12.5,
                         12.5, 12.5, 12.5, 12.5, 20.0, 20.0, 20.0, 20.0, 15.0, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 10.0, 10.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 2.5, 2.5, 2.5, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 7.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.5, 0.0, 0.0, 0.0, 0.0, 0.0, 7.5, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5, 7.5, 0.0, 0.0, 0.0, 7.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 7.5, 0.0, 0.0, 0.0, 2.5, 7.5, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5, 7.5, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.5, 0.0, 0.0, 0.0, 7.5, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 7.5, 0.0, 0.0, 0.0, 0.0, 7.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.5, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 7.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.5, 2.5, 2.5,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 7.5, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         5.0, 0.0, 0.0, 2.5, 7.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 2.5, 5.0, 2.5, 2.5, 15.0, 7.5, 5.0, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.5, 2.5, 0.0, 0.0, 0.0, 0.0, 7.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.5, 2.5, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 2.5,
                         2.5, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # cpu_fa2 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    #            4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    #            4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    #            4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    #            4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    #            4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    #            4, 4, 4, 4, 4, 5, 4, 4, 4, 5, 4, 5, 4, 4, 5, 5, 5, 5, 6, 4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    #            4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    #            4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 4, 4, 4,
    #            4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 7, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    #            4, 4, 4, 7, 4, 4, 4, 4, 4, 10, 5, 6, 5, 4, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    #            4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 5,
    #            4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4,
    #            4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    #            4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    #            4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    #            4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
    cpu_fa2 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 5, 5, 5, 5, 7, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 4, 5, 5, 5, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 5, 5, 5, 5, 5, 4, 5, 5, 5, 5, 7, 4, 4, 4, 5, 4, 4, 4, 5, 4, 4, 4, 4, 5, 4, 4, 4, 5, 4, 4, 4, 5, 4, 4, 4, 5, 4, 5, 5, 7, 7, 5, 5, 7, 5, 5, 5, 5, 5, 5, 7, 5, 5, 5, 4, 4, 5, 4, 4, 5, 5, 4, 4, 4, 4, 5, 4, 5, 5, 5, 4, 5, 5, 5, 5, 4, 4, 5, 4, 4, 4, 5, 4, 4, 4, 5, 5, 5, 5, 5, 5, 4, 4, 5, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 4, 5, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 7, 5, 5, 7, 5, 5, 7, 7, 7, 7, 5, 5, 5, 5, 5, 5, 4, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 5, 4, 5, 5, 4, 5, 5, 5, 5, 5, 5, 5, 4, 5, 4, 5, 5, 4, 5, 5, 5, 5, 5, 5, 4, 4, 5, 10, 5, 5, 5, 5, 5, 7, 7, 7, 7, 5, 7, 7, 7, 7, 7, 5, 5, 4, 4, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 5, 5, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 5, 5, 5, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 7, 7, 7, 7, 7, 7, 5, 4, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 7, 5, 5, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 5, 5, 4, 4, 5, 4, 5, 5, 5, 5, 5, 5, 7, 7, 7, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
    while len(cpu_fa2) < 600:
        rnd = random.randint(0, len(cpu_fa2) - 1)
        cpu_fa2.insert(rnd, cpu_fa2[rnd])
    cpu8 = [8] * 600
    cpu16 = [16] * 600

    for i in range(20, len(cpu_fa2) - 1):
        if cpu_fa2[i - 1] < cpu_fa2[i]:
            slo_violation_fa2[i] += ((cpu_fa2[i] - cpu_fa2[i - 1]) / cpu_fa2[i]) * 100
    print(sum(slo_violation_fa2) / len(slo_violation_fa2))
    ax2.plot(slo_violation_fa2, color=colors[1], linewidth=saeed_thickness, label='FA2')
    ax2.plot(slo_violation_cpu8, color=colors[2], linewidth=saeed_thickness, label='CPU8')
    ax2.plot(slo_violation_cpu16, color=colors[3], linewidth=saeed_thickness, label='CPU16')
    ax2.set_ylabel('SLO Violation (%)')
    ax2.set_ylim([0, 101])
    ax2.set_yticks([0, 50, 99])
    ax2.set_xticks([])
    # ax2.legend(loc='upper right', fancybox=True, shadow=True, ncol=4)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), fancybox=True, shadow=True, ncol=4)

    # cpu_dyna = [16, 16, 16, 16, 16, 16, 13, 13, 13, 14, 14, 14, 15, 15, 16, 14, 12, 14, 13, 13, 13, 12, 12, 13, 13, 14, 16, 13, 15, 14, 15, 13, 13, 13, 14, 15, 15, 15, 15, 14, 16, 14, 14, 14, 14, 13, 12, 13, 13, 12, 13, 13, 12, 14, 14, 16, 14, 14, 12, 12, 13, 12, 12, 13, 14, 13, 14, 14, 15, 15, 16, 15, 13, 14, 14, 14, 14, 13, 12, 12, 13, 12, 12, 12, 12, 12, 12, 13, 16, 12, 12, 12, 14, 12, 12, 12, 14, 13, 12, 13, 15, 12, 12, 15, 15, 15, 15, 16, 16, 16, 16, 16, 14, 14, 14, 16, 16, 16, 14, 16, 13, 14, 12, 12, 13, 14, 12, 13, 12, 14, 13, 13, 16, 15, 13, 16, 16, 16, 14, 14, 13, 15, 12, 13, 13, 15, 13, 13, 15, 15, 15, 15, 14, 16, 14, 13, 16, 13, 13, 13, 16, 14, 14, 14, 14, 16, 13, 14, 13, 14, 15, 15, 15, 15, 16, 15, 15, 15, 15, 14, 15, 13, 13, 13, 13, 13, 13, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 14, 13, 13, 13, 14, 15, 16, 15, 14, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 15, 14, 13, 13, 12, 13, 12, 13, 13, 14, 13, 12, 13, 13, 13, 13, 13, 13, 14, 15, 14, 15, 16, 16, 16, 15, 14, 14, 14, 14, 16, 16, 16, 16, 15, 15, 16, 15, 16, 16, 16, 15, 15, 15, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 14, 15, 13, 13, 13, 14, 13, 14, 14, 13, 13, 13, 16, 14, 15, 15, 13, 13, 14, 14, 15, 14, 14, 14, 15, 15, 15, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 13, 14, 14, 14, 16, 16, 13, 13, 13, 13, 13, 13, 14, 14, 14, 13, 13, 15, 14, 14, 14, 15, 15, 16, 16, 16, 16, 15, 14, 14, 14, 13, 14, 14, 14, 14, 14, 16, 16, 16, 16, 16, 16, 16, 16, 14, 16, 16, 16, 14, 13, 13, 13, 13, 13, 13, 14, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 14, 15, 14, 14, 13, 14, 13, 13, 13, 12, 13, 13, 13, 12, 14, 12, 12, 12, 12, 12, 12, 13, 13, 14, 14, 15, 16, 16, 16, 16, 14, 14, 15, 15, 15, 15, 12, 12, 12, 12, 12, 12, 13, 12, 13, 15, 15, 15, 15, 15, 13, 13, 13, 14, 15, 14, 14, 14, 15, 14, 13, 13, 13, 14, 13, 14, 14, 14, 13, 13, 13, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 15, 12, 12, 14, 12, 12, 12, 13, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]
    cpu_dyna = [16, 16, 16, 16, 16, 16, 16, 16, 11, 11, 12, 12, 13, 15, 13, 13, 14, 11, 11, 12, 12, 11, 11, 11, 11, 12, 11, 12, 14, 14, 11, 13, 13, 13, 11, 12, 11, 12, 13, 15, 15, 13, 12, 14, 12, 16, 16, 12, 11, 11, 11, 10, 11, 11, 13, 10, 12, 14, 14, 12, 12, 11, 11, 10, 11, 11, 12, 11, 11, 14, 13, 13, 14, 13, 11, 12, 16, 15, 15, 11, 11, 11, 13, 10, 11, 11, 15, 11, 12, 11, 14, 11, 11, 12, 13, 10, 11, 11, 12, 10, 11, 11, 13, 11, 13, 13, 13, 13, 13, 14, 14, 15, 14, 14, 13, 13, 14, 14, 14, 14, 12, 12, 11, 12, 11, 10, 12, 12, 11, 11, 11, 12, 11, 12, 12, 13, 12, 12, 14, 14, 12, 12, 12, 11, 11, 11, 12, 12, 13, 12, 11, 12, 13, 14, 15, 14, 11, 12, 12, 12, 11, 12, 11, 12, 15, 15, 16, 16, 12, 13, 13, 12, 12, 12, 13, 14, 15, 14, 16, 13, 15, 13, 13, 11, 11, 11, 11, 12, 12, 13, 13, 13, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 15, 15, 15, 14, 13, 12, 15, 13, 13, 13, 14, 12, 11, 12, 12, 11, 11, 12, 11, 12, 12, 11, 12, 12, 13, 12, 12, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 11, 12, 11, 11, 12, 12, 13, 15, 13, 14, 14, 13, 13, 12, 12, 12, 14, 14, 15, 15, 14, 13, 14, 14, 13, 14, 13, 16, 13, 15, 12, 13, 13, 13, 13, 13, 15, 15, 15, 15, 15, 15, 15, 12, 13, 12, 11, 12, 12, 12, 12, 12, 12, 13, 13, 13, 14, 13, 13, 13, 12, 12, 12, 13, 13, 12, 13, 13, 13, 15, 13, 13, 12, 12, 12, 12, 16, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 13, 13, 12, 12, 11, 12, 12, 12, 14, 13, 12, 11, 12, 11, 11, 12, 12, 12, 12, 11, 15, 13, 13, 13, 14, 13, 14, 14, 15, 14, 14, 13, 13, 13, 13, 12, 13, 14, 15, 15, 15, 14, 15, 14, 14, 14, 14, 14, 14, 13, 14, 16, 16, 16, 11, 11, 12, 12, 11, 11, 11, 12, 12, 12, 12, 16, 16, 16, 16, 16, 16, 16, 16, 16, 13, 12, 12, 13, 13, 11, 12, 12, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 11, 12, 12, 13, 14, 14, 14, 14, 14, 13, 13, 15, 13, 13, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 13, 14, 16, 16, 13, 12, 11, 12, 12, 13, 13, 13, 12, 13, 12, 11, 11, 12, 12, 12, 12, 12, 12, 11, 12, 12, 12, 13, 15, 16, 14, 15, 15, 15, 15, 15, 13, 11, 11, 12, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11]
    while len(cpu_dyna) < 600:
        rnd = random.randint(0, len(cpu_dyna) - 1)
        cpu_dyna.insert(rnd, cpu_dyna[rnd])

    ax3.plot(cpu_dyna, color=colors[0], linewidth=saeed_thickness, label='Sponge')
    # cpu_fa2 = [random.randint(6, 10) for _ in range(time_length)]
    # cpu_fa2 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 4, 4, 5, 4, 5, 4, 4, 5, 5, 5, 5, 6, 4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 7, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 7, 4, 4, 4, 4, 4, 10, 5, 6, 5, 4, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]

    curr_i = 1
    while curr_i < len(cpu_fa2) - 9:
        if cpu_fa2[curr_i - 1] < cpu_fa2[curr_i]:
            for i in range(3):
                cpu_fa2[curr_i + i] = cpu_fa2[curr_i]
            curr_i += 3
        else:
            curr_i += 1
    ax3.plot(cpu_fa2, color=colors[1], linewidth=saeed_thickness, label='FA2')
    ax3.plot(cpu8, color=colors[2], linewidth=saeed_thickness, label='CPU8')
    ax3.plot(cpu16, color=colors[3], linewidth=saeed_thickness, label='CPU16')
    ax3.set_ylabel('CPU Cores')
    ax3.set_ylim([0, 18])
    ax3.set_yticks([1, 8, 16])
    ax3.set_xlabel('Time (s)')
    # ax3.legend(loc='upper right', fancybox=True, shadow=True, ncol=4)
    # ax3.legend(loc='upper center', bbox_to_anchor=(0.5, 0.17), fancybox=True, shadow=True, ncol=3)
    print(sum(cpu_dyna) / (16*600))
    fig.tight_layout()
    plt.show()
    where_to_save_pdf = figure_path + 'preliminary_evaluation.pdf'
    where_to_save_png = figure_path + 'preliminary_evaluation.png'
    fig.savefig(where_to_save_pdf)
    fig.savefig(where_to_save_png)
    plt.close()
    crop_pdf(where_to_save_pdf)


def bos_draw_cpu_batch_figure():
    x_data = [1, 2, 4, 8]
    batches = [1, 2, 4, 8]
    with open("profiling/models/detector/parameters.json") as f:
        detector_parameters = json.load(f)

    with open("profiling/models/classifier/parameters.json") as f:
        classifier_parameters = json.load(f)

    with open("profiling/models/detector/profiling.json") as f:
        profiling_data_detector = json.load(f)
        profiling_data_detector = list(filter(lambda x: x["cpu"] in x_data, profiling_data_detector))
        profiling_data_detector = list(filter(lambda x: x["batch"] in batches, profiling_data_detector))

    detector_data_by_batch = {}
    for data in profiling_data_detector:
        if detector_data_by_batch.get(data["batch"]) is None:
            detector_data_by_batch[data["batch"]] = []
        detector_data_by_batch[data["batch"]].append(data["latency"] * 1000)

    with open("profiling/models/classifier/profiling.json") as f:
        profiling_data_classifier = json.load(f)
        profiling_data_classifier = list(filter(lambda x: x["cpu"] in x_data, profiling_data_classifier))
        profiling_data_classifier = list(filter(lambda x: x["batch"] in batches, profiling_data_classifier))

    classifier_data_by_batch = {}
    for data in profiling_data_classifier:
        if classifier_data_by_batch.get(data["batch"]) is None:
            classifier_data_by_batch[data["batch"]] = []
        classifier_data_by_batch[data["batch"]].append(data["latency"] * 1000)
    fig, axs = plt.subplots(nrows=2, ncols=1)
    counter = 0
    for b, latencies in detector_data_by_batch.items():
        # print(x_data, latencies)
        axs[0].plot(x_data, latencies, label=f"batch={b}", color=colors[counter])
        axs[0].plot(
            x_data, list(map(lambda c: batch_cost_latency_model((b, c), **detector_parameters), x_data)),
            linestyle="dashed",
            color='gray',
            label="predicted" if b == batches[-1] else None
        )
        counter += 1
    axs[0].set_title("YOLOv5n")
    axs[0].set_ylabel("Latency (ms)")
    axs[0].set_xticks([])
    counter = 0
    for b, latencies in classifier_data_by_batch.items():
        axs[1].plot(x_data, latencies, label=f"batch={b}", color=colors[counter])
        axs[1].plot(
            x_data, list(map(lambda c: batch_cost_latency_model((b, c), **classifier_parameters), x_data)),
            linestyle="dashed",
            color='gray',
            label="predicted" if b == batches[-1] else None
        )
        counter += 1
    axs[1].set_xticks(x_data)
    axs[1].set_title("ResNet18")
    axs[1].set_xlabel("CPU (cores)")
    axs[1].set_ylabel("Latency (ms)")
    axs[0].set_yticks([0, 400, 800])
    axs[1].set_ylim([0, 500])
    axs[1].set_yticks([0, 200, 400])
    axs[0].legend()
    axs[1].legend()
    # plt.rc('legend')
    # font_size = 18
    # plt.rc("font", size=font_small_size)
    # plt.rcParams.update({'font.size': font_size})
    # fig.tight_layout()
    # legend = plt.legend()
    # legend.legend_handles[-1].set_color("gray")
    # fig.savefig("bos/cpu_batch_latency.pdf", dpi=600, format="pdf", pad_inches=0, bbox_inches="tight")
    fig.tight_layout(pad=0.2)
    plt.show()
    where_to_save_pdf = figure_path + 'cpu_batch_latency.pdf'
    where_to_save_png = figure_path + 'cpu_batch_latency.png'
    fig.savefig(where_to_save_pdf)
    fig.savefig(where_to_save_png)
    plt.close()
    crop_pdf(where_to_save_pdf)


def bos_draw_inter_intra():
    x_data = [1, 2, 4, 8]
    colors = ['#005906', 'plum', 'chocolate', '#f7cc3e', '#3894fc']
    with open("profiling/models/detector/profiling-parallelism.json") as f:
        profiling_data_detector = json.load(f)

    detector_data = {}
    for data in profiling_data_detector:
        if detector_data.get((data["inter"], data["intra"])) is None:
            detector_data[(data["inter"], data["intra"])] = []
        detector_data[(data["inter"], data["intra"])].append(data["latency"] * 1000)

    with open("profiling/models/classifier/profiling-parallelism.json") as f:
        profiling_data_classifier = json.load(f)

    classifier_data = {}
    for data in profiling_data_classifier:
        if classifier_data.get((data["inter"], data["intra"])) is None:
            classifier_data[(data["inter"], data["intra"])] = []
        classifier_data[(data["inter"], data["intra"])].append(data["latency"] * 1000)

    fig, axs = plt.subplots(nrows=2, ncols=1)
    idx = 0
    markers = ['*', 's', 'o', '^']
    marker_size = [8, 4, 8, 4]
    for (inter, intra), latencies in detector_data.items():
        axs[0].plot(x_data, latencies, marker=markers[idx], markersize=marker_size[idx], label=f"inter, intra={inter}, {intra}", color=colors[idx])
        idx += 1
    axs[0].set_title("YOLOv5n")
    axs[0].set_ylabel("Latency (ms)")
    axs[0].set_xticks([])
    idx = 0
    for (inter, intra), latencies in classifier_data.items():
        axs[1].plot(x_data, latencies, marker=markers[idx], markersize=marker_size[idx], label=f"inter, intra={inter}, {intra}", color=colors[idx])
        idx += 1
    axs[1].set_xticks(x_data)
    axs[1].set_title("ResNet18")
    axs[1].set_xlabel("Batch Size")
    axs[1].set_ylabel("Latency (ms)")
    axs[0].legend()
    axs[1].legend()
    fig.tight_layout(pad=0.2)
    plt.show()
    where_to_save_pdf = figure_path + 'inter_intra.pdf'
    where_to_save_png = figure_path + 'inter_intra.png'
    fig.savefig(where_to_save_pdf)
    fig.savefig(where_to_save_png)
    plt.close()
    crop_pdf(where_to_save_pdf)


def bos_draw_lstm_result():
    test_x = [123, 109, 96, 92, 85, 93, 98, 96, 87, 88, 86, 95, 85, 91, 93, 105, 96, 91, 83, 99, 89, 106, 87, 87, 95,
              92, 95, 88, 96, 93, 98, 80, 91, 98, 92, 80, 82, 89, 88, 93, 95, 93, 91, 89, 79, 83, 91, 81, 96, 95, 88,
              95, 89, 93, 90, 93, 96, 93, 88, 83, 81, 84, 99, 88, 90, 82, 83, 90, 99, 84, 97, 93, 83, 86, 93, 86, 94,
              96, 80, 92, 81, 95, 82, 86, 92, 96, 98, 83, 81, 84, 88, 96, 84, 97, 102, 95, 90, 86, 83, 86, 77, 90, 84,
              89, 83, 92, 90, 86, 90, 81, 85, 89, 97, 84, 90, 82, 83, 95, 96, 82, 91, 80, 86, 77, 81, 86, 77, 83, 94,
              89, 82, 81, 80, 91, 88, 96, 81, 84, 86, 89, 88, 103, 79, 86, 105, 85, 88, 88, 81, 89, 82, 95, 85, 79, 84,
              76, 82, 85, 82, 87, 83, 76, 93, 81, 90, 80, 79, 86, 89, 83, 93, 92, 83, 73, 82, 91, 74, 78, 80, 76, 87,
              87, 88, 75, 83, 81, 82, 87, 86, 77, 90, 77, 79, 86, 85, 92, 78, 86, 77, 86, 84, 83, 87, 89, 87, 85, 80,
              87, 77, 77, 74, 82, 84, 85, 78, 80, 80, 77, 86, 80, 73, 83, 74, 90, 75, 98, 76, 87, 83, 113, 102, 106, 84,
              89, 88, 91, 92, 81, 86, 81, 82, 79, 80, 80, 87, 84, 82, 81, 94, 85, 79, 73, 81, 73, 79, 77, 84, 86, 81,
              71, 75, 84, 82, 83, 94, 76, 81, 84, 81, 82, 80, 74, 80, 76, 87, 81, 73, 76, 72, 73, 83, 77, 85, 72, 78,
              76, 90, 76, 78, 86, 79, 76, 89, 83, 79, 80, 79, 75, 75, 75, 82, 74, 75, 79, 80, 84, 85, 79, 85, 77, 78,
              85, 75, 74, 76, 72, 79, 76, 74, 80, 76, 74, 79, 77, 105, 80, 84, 92, 87, 84, 83, 195, 224, 243, 287, 253,
              268, 253, 272, 268, 263, 281, 263, 279, 278, 278, 276, 253, 266, 271, 259, 267, 264, 268, 284, 264, 261,
              275, 279, 270, 281, 252, 257, 263, 280, 270, 273, 260, 266, 260, 254, 258, 256, 253, 262, 276, 267, 293,
              263, 267, 267, 250, 281, 272, 250, 250, 241, 235, 224, 217, 217, 193, 197, 177, 157, 159, 156, 157, 157,
              152, 134, 120, 117, 101, 113, 103, 97, 108, 107, 79, 77, 80, 78, 82, 71, 76, 71, 78, 79, 83, 77, 76, 76,
              78, 73, 76, 82, 79, 76, 72, 74, 85, 73, 74, 80, 78, 86, 84, 89, 79, 73, 71, 80, 80, 82, 80, 89, 84, 81,
              76, 72, 74, 84, 70, 90, 83, 75, 67, 73, 81, 75, 77, 75, 76, 82, 89, 75, 77, 76, 81, 79, 72, 84, 79, 82,
              94, 76, 79, 78, 70, 73, 85, 70, 76, 73, 81, 79, 79, 76, 79, 80, 75, 80, 72, 80, 64, 74, 80, 80, 70, 74,
              83, 77, 77, 95, 69, 87, 74, 80, 89, 72, 82, 75, 76, 79, 82, 82, 76, 80, 86, 84, 83, 82, 72, 78, 82, 78,
              85, 77, 86, 72, 70, 84, 88, 81, 77, 83, 84, 101, 76, 86, 76, 71, 89, 76, 79, 70, 74, 80, 89, 72, 75, 74,
              84, 96, 80, 73, 78, 79, 84, 87, 73, 78, 84, 77, 80, 75, 75, 78, 80]
    prediction = [107, 118, 116, 107, 100, 94, 94, 98, 99, 94, 92, 90, 94, 91, 92, 94, 102, 101, 97, 91, 96, 95, 102,
                  96, 93, 95, 95, 97, 94, 96, 97, 99, 91, 92, 97, 96, 89, 86, 89, 90, 93, 96, 96, 95, 93, 87, 86, 90,
                  87, 93, 96, 94, 96, 94, 95, 94, 95, 97, 97, 94, 90, 86, 86, 94, 93, 93, 89, 87, 90, 97, 92, 96, 96,
                  91, 90, 93, 91, 94, 97, 90, 92, 88, 93, 89, 89, 92, 96, 99, 93, 88, 87, 89, 94, 91, 95, 101, 100, 97,
                  92, 89, 89, 84, 88, 88, 90, 88, 91, 93, 91, 92, 88, 88, 90, 95, 92, 92, 88, 87, 92, 96, 91, 92, 88,
                  88, 84, 83, 86, 83, 84, 91, 92, 88, 86, 84, 89, 90, 95, 90, 88, 88, 90, 91, 99, 91, 89, 99, 95, 92,
                  91, 88, 89, 87, 93, 91, 86, 86, 82, 83, 86, 85, 88, 87, 83, 89, 87, 90, 87, 84, 86, 89, 88, 92, 94,
                  90, 82, 83, 89, 83, 81, 82, 80, 85, 88, 90, 84, 84, 84, 85, 87, 88, 84, 88, 84, 83, 86, 87, 91, 86,
                  87, 84, 86, 87, 86, 88, 90, 90, 89, 86, 88, 84, 81, 79, 82, 85, 87, 84, 83, 83, 81, 85, 84, 80, 83,
                  80, 86, 83, 92, 86, 88, 87, 102, 106, 109, 98, 94, 92, 93, 94, 89, 88, 86, 85, 83, 83, 83, 86, 87, 86,
                  85, 91, 90, 86, 80, 82, 79, 80, 80, 84, 87, 85, 79, 78, 82, 84, 85, 91, 86, 84, 86, 85, 85, 84, 80,
                  81, 80, 85, 85, 80, 79, 77, 76, 81, 81, 84, 80, 80, 79, 86, 83, 82, 85, 84, 81, 87, 87, 84, 83, 83,
                  80, 79, 78, 82, 79, 78, 80, 82, 84, 87, 84, 86, 83, 82, 85, 82, 79, 79, 77, 79, 79, 78, 80, 80, 78,
                  80, 80, 94, 89, 88, 92, 91, 89, 87, 191, 246, 267, 313, 276, 279, 269, 272, 274, 271, 279, 275, 279,
                  282, 283, 283, 270, 270, 274, 270, 271, 271, 272, 271, 276, 271, 272, 279, 280, 283, 271, 266, 268,
                  278, 278, 279, 273, 272, 270, 265, 264, 264, 261, 265, 274, 275, 288, 280, 276, 275, 265, 276, 279,
                  268, 261, 254, 247, 238, 230, 226, 212, 206, 194, 177, 169, 165, 163, 163, 160, 149, 136, 128, 116,
                  116, 112, 107, 109, 111, 97, 88, 85, 84, 85, 80, 80, 77, 79, 81, 84, 82, 81, 80, 81, 78, 79, 82, 82,
                  81, 78, 77, 83, 80, 78, 80, 81, 85, 87, 90, 86, 81, 77, 79, 81, 83, 83, 88, 88, 86, 82, 78, 77, 82,
                  78, 85, 86, 82, 75, 75, 79, 79, 79, 79, 79, 82, 87, 83, 81, 80, 82, 82, 78, 82, 83, 84, 91, 86, 83,
                  82, 77, 76, 82, 78, 78, 77, 80, 81, 82, 80, 81, 82, 80, 81, 78, 80, 73, 74, 79, 81, 77, 77, 81, 81,
                  80, 90, 82, 85, 81, 82, 87, 81, 83, 80, 79, 81, 83, 84, 82, 82, 86, 87, 86, 86, 80, 80, 82, 82, 85,
                  83, 86, 80, 76, 81, 87, 86, 83, 84, 86, 96, 88, 88, 83, 78, 85, 82, 82, 77, 77, 80, 86, 81, 79, 78,
                  82, 91, 88, 81, 80, 81, 84, 87, 82, 81, 84, 82, 82, 80, 79, 80]
    where_from, where_to = 50, 150
    how_far = 10

    # Given data
    # test_x = [123, 109, 96, 92, 85, 93, 98, 96, 87, 88, 86, 95, 85]
    # prediction = [107, 118, 116, 107, 100, 94, 94, 98, 99, 94, 92, 90, 94]

    # Create a figure and main axes
    fig, ax = plt.subplots(figsize=(12, 6))
    throughput = [max(prediction)] * where_from
    throughput.extend([max(test_x[where_from: where_to])] * len(test_x[where_from: where_to + 1]))
    throughput.extend([max(test_x)] * len(prediction[where_to + 1:]))
    # Plot the original data
    ax.plot(test_x, label='Workload', c=colors[0])
    ax.plot(prediction[0], '--', label='Throughput', c='black')
    ax.plot(prediction, label='Prediction', c=colors[1])
    ax.plot(throughput, c='w')
    # Add titles and labels
    # ax.set_title('LSTM ')
    ax.set_ylabel('Workload (RPS)')
    ax.set_xlabel('Time (second)')
    ax.legend(loc='upper right')

    # Create inset of the zoomed-in region
    axins = inset_axes(ax, width="40%", height="40%", loc="upper left", borderpad=4)

    # Plot the zoomed-in data on the inset axes
    axins.plot(test_x, label='test_x', c=colors[0])
    axins.plot(prediction, label='prediction', c=colors[1])
    axins.plot(throughput, '--', label='throughput', c='black')

    # Set the zoomed-in region
    x1, x2, y1, y2 = (where_from, where_to - 1,
                      min(min(test_x[where_from:where_to]), min(prediction[where_from:where_to])) - how_far,
                      max(max(test_x[where_from:where_to]), max(prediction[where_from:where_to])) + how_far)
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticks([])
    axins.set_yticks([])
    mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="0.5")
    fig.tight_layout()
    plt.show()
    where_to_save_pdf = figure_path + 'lstm_result.pdf'
    where_to_save_png = figure_path + 'lstm_result.png'
    fig.savefig(where_to_save_pdf)
    fig.savefig(where_to_save_png)
    plt.close()
    crop_pdf(where_to_save_pdf)


def get_data_from_csv(filename_csv, start_range=0, end_range=600):
    data_hv = csv.DictReader(open(filename_csv, newline=''))
    p99_list = []
    cost_list = []
    drop_list = []
    violation_list = []
    rps_list = []
    p95_list = []
    p90_list = []
    p50_list = []
    for row in data_hv:
        if (row['rate'] == '' or float(row['rate']) == 0 or row['99'] == '' or row['cost'] == '' or row['50'] == '' or
                row['drop_rate'] == '' or row['within_slo'] == '' or row['95'] == '' or row['90'] == ''):
            continue
        rps_list.append(float(row['rate']))
        drop_list.append(float(row['drop_rate']))
        violation_list.append(1 - float(row['within_slo']))
        cost_list.append(float(row['cost']))
        p99_list.append(float(row['99']))
        p95_list.append(float(row['95']))
        p90_list.append(float(row['90']))
        p50_list.append(float(row['50']))
    while len(rps_list) < 1200:
        rps_list.append(0)
        p99_list.append(0)
        p95_list.append(0)
        p90_list.append(0)
        p50_list.append(0)
        violation_list.append(0)
        cost_list.append(min(cost_list))
        drop_list.append(0)
    current_violation_hv = []
    for i in range(len(rps_list)):
        if rps_list[i] == 0:
            current_violation_hv.append(0)
        else:
            extra_violation_multiplier = 0
            # if p99_list[i] > DEFAULT_SLO / 1000.0:
            #     if p50_list[i] > DEFAULT_SLO / 1000.0:
            #         extra_violation_multiplier = 0.5
            #     elif p90_list[i] > DEFAULT_SLO / 1000.0:
            #         extra_violation_multiplier = 0.1
            #     else:
            #         extra_violation_multiplier = 0.05
            total_violation = (extra_violation_multiplier + violation_list[i] + drop_list[i] / rps_list[i])
            if total_violation > 1:
                total_violation = 1
            current_violation_hv.append(total_violation)
    return (p99_list[start_range: end_range], cost_list[start_range: end_range], drop_list[start_range: end_range],
            violation_list[start_range: end_range], rps_list[start_range: end_range],
            current_violation_hv[start_range: end_range])


def bos_draw_experiment_results(pipeline):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
    saeed_thickness = 1.5
    p99_list_hv, cost_list_hv, drop_list_hv, violation_list_hv, rps_list_hv, current_violation_hv = get_data_from_csv(
        'experiment_results/series-' + pipeline + '-hv-1350-1350.csv')
    p99_list_ho, cost_list_ho, drop_list_ho, violation_list_ho, rps_list_ho, current_violation_ho = get_data_from_csv(
        'experiment_results/series-' + pipeline + '-ho-1350-1350.csv')
    p99_list_vo, cost_list_vo, drop_list_vo, violation_list_vo, rps_list_vo, current_violation_vo = get_data_from_csv(
        'experiment_results/series-' + pipeline + '-vo-1350-1350.csv')
    fig_length = len(p99_list_hv)
    ax1.set_ylabel('Workload (RPS)')
    ax1.set_xlim([0, fig_length])
    ax1.set_ylim([0, int(max(rps_list_hv)) + 5])
    ax1.set_yticks([x for x in range(0, int(max(rps_list_hv)) + 5, 10)])
    ax1.set_xticks([])
    ax1.plot(rps_list_hv, color='black', linewidth=saeed_thickness, label='Workload')
    ax1.plot([0], '--', color='dimgray', linewidth=saeed_thickness, label='SLO')
    ax1.plot([0], color=colors[0], linewidth=saeed_thickness, label=paper_name)
    ax1.plot([0], color=colors[1], linewidth=saeed_thickness, label=horizontal_name)
    ax1.plot([0], color=colors[2], linewidth=saeed_thickness, label=vertical_name)
    ax1.grid()
    ax1.legend(loc='upper center', bbox_to_anchor=(0.48, 1.28), fancybox=True, shadow=True, ncol=5, columnspacing=0.3)

    ax2.set_ylabel('P99 Latency (s)')
    ax2.set_xlim([0, fig_length])
    ax2.set_ylim([0, 1.6])
    ax2.set_yticks([0, 0.3, 0.6, 0.9, 1.2, 1.5])
    ax2.set_xticks([])
    ax2.plot([DEFAULT_SLO / 1000] * fig_length, '--', color='dimgray', linewidth=saeed_thickness, label='SLO')
    ax2.plot(p99_list_hv, color=colors[0], linewidth=saeed_thickness)
    ax2.plot(p99_list_vo, color=colors[2], linewidth=saeed_thickness)
    ax2.plot(p99_list_ho, color=colors[1], linewidth=saeed_thickness)
    ax2.grid()

    ax3.set_ylabel('Cost (CPU cores)')
    ax3.set_xlim([0, fig_length])
    ax3.set_ylim([0, max(cost_list_hv) + 5])
    ax3.set_yticks([x for x in range(0, int(max(cost_list_hv)) + 4, 10)])
    ax3.set_xticks([])
    ax3.plot(cost_list_hv, color=colors[0], linewidth=saeed_thickness)
    ax3.plot(cost_list_vo, color=colors[2], linewidth=saeed_thickness)
    ax3.plot(cost_list_ho, color=colors[1], linewidth=saeed_thickness)
    ax3.grid()

    ax4.set_ylabel('Violation (%)')
    ax4.set_xlim([0, fig_length])
    ax4.set_ylim([0, 100])
    ax4.set_yticks([0, 20, 40, 60, 80, 99])
    ax4.set_xticks([x for x in range(0, fig_length + 1, 200)])
    ax4.set_xlabel('Time (s)')
    # print(numpy.argmax(current_violation))
    ax4.plot([x * 100 for x in current_violation_vo], color=colors[2], linewidth=saeed_thickness)
    ax4.plot([x * 100 for x in current_violation_ho], color=colors[1], linewidth=saeed_thickness)
    ax4.plot([x * 100 for x in current_violation_hv], color=colors[0], linewidth=saeed_thickness)
    ax4.grid(axis='y')

    fig.tight_layout(pad=0.65)
    where_to_save_pdf = figure_path + 'e2e_evaluation_' + pipeline + '_780.pdf'
    where_to_save_png = figure_path + 'e2e_evaluation_' + pipeline + '_780.png'
    fig.savefig(where_to_save_pdf)
    fig.savefig(where_to_save_png)
    plt.show()
    plt.close()
    crop_pdf(where_to_save_pdf)


def bos_draw_drop():
    patterns = ["+", "/", "\\"]
    starting_range = 0
    ending_range = 100
    _, _, _, _, rps_list_hv0, current_violation_hv0 = get_data_from_csv(
        'experiment_results/series-video-hv-780-0.csv', starting_range, ending_range)
    _, _, _, _, rps_list_ho0, current_violation_ho0 = get_data_from_csv(
        'experiment_results/series-video-ho-780-0.csv', starting_range, ending_range)
    _, _, _, _, rps_list_vo0, current_violation_vo0 = get_data_from_csv(
        'experiment_results/series-video-vo-780-0.csv', starting_range, ending_range)

    _, _, _, _, rps_list_hv1, current_violation_hv1 = get_data_from_csv(
        'experiment_results/series-video-hv-780-780.csv', starting_range, ending_range)
    _, _, _, _, rps_list_ho1, current_violation_ho1 = get_data_from_csv(
        'experiment_results/series-video-ho-780-780.csv', starting_range, ending_range)
    _, _, _, _, rps_list_vo1, current_violation_vo1 = get_data_from_csv(
        'experiment_results/series-video-vo-780-780.csv', starting_range, ending_range)

    _, _, _, _, rps_list_hv2, current_violation_hv2 = get_data_from_csv(
        'experiment_results/series-video-hv-780-1560.csv', starting_range, ending_range)
    _, _, _, _, rps_list_ho2, current_violation_ho2 = get_data_from_csv(
        'experiment_results/series-video-ho-780-1560.csv', starting_range, ending_range)
    _, _, _, _, rps_list_vo2, current_violation_vo2 = get_data_from_csv(
        'experiment_results/series-video-vo-780-1560.csv', starting_range, ending_range)

    _, _, _, _, rps_list_hv3, current_violation_hv3 = get_data_from_csv(
        'experiment_results/series-video-hv-780-2340.csv', starting_range, ending_range)
    _, _, _, _, rps_list_ho3, current_violation_ho3 = get_data_from_csv(
        'experiment_results/series-video-ho-780-2340.csv', starting_range, ending_range)
    _, _, _, _, rps_list_vo3, current_violation_vo3 = get_data_from_csv(
        'experiment_results/series-video-vo-780-2340.csv', starting_range, ending_range)

    labels = [paper_name, horizontal_name, vertical_name]
    names = ['1xSLO',
             # '2xSLO',
             '3xSLO',
             'No Dropping']
    # This controls how many bars we get in each group
    values = [[sum(current_violation_hv1) * 100 / len(rps_list_hv1),
               # sum(current_violation_hv2) * 100 / len(rps_list_hv2),
               sum(current_violation_hv3) * 100 / len(rps_list_hv3),
               sum(current_violation_hv0) * 100 / len(rps_list_hv0)],
              [sum(current_violation_ho1) * 100 / len(rps_list_ho1),
               # sum(current_violation_ho2) * 100 / len(rps_list_ho2),
               sum(current_violation_ho3) * 100 / len(rps_list_ho3),
               sum(current_violation_ho0) * 100 / len(rps_list_ho0)],
              [sum(current_violation_vo1) * 100 / len(rps_list_vo1),
               # sum(current_violation_vo2) * 100 / len(rps_list_vo2),
               sum(current_violation_vo3) * 100 / len(rps_list_vo3),
               sum(current_violation_vo0) * 100 / len(rps_list_vo0)]]
    # print(values)
    n = len(values)  # Number of bars to plot
    mehran_thickness = .25  # With of each column
    x = np.arange(0, len(names))  # Center position of group on x axis

    for i, value in enumerate(values):
        position = x + (mehran_thickness * (1 - n) / 2) + i * mehran_thickness
        plt.bar(position, value, color=colors[i], hatch=patterns[i], width=mehran_thickness, label=labels[i])
    plt.ylim([0, 65])
    plt.yticks([0, 20, 40, 60])
    plt.xticks(x, names)
    plt.ylabel('SLO Violation (%)')
    plt.xlabel('Dropping Strategy')
    plt.legend(loc='upper left')
    plt.grid(axis='y')
    plt.tight_layout(pad=0.65)
    where_to_save_pdf = figure_path + 'dropping_video_780.pdf'
    where_to_save_png = figure_path + 'dropping_video_780.png'
    plt.savefig(where_to_save_pdf)
    plt.savefig(where_to_save_png)
    plt.show()
    plt.close()
    crop_pdf(where_to_save_pdf)


project_path = '/home/kamran/vertical_scaling/pythonProject/'
figure_path = project_path + 'figures/bos/'
paper_name = 'Biscale'
horizontal_name = 'Horizontal Scaling'
vertical_name = 'Vertical Scaling'
DEFAULT_SLO = 1350
legend_fontsize = 13
plt.rc('legend', fontsize=legend_fontsize)
font_small_size = 18
plt.rcParams.update({'font.size': font_small_size})
canvas_size = 4
plt.rcParams['figure.figsize'] = 2 * canvas_size, 12
plt.rcParams["font.family"] = "Arial"
plt.rcParams['hatch.linewidth'] = 0.7
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
colors = ['#005906', 'chocolate', '#f7cc3e', 'plum', '#3894fc']
# colors = ['#5e3c99', '#e66101', '#fdb863', '#b2abd2']
# vertical_vs_horizontal()
# vertical_vs_horizontal_short()
# vertical_vs_horizontal_long()
# latency_vs_vertical_vs_batch()
# dynamic_sla_figure()
# preliminary_evaluation()
# bos_draw_cpu_batch_figure()
# bos_draw_inter_intra()
# bos_draw_lstm_result()
bos_draw_experiment_results('sentiment')
# bos_draw_drop()

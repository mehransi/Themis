import math
import os

core_to_G_RAM = 18

def latency(core, batch, alpha, beta, gamma, zeta):
    mlt = float(os.getenv("LATENCY_MODEL_MULTIPLIER", "1"))
    mltb = float(os.getenv("LATENCY_MODEL_BATCH_MULTIPLIER", "1"))
    return round(mlt * (alpha * batch / core + mltb * (beta * batch) + gamma / core + zeta))


def get_throughput(state: dict, models: dict):
    current_tp = math.inf
    for s in range(len(state)):
        b = state[s][2]
        replicas = state[s][1]
        cores = state[s][0]
        current_latency = latency(cores, b, *models[s])
        tp = replicas * int(1000 * b / current_latency)
        if tp < current_tp:
            current_tp = tp
    return current_tp


def horizontal_2d(b_max: list, c_max: list, slo, models, stage_memory_requests_G, workload):
    if workload == 0:
        return {s: [1, 1] for s in range(len(models))}
    dp = []
    best = []
    for i in range(len(models)):
        dp.append([False] * (slo + 1))
        best.append([(math.inf, math.inf, math.inf, math.inf)] * (slo + 1))
    best[0][0] = (0, 0, 0, 0)
    dp[0][0] = True
    for s in range(len(models)):
        for i in range(slo, -1, -1):
            if (s == 0 and dp[s][i]) or s != 0 and dp[s - 1][i]:
                for c in range(1, c_max[s] + 1):
                    for b in range(1, b_max[s] + 1):
                        curr_latency = latency(c, b, models[s][0], models[s][1], models[s][2], models[s][3])
                        throughput = int(1000 * b / curr_latency)
                        curr_latency += int((b - 1) * 1000 / workload)
                        
                        if i + curr_latency > slo:
                            continue
                        
                        needed_instances = math.ceil(workload / throughput)
                        cost =  needed_instances * (stage_memory_requests_G[s] + c * core_to_G_RAM)
                        # The first model
                        if s == 0:
                            if dp[s][i + curr_latency] is False:
                                dp[s][i + curr_latency] = True
                                best[s][i + curr_latency] = (cost, needed_instances, c, b)
                            elif cost < best[s][i + curr_latency][0]:
                                best[s][i + curr_latency] = (cost, needed_instances, c, b)
                        elif dp[s - 1][i] and i > 0:
                            if dp[s][i + curr_latency] is False:
                                dp[s][i + curr_latency] = True
                                best[s][i + curr_latency] = (best[s - 1][i][0] + cost, needed_instances, c, b)
                            elif best[s - 1][i][0] + cost < best[s][i + curr_latency][0]:
                                best[s][i + curr_latency] = (best[s - 1][i][0] + cost, needed_instances, c, b)
    least_n = math.inf
    ind = -1
    for i in range(slo):
        if dp[len(models) - 1][i]:
            if ind == -1:
                ind = i
                least_n = best[len(models) - 1][i][0]
            elif best[len(models) - 1][i][0] < least_n:
                ind = i
                least_n = best[len(models) - 1][i][0]
    res = {}
    if ind == -1:
        return -1
    else:
        stage = len(models) - 1
        while stage >= 0:
            res[stage] = [best[stage][ind][1], best[stage][ind][2], best[stage][ind][3]]
            ind -= (
                latency(
                    best[stage][ind][2],
                    best[stage][ind][3],
                    models[stage][0],
                    models[stage][1],
                    models[stage][2],
                    models[stage][3]
                ) + int((best[stage][ind][3] - 1) * 1000 / workload))
            stage -= 1
    return res


def vertical_2d(b_max: list, c_max: list, slo, models, current_instance, workload, depth=1):
    if workload == 0:
        return {s: [1, 1] for s in range(len(models))}, [0] * len(models)
    dp = []
    best = []
    for i in range(len(models)):
        dp.append([False] * (slo + 1))
        best.append([(math.inf, math.inf, math.inf)] * (slo + 1))
    best[0][0] = (0, 0, 0)
    dp[0][0] = True
    for s in range(len(models)):
        for i in range(slo, -1, -1):
            if (s == 0 and dp[s][i]) or s != 0 and dp[s - 1][i]:
                for c in range(1, c_max[s] + 1):
                    for b in range(1, b_max[s] + 1):
                        curr_latency = latency(c, b, models[s][0], models[s][1], models[s][2], models[s][3])
                        throughput = int(1000 * b / curr_latency)
                        curr_latency += int((b - 1) * 1000 / workload)
                        
                        if i + curr_latency > slo:
                            continue
                        
                        if throughput * current_instance[s][1] < workload:
                            continue
                        # The first model
                        if s == 0:
                            if dp[s][i + curr_latency] is False:
                                dp[s][i + curr_latency] = True
                                best[s][i + curr_latency] = (c, c, b)
                        # Not the first model
                        elif dp[s - 1][i] and i > 0:
                            # if TOTAL_CORE < best[s - 1][i][0] + c:
                            #     continue
                            if dp[s][i + curr_latency] is False:
                                dp[s][i + curr_latency] = True
                                best[s][i + curr_latency] = (best[s - 1][i][0] + c, c, b)
                            elif best[s - 1][i][0] + c < best[s][i + curr_latency][0]:
                                best[s][i + curr_latency] = (best[s - 1][i][0] + c, c, b)
    least_c = math.inf
    ind = -1
    for i in range(slo):
        if dp[len(models) - 1][i] and best[len(models) - 1][i][0] < least_c:
            ind = i
            least_c = best[len(models) - 1][i][0]
    res = {}
    if ind == -1:
        if depth != 1:
            return -1, []
        left, right = 1, workload
        while right - left > 1:
            mid = (right + left) // 2
            rec, _ = vertical_2d(b_max, c_max, slo, models, current_instance, mid, 2)
            if rec == -1:
                right = mid
            else:
                left = mid
        config_vertical_limited, _ = vertical_2d(b_max, c_max, slo, models, current_instance, left)
        wl = workload - left
        extra_list = []
        for x in range(len(models)):
            cl = latency(config_vertical_limited[x][0], config_vertical_limited[x][1],
                         models[x][0], models[x][1], models[x][2], models[x][3])
            th = int(1000 * config_vertical_limited[x][1] / cl)
            cl += int((config_vertical_limited[x][1] - 1) * 1000 / wl)
            
            extra_list.append(math.ceil(wl / th))
        return config_vertical_limited, extra_list
    elif ind != -1 and depth != 1:
        return 0, []
    else:
        counter = len(models) - 1
        while counter >= 0:
            res[counter] = [best[counter][ind][1], best[counter][ind][2]]
            ind -= (latency(best[counter][ind][1], best[counter][ind][2], models[counter][0], models[counter][1],
                            models[counter][2], models[counter][3])
                    + int((best[counter][ind][2] - 1) * 1000 / workload))
            counter -= 1
    return res, [0] * len(models)


if __name__ == "__main__":
    batch_max = 4
    core_max = 4
    slo_max = 300
    memory_stages = {0: 2, 1: 1}
    models_set = {0: [39.90889958924582, 7.784141716703453, 1.8333345492230113, 0.4507066167132669],
                1: [28.084275010842124, 2.626952412159743, 6.807172001409088, 0.4588339579256442]}

    current_workload = 150
    config_current = {0: [1, 1, 1], 1: [1, 1, 1]}
    config_vertical, extra_instances = vertical_2d([batch_max, batch_max], [core_max, core_max], slo_max,
                                                   models_set, config_current, current_workload)
    config_horizontal = horizontal_2d([batch_max, batch_max], [core_max, core_max], slo_max, models_set, memory_stages, current_workload)
    print('Current Config: {MODEL: CORE, INSTANCE, BATCH}')
    print(config_current)
    print('Horizontal Config: {MODEL: [INSTANCE, BATCH]}')
    print(config_horizontal)
    print('Vertical Config: {MODEL: [CORE, BATCH]}, Extra: [INSTANCE]')
    print(config_vertical, extra_instances)

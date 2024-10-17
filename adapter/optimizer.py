import math
import os

TOTAL_CORE = 32


def latency(core, batch, alpha, beta, gamma, zeta):
    mlt = float(os.getenv("LATENCY_MODEL_MULTIPLIER", "1"))
    mltb = float(os.getenv("LATENCY_MODEL_BATCH_MULTIPLIER", "1"))
    return int(mlt * (alpha * batch / core + mltb * (beta * batch) + gamma / core + zeta))


def horizontal_2d(b_max, slo, models, workload):
    if workload == 0:
        return {s: [1, 1] for s in range(len(models))}
    dp = []
    best = []
    for i in range(len(models)):
        dp.append([False] * (slo + 1))
        best.append([(1000, 1000, 1000)] * (slo + 1))
    best[0][0] = (0, 0, 0)
    dp[0][0] = True
    for s in range(len(models)):
        for i in range(slo, -1, -1):
            if (s == 0 and dp[s][i]) or s != 0 and dp[s - 1][i]:
                for b in range(1, b_max + 1):
                    curr_latency = latency(1, b, models[s][0], models[s][1], models[s][2], models[s][3])
                    throughput = int(1000 * b / curr_latency)
                    curr_latency += int((b - 1) * 1000 / workload)
                    
                    if i + curr_latency > slo:
                        continue
                    
                    needed_instances = math.ceil(workload / throughput)
                    # The first model
                    if s == 0:
                        if dp[s][i + curr_latency] is False:
                            dp[s][i + curr_latency] = True
                            best[s][i + curr_latency] = (needed_instances, needed_instances, b)
                        elif needed_instances < best[s][i + curr_latency][0]:
                            best[s][i + curr_latency] = (needed_instances, needed_instances, b)
                    elif dp[s - 1][i] and i > 0:
                        if dp[s][i + curr_latency] is False:
                            dp[s][i + curr_latency] = True
                            best[s][i + curr_latency] = (best[s - 1][i][0] + needed_instances, needed_instances, b)
                        elif best[s - 1][i][0] + needed_instances < best[s][i + curr_latency][0]:
                            best[s][i + curr_latency] = (best[s - 1][i][0] + needed_instances, needed_instances, b)
    least_n = 1000
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
        counter = len(models) - 1
        while counter >= 0:
            res[counter] = [best[counter][ind][1], best[counter][ind][2]]
            ind -= (latency(1, best[counter][ind][2], models[counter][0], models[counter][1], models[counter][2],
                            models[counter][3]) + int((best[counter][ind][2] - 1) * 1000 / workload))
            counter -= 1
    return res


def vertical_2d(b_max, c_max, slo, models, current_instance, workload, depth=1):
    if workload == 0:
        return {s: [1, 1] for s in range(len(models))}, [0] * len(models)
    dp = []
    best = []
    for i in range(len(models)):
        dp.append([False] * (slo + 1))
        best.append([(1000, 1000, 1000)] * (slo + 1))
    best[0][0] = (0, 0, 0)
    dp[0][0] = True
    for s in range(len(models)):
        for i in range(slo, -1, -1):
            if (s == 0 and dp[s][i]) or s != 0 and dp[s - 1][i]:
                for c in range(1, c_max + 1):
                    for b in range(1, b_max + 1):
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
                            if TOTAL_CORE < best[s - 1][i][0] + c:
                                continue
                            if dp[s][i + curr_latency] is False:
                                dp[s][i + curr_latency] = True
                                best[s][i + curr_latency] = (best[s - 1][i][0] + c, c, b)
                            elif best[s - 1][i][0] + c < best[s][i + curr_latency][0]:
                                best[s][i + curr_latency] = (best[s - 1][i][0] + c, c, b)
    least_c = 1000
    ind = -1
    for i in range(slo):
        if dp[len(models) - 1][i]:
            if ind == -1:
                ind = i
                least_c = best[len(models) - 1][i][0]
            elif best[len(models) - 1][i][0] < least_c:
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
        current_extra = {}
        mehran_list = []
        for x in range(len(models)):
            cl = latency(config_vertical_limited[x][0], config_vertical_limited[x][1],
                         models[x][0], models[x][1], models[x][2], models[x][3])
            th = int(1000 * config_vertical_limited[x][1] / cl)
            cl += int((config_vertical_limited[x][1] - 1) * 1000 / wl)
            
            current_extra[x] = [config_vertical_limited[x][0], math.ceil(wl / th), config_vertical_limited[x][1]]
            mehran_list.append(math.ceil(wl / th))
        return config_vertical_limited, mehran_list
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
    batch_max = 8
    core_max = 8
    slo_max = 1000
    models_set = {0: [84.77978914419758, 21.867603656295096, 0.3475406625080165, -3.3934556468994534],
                  1: [61.4976951513907, 3.815267577294629, 14.901415567121793, 11.041543118691706]}

    current_workload = 30
    config_current = {0: [1, 1, 1], 1: [1, 1, 1]}
    config_vertical, extra_instances = vertical_2d(batch_max, core_max, slo_max,
                                                   models_set, config_current, current_workload)
    config_horizontal = horizontal_2d(batch_max, slo_max, models_set, current_workload)
    print('Current Config: {MODEL: CORE, INSTANCE, BATCH}')
    print(config_current)
    print('Horizontal Config: {MODEL: [INSTANCE, BATCH]}')
    print(config_horizontal)
    print('Vertical Config: {MODEL: [CORE, BATCH]}, Extra: [INSTANCE]')
    print(config_vertical, extra_instances)

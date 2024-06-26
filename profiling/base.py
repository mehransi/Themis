from scipy.optimize import curve_fit
import time
import torch

    
def batch_cost_latency_model(cpu_batch_tuple, alpha, beta, gamma, zeta):
        cpu, batch = cpu_batch_tuple
        return alpha * batch / cpu + beta * batch + gamma / cpu + zeta

            
# FIXME
batch_sizes = []
cpu_sizes = []
latencies = [] # ms
params, _ = curve_fit(batch_cost_latency_model, (batch_sizes, cpu_sizes), latencies)
alpha, beta, gamma, zeta = params  # eq 2 parameters
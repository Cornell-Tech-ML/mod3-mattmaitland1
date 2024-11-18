import random
from collections import defaultdict
import minitorch
import time
import sys
import numpy as np
import matplotlib.pyplot as plt

FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)
GPUBackend = minitorch.TensorBackend(minitorch.CudaOps)


def run_matmul(backend, size=16) -> None:
    batch_size = 2

    x = minitorch.rand((batch_size, size, size), backend=backend)
    y = minitorch.rand((batch_size, size, size), backend=backend)
    z = x @ y


if __name__ == "__main__":
    # Warmup
    run_matmul(FastTensorBackend)
    run_matmul(GPUBackend)

    ntrials = 3
    times = {}
    sizes = [64, 128, 256, 512, 1024]
    
    for size in sizes:
        print(f"Running size {size}")
        times[size] = {}
        fast_times = []
        gpu_times = []
        
        for _ in range(ntrials):
            start_fast = time.time()
            run_matmul(FastTensorBackend, size)
            end_fast = time.time()

            start_gpu = time.time()
            run_matmul(GPUBackend, size)
            end_gpu = time.time()

            fast_times.append(end_fast - start_fast)
            gpu_times.append(end_gpu - start_gpu)

        times[size]["fast"] = np.mean(fast_times)
        times[size]["gpu"] = np.mean(gpu_times)
        print(times[size])

    # Create plots
    plt.figure(figsize=(12, 6))
    
    # Line plot
    plt.subplot(1, 2, 1)
    plt.plot(sizes, [times[s]["fast"] for s in sizes], 'b-o', label='CPU (Fast)')
    plt.plot(sizes, [times[s]["gpu"] for s in sizes], 'r-o', label='GPU')
    plt.xlabel('Matrix Size')
    plt.ylabel('Time (seconds)')
    plt.title('Matrix Multiplication Performance')
    plt.legend()
    plt.grid(True)
    
    # Log scale plot
    plt.subplot(1, 2, 2)
    plt.loglog(sizes, [times[s]["fast"] for s in sizes], 'b-o', label='CPU (Fast)')
    plt.loglog(sizes, [times[s]["gpu"] for s in sizes], 'r-o', label='GPU')
    plt.xlabel('Matrix Size (log scale)')
    plt.ylabel('Time (seconds) (log scale)')
    plt.title('Performance (Log Scale)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('matmul_performance.png')
    plt.show()

    # Print timing summary
    print("\nTiming summary")
    for size, stimes in times.items():
        print(f"Size: {size}")
        for b, t in stimes.items():
            print(f"    {b}: {t:.5f}")
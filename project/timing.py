import random
from collections import defaultdict
import minitorch
import time
import sys
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

    # Create plots using plotly
    fig = make_subplots(rows=1, cols=2,
                       subplot_titles=('Linear Scale', 'Log Scale'))

    # Linear scale plot
    fig.add_trace(
        go.Scatter(x=sizes, y=[times[s]["fast"] for s in sizes],
                  name="CPU (Fast)", line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=sizes, y=[times[s]["gpu"] for s in sizes],
                  name="GPU", line=dict(color='red')),
        row=1, col=1
    )

    # Log scale plot
    fig.add_trace(
        go.Scatter(x=sizes, y=[times[s]["fast"] for s in sizes],
                  name="CPU (Fast)", line=dict(color='blue', log_y=True)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=sizes, y=[times[s]["gpu"] for s in sizes],
                  name="GPU", line=dict(color='red', log_y=True)),
        row=1, col=2
    )

    fig.update_xaxes(title_text='Matrix Size')
    fig.update_yaxes(title_text='Time (seconds)')
    fig.update_layout(title_text='Matrix Multiplication Performance')

    fig.show()

    # Print timing summary
    print("\nTiming summary")
    for size, stimes in times.items():
        print(f"Size: {size}")
        for b, t in stimes.items():
            print(f"    {b}: {t:.5f}")
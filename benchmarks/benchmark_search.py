import asyncio
import httpx
import time
import numpy as np
import psutil

URL = "http://127.0.0.1:8000/search?q=phone&k=20"
DURATION = 20

async def worker(client, results):
    try:
        start = time.time()
        await client.get(URL)
        results.append(time.time() - start)
    except:
        pass

async def run(concurrency):
    results = []
    limits = httpx.Limits(max_connections=concurrency*2)
    timeout = httpx.Timeout(30.0)

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        start_time = time.time()
        while time.time() - start_time < DURATION:
            tasks = [worker(client, results) for _ in range(concurrency)]
            await asyncio.gather(*tasks)
    return results

def summarize(latencies):
    arr = np.array(latencies)
    return {
        "requests": len(arr),
        "rps": len(arr) / DURATION,
        "min": arr.min(),
        "mean": arr.mean(),
        "median": np.median(arr),
        "p95": np.percentile(arr, 95),
        "p99": np.percentile(arr, 99)
    }

if __name__ == "__main__":
    for c in [50, 100, 200, 400, 800]:
        latencies = asyncio.run(run(c))

        if len(latencies) == 0:
            print(f"\n=== Concurrency {c} === FAILED")
            continue

        stats = summarize(latencies)
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent

        print(f"\n=== Concurrency {c} ===")
        for k, v in stats.items():
            print(f"{k}: {v:.4f}")
        print(f"CPU: {cpu}% | MEM: {mem}%")

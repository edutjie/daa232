import time
import psutil
import os
from knapsack.unbounded import UnboundedKnapsack, BranchBoundUnboundedKnapsack


def run_algo_and_measure(algo, *args):
    start_time = time.time()
    # Replace this line with your sorting algorithm of choice
    result = algo(*args)
    end_time = time.time()
    execution_time = end_time - start_time
    # convert to millisecondsq
    execution_time *= 1000
    return execution_time, result


def evaluate(algo, *args) -> None:
    execution_time, result = run_algo_and_measure(algo, *args)
    # convert to MB
    mem_usage = psutil.Process().memory_info().rss / 1024 / 1024
    psutil.Process(os.getpid()).memory_info().rss
    if type(result) == tuple:
        result = result[0]
    print(f"Result: {result}")
    print(f"Execution time: {execution_time:.6f} ms; Memory usage: {mem_usage} MB")


def load_dataset_from_txt(path: str) -> list[int]:
    with open(path, "r") as f:
        return [int(line) for line in f.readlines()]


if __name__ == "__main__":
    dataset = [
        (10, "v_kecil.txt", "w_kecil.txt"),
        (100, "v_sedang.txt", "w_sedang.txt"),
        (1000, "v_besar.txt", "w_besar.txt"),
    ]

    for algo in [UnboundedKnapsack, BranchBoundUnboundedKnapsack]:
        print(f"Algo: {algo.__name__}")
        for W, v_file, w_file in dataset:
            print(f"Dataset: {v_file}, {w_file}; W: {W}")
            v = load_dataset_from_txt(v_file)
            w = load_dataset_from_txt(w_file)
            uk = algo(W, v, w)
            evaluate(uk.calculate_max_value)
            print()
        print()

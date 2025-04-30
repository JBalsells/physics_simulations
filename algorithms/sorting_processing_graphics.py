import time
import random
from multiprocessing import Process, Manager
import matplotlib.pyplot as plt

# ----------------------------
# Decorador de benchmark
# ----------------------------

def time_measurement(times_list):
    def decorator(func):
        def wrapper(arr):
            start = time.time()
            func(arr)
            end = time.time()
            times_list.append((len(arr), end - start))
        return wrapper
    return decorator

# ----------------------------
# Algoritmos de ordenamiento
# ----------------------------

def bubble_sort_algorithm(arr):
    n = len(arr)
    a = arr.copy()
    for i in range(n):
        for j in range(0, n - i - 1):
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
    return a

def counting_sort(arr, exp):
    n = len(arr)
    output = [0] * n
    count = [0] * 10
    for num in arr:
        index = (num // exp) % 10
        count[index] += 1
    for i in range(1, 10):
        count[i] += count[i - 1]
    i = n - 1
    while i >= 0:
        index = (arr[i] // exp) % 10
        output[count[index] - 1] = arr[i]
        count[index] -= 1
        i -= 1
    return output

def radix_sort_algorithm(arr):
    if len(arr) == 0:
        return arr
    max_num = max(arr)
    exp = 1
    result = arr.copy()
    while max_num // exp > 0:
        result = counting_sort(result, exp)
        exp *= 10
    return result

def quick_sort_algorithm(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort_algorithm(left) + middle + quick_sort_algorithm(right)

# ----------------------------
# Benchmark runner
# ----------------------------

def run_benchmarks(max_n, step):
    manager = Manager()
    bubble_times = manager.list()
    quick_times = manager.list()
    radix_times = manager.list()

    @time_measurement(bubble_times)
    def bubble(arr):
        bubble_sort_algorithm(arr)

    @time_measurement(quick_times)
    def quick(arr):
        quick_sort_algorithm(arr)

    @time_measurement(radix_times)
    def radix(arr):
        radix_sort_algorithm(arr)

    for n in range(100, max_n + 1, step):
        array = [random.randint(0, 10000) for _ in range(n)]
        arr_bubble = array.copy()
        arr_quick = array.copy()
        arr_radix = array.copy()

        p1 = Process(target=bubble, args=(arr_bubble,))
        p2 = Process(target=quick, args=(arr_quick,))
        p3 = Process(target=radix, args=(arr_radix,))

        p1.start()
        p2.start()
        p3.start()

        p1.join()
        p2.join()
        p3.join()

    return list(bubble_times), list(quick_times), list(radix_times)

# ----------------------------
# Visualización
# ----------------------------

if __name__ == "__main__":
    max_n = 5000
    step = 100
    bubble, quick, radix = run_benchmarks(max_n, step)

    #bubble.sort()
    #quick.sort()
    #radix.sort()

    plt.figure(figsize=(10, 6))
    plt.plot([x[0] for x in bubble], [x[1] for x in bubble], label="Bubble Sort")
    plt.plot([x[0] for x in quick], [x[1] for x in quick], label="Quick Sort")
    plt.plot([x[0] for x in radix], [x[1] for x in radix], label="Radix Sort")
    plt.xlabel("Tamaño del arreglo (n)")
    plt.ylabel("Tiempo de ejecución (segundos)")
    plt.title("Comparación de tiempos de ejecución de algoritmos de ordenamiento")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

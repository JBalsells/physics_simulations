import time
import random
from multiprocessing import Process, current_process

def time_measurement(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Execution time in {func.__name__}: {end - start:.6f} seconds")
        return result
    return wrapper

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

@time_measurement
def bubble_sort(arr):
    return bubble_sort_algorithm(arr)

@time_measurement
def quick_sort(arr): # funcion de envoltorio
    return quick_sort_algorithm(arr)

@time_measurement
def radix_sort(arr):
    return radix_sort_algorithm(arr)

if __name__ == "__main__":
    n = 10000
    array = [random.randint(0, 10000) for _ in range(n)]

    arr_bubble = array.copy()
    arr_quick = array.copy()
    arr_radix = array.copy()

    p1 = Process(target=bubble_sort, args=(arr_bubble,), name="BubbleSort")
    p2 = Process(target=quick_sort, args=(arr_quick,), name="QuickSort")
    p3 = Process(target=radix_sort, args=(arr_radix,), name="RadixSort")

    p1.start()
    p2.start()
    p3.start()
    
    p1.join()
    p2.join()
    p3.join()

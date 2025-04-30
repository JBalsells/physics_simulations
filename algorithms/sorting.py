import time
import random

def bubble_sort(arr):
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

def radix_sort(arr):
    if len(arr) == 0:
        return arr
    max_num = max(arr)
    exp = 1
    result = arr.copy()
    while max_num // exp > 0:
        result = counting_sort(result, exp)
        exp *= 10
    return result

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

if __name__ == "__main__":
    n = 5000
    array = [random.randint(0, 10000) for _ in range(n)]

    # Quick Sort
    start = time.time()
    quick_sorted_array = quick_sort(array)
    end = time.time()
    print(f"⏱ Quick Sort time: {end - start:.6f} seconds")

    # Bubble Sort
    start = time.time()
    bubble_sorted_array = bubble_sort(array)
    end = time.time()
    print(f"🐢 Bubble Sort time: {end - start:.6f} seconds")

    # Radix Sort
    start = time.time()
    radix_sorted_array = radix_sort(array)
    end = time.time()
    print(f"⚡ Radix Sort time: {end - start:.6f} seconds")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
image = np.array([[1, 2, 3, 4, 5], [4, 5, 6, 7, 8], [7, 8, 9, 1, 2], [3, 5, 2, 3, 4], [1, 6, 7, 5, 6]])
kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
kernel2 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
def padding(array, p):
    new_image = np.pad(array, pad_width=p, mode='constant', constant_values=0)
    return new_image
padded_image = padding(image,1)
print(padded_image)
def filtering(old_array, new_array, array2):
    kernel_height, kernel_width=array2.shape
    array_height, array_width=old_array.shape
    result=np.zeros((array_height, array_width))
    for x in range(array_height):
        for y in range(array_width):
            result[x,y]=np.sum(new_array[y:y+kernel_height, x:x+kernel_width]*array2)
    return result
filtered_image = filtering(image, padded_image, kernel)
print(filtered_image)
def relu(array):
    return np.maximum(0, array)
relu_output=relu(filtered_image)
print(relu_output)
def max_pool(array, pool_size=2, stride=1):
    H, W = array.shape
    out_H = (H - pool_size) // stride + 1
    out_W = (W - pool_size) // stride + 1
    result1 = np.zeros((out_H, out_W))
    for x in range(out_H):
        for y in range(out_W):
            window = array[x * stride: x * stride + pool_size, y * stride: y * stride + pool_size]
            result1[x, y] = np.max(window)
    return result1
pooled_matrix=max_pool(relu_output,2,1)
print(pooled_matrix)
padded_image2=padding(pooled_matrix, 1)
filtered_image2=filtering(pooled_matrix, padded_image2, kernel2)
relu_output2=relu(filtered_image2)
pooled_matrix2=max_pool(relu_output2,2,1)
print(pooled_matrix2)
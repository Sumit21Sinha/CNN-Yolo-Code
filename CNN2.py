import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define the image and kernel
image = np.array([[1, 2, 3, 4, 5],
                  [4, 5, 6, 7, 8],
                  [7, 8, 9, 1, 2],
                  [3, 5, 2, 3, 4],
                  [1, 6, 7, 5, 6]])

kernel = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]])


# Function to pad the image
def padding(array, p):
    new_image = np.pad(array, pad_width=p, mode='constant', constant_values=0)
    return new_image


# Pad the image
new_image = padding(image, 1)
print("Padded Image:\n", new_image)


# Function to apply filtering (convolution)
def filtering(image, kernel):
    # Get dimensions
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Calculate output dimensions
    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1

    # Initialize the result array
    result = np.zeros((output_height, output_width))

    # Perform convolution
    for i in range(output_height):
        for j in range(output_width):
            # Element-wise multiplication and sum
            result[i, j] = np.sum(image[i:i + kernel_height, j:j + kernel_width] * kernel)

    return result


# Apply filtering
result = filtering(new_image, kernel)
print("Filtered Result:\n", result)
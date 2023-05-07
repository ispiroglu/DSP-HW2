import cv2
import numpy as np

# Load the source and kernel images
source_image = cv2.imread('/home/xpirr/workspace/python/DSP/HW2/Resim6_8.jpg', cv2.IMREAD_GRAYSCALE)
kernel_image = cv2.imread('/home/xpirr/workspace/python/DSP/HW2/EvrenIspiroglu.py', cv2.IMREAD_GRAYSCALE)

# Convert the kernel image to a numpy array of type np.float32
kernel_image = np.array(kernel_image, dtype=np.float32)

# Normalize the kernel image so that its sum is 1
kernel_image = kernel_image / np.sum(kernel_image)

# Pad the source image with zeros
padded_image = np.pad(source_image, ((1, 1), (1, 1)), 'constant')

# Compute the output image using 2D convolution
output_image = np.zeros_like(source_image)
for i in range(1, padded_image.shape[0]-1):
    for j in range(1, padded_image.shape[1]-1):
        patch = padded_image[i-1:i+2, j-1:j+2]
        output_image[i-1, j-1] = np.sum(patch * kernel_image)

# Display the input and output images
cv2.imshow('Source Image', source_image)
cv2.imshow('Kernel Image', kernel_image)
cv2.imshow('Output Image', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

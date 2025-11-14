import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load binary image
img = cv2.imread('3.jpg', cv2.IMREAD_GRAYSCALE)

# Ensure it's binary (threshold)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Step 1: Remove small bright specks (Opening)
kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

# Step 2: Fill narrow dark cracks (Closing)
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)

# Show results
plt.figure(figsize=(10,6))
plt.subplot(1,3,1), plt.imshow(binary, cmap='gray'), plt.title('Original')
plt.subplot(1,3,2), plt.imshow(opened, cmap='gray'), plt.title('After Opening')
plt.subplot(1,3,3), plt.imshow(closed, cmap='gray'), plt.title('After Opening + Closing')
plt.show()

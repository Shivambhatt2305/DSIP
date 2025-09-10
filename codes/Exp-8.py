import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_smoothing_filter(image, kernel_size):
    # Apply smoothing filter to the image
    smoothed_image = cv2.blur(image, (kernel_size, kernel_size))
    return smoothed_image

def apply_sharpening_filter(image):
    # Create a sharpening kernel
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    # Apply the sharpening kernel to the image
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image

# Load the input image
image_path = "D:\DSIP\codes\ex1_2.png"
input_image = cv2.imread(image_path)

# Apply smoothing filter
smoothed_image = apply_smoothing_filter(input_image, kernel_size=5)

# Apply sharpening filter
sharpened_image = apply_sharpening_filter(input_image)

# Display the original image and the filtered images side by side
combined_image = np.hstack((input_image, smoothed_image, sharpened_image))
cv2.imshow("Original | Smoothed | Sharpened", combined_image)
cv2.waitKey(0)

# Save the filtered images (optional)
smoothed_path = 'smoothed_image.jpg'
sharpened_path = 'sharpened_image.jpg'
cv2.imwrite(smoothed_path, smoothed_image)
cv2.imwrite(sharpened_path, sharpened_image)
print(f"Smoothed image saved at: {smoothed_path}")
print(f"Sharpened image saved at: {sharpened_path}")

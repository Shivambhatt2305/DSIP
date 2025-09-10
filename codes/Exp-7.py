import cv2
import numpy as np

# List of image filenames
image_filenames = ['ex1_1.png', 'ex1_2.png', 'ex1_3.png', 'ex1_4.png', 'ex1_5.png']

# Function to perform image negation
def image_negation(input_image):
    return 255 - input_image

# Function to perform image thresholding
def image_thresholding(input_image, threshold_value=127):
    _, thresholded_image = cv2.threshold(input_image, threshold_value, 255, cv2.THRESH_BINARY)
    return thresholded_image

# Function to perform image gamma correction
def image_gamma_correction(input_image, gamma=0.6):
    # Normalize, apply gamma, then scale back to [0,255]
    normalized = input_image / 255.0
    gamma_corrected = np.power(normalized, gamma) * 255.0
    return np.uint8(gamma_corrected)

# Loop through all images and apply operations
for i, filename in enumerate(image_filenames, start=1):
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Error: Could not open or find the image '{filename}'")
        continue

    # Apply operations
    negated = image_negation(image)
    thresholded = image_thresholding(image)
    gamma_corrected = image_gamma_correction(image)

    # Display the results
    cv2.imshow(f'Original Image {i}', image)
    cv2.imshow(f'Negated Image {i}', negated)
    cv2.imshow(f'Thresholded Image {i}', thresholded)
    cv2.imshow(f'Gamma Corrected Image {i}', gamma_corrected)

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
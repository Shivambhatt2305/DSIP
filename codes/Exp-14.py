import cv2
import numpy as np
from google.colab.patches import cv2_imshow
# Load the image
image = cv2.imread('reference.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Gaussian blur to reduce noise
image_blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Apply Canny edge detection
edges = cv2.Canny(image_blurred, threshold1=30, threshold2=100)

# Create a black image with the same size as the original image
boundary_image = np.zeros_like(image)

# Copy the detected edges to the boundary image
boundary_image[edges > 0] = 255

# Display the original image and the boundary image
cv2_imshow(image)
cv2_imshow(boundary_image)

# Save the boundary image
cv2.imwrite('boundary_image.jpg', boundary_image)

# Wait for a key press and then close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()

import numpy as np
import matplotlib.pyplot as plt
import cv2

def arnold_cat_map(image, iterations):
    height, width = image.shape[:2]
    assert height == width, "Height and width must be equal"
    
    # Create an empty numpy array with the same shape as the image
    new_image = np.zeros((height, width, 3), dtype=image.dtype)
    
    # Define the Arnold Cat Map transformation matrix
    A = np.array([[1, 1], [1, 2]])
    
    # Loop through each pixel in the image
    for i in range(height):
        for j in range(width):
            # Apply the Arnold Cat Map transformation
            ij = np.array([i, j])
            ij = np.dot(A, ij) % height
            new_image[ij[0], ij[1]] = image[i, j]
    
    return new_image

# Load the image
image = cv2.imread('image.jpg')

# Get the minimum dimension of the image
min_dim = min(image.shape[:2])

# Crop the image to a square shape
image = image[:min_dim, :min_dim]

# Display the original image
plt.figure()
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Apply the Arnold Cat Map algorithm
iterations = 1
shuffled_image = arnold_cat_map(image, iterations)

# Save the shuffled image
cv2.imwrite('shuffled_image.jpg', shuffled_image)

plt.show()

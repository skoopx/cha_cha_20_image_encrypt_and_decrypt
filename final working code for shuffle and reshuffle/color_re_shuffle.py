import numpy as np
import matplotlib.pyplot as plt
import cv2

def inverse_arnold_cat_map(image, iterations):
    height, width = image.shape[:2]
    assert height == width, "Height and width must be equal"
    
    # Create an empty numpy array with the same shape as the image
    new_image = np.zeros((height, width, 3), dtype=image.dtype)
    
    # Define the inverse Arnold Cat Map transformation matrix
    A_inv = np.array([[2, -1], [-1, 1]])
    
    # Loop through each pixel in the image
    for i in range(height):
        for j in range(width):
            # Apply the inverse Arnold Cat Map transformation
            ij = np.array([i, j])
            ij = np.dot(A_inv, ij) % height
            new_image[ij[0], ij[1]] = image[i, j]
    
    return new_image

# Load the shuffled image
shuffled_image = cv2.imread('decrypted.jpg')

# Define the number of iterations
iterations = 1

# Apply the inverse Arnold Cat Map algorithm
reshuffled_image = inverse_arnold_cat_map(shuffled_image, iterations)

# Display the reshuffled image
plt.figure()
plt.title('Reshuffled Image')
plt.imshow(cv2.cvtColor(reshuffled_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()

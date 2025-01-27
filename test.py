import matplotlib.pyplot as plt
import numpy as np

# Create a test image
image = np.random.rand(128, 128)

# Plot the image
plt.imshow(image, cmap="gray")
plt.title("Test Image")
plt.axis("off")
plt.show()
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image



data = np.load('/media/jay/apple/uw_depth_lizard_data/features-saved/new_probability_0422.npy')
print(np.shape(data))

# resized_channel_1 = cv2.resize(data[:, :, 0], (31, 19), interpolation=cv2.INTER_LINEAR)
# resized_channel_2 = cv2.resize(data[:, :, 1], (31, 24), interpolation=cv2.INTER_LINEAR)

plt.figure(figsize=(12, 6))

# RGB Image
plt.subplot(1, 2, 1)
plt.imshow(data[:,:,0])
plt.title("distance map")

# Depth Map
plt.subplot(1, 2, 2)
plt.imshow(data[:,:,1], cmap='plasma')
plt.colorbar()
plt.title("probability map")

plt.tight_layout()
plt.show()
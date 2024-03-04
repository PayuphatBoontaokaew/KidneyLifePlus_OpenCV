# final code
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('/Users/payuphatboontaokaew/Documents/GitHub/KidneyLifePlus/PDR2.jpg')
img = cv2.resize(img, (640, 640))

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

lower = np.array([10, 10, 10], dtype=np.uint8)
upper = np.array([250, 250, 250], dtype=np.uint8)

mask = cv2.inRange(img, lower, upper)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

cont_img = img.copy()
cv2.drawContours(cont_img, contours, -1, (0, 255, 0), 3)  # Use (0, 255, 0) for green color

c = max(contours, key=cv2.contourArea)

(x, y), radius = cv2.minEnclosingCircle(c)
center = (int(x), int(y))

mask = np.zeros_like(img_gray)

cv2.circle(mask, center, int(radius), (255, 255, 255), thickness=-1)  # Filling with white color

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

result = cv2.bitwise_and(img_gray, mask)

plt.figure(figsize=(12, 6))
plt.subplot(131)
plt.imshow(img_rgb)
plt.title('Original Image')

plt.subplot(132)
plt.imshow(cont_img)
plt.title('Contours')

plt.subplot(133)
plt.imshow(result, cmap='gray')
plt.title('Result (Grayscale)')
plt.show()

# # Save the result image to your Colab environment
# cv2.imwrite('/content/Result_Grayscale.jpg', result)

# # Provide a download link for the saved image
# from google.colab import files
# files.download('/content/Result_Grayscale.jpg')
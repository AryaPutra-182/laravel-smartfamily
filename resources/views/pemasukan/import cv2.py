import cv2
import numpy as np
from matplotlib import pyplot as plt

# Fungsi untuk menampilkan gambar
def show_images(images, titles):
    plt.figure(figsize=(15, 5))
    for i in range(len(images)):
        plt.subplot(1, len(images), i+1)
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.title(titles[i])
        plt.axis('off')
    plt.show()

# Baca gambar input
image_path = 'C:\Users\arya2\Downloads\WhatsApp Image 2024-12-09 at 21.07.11_31b1fe1e.jpg'  # Ganti dengan path gambar Anda
image = cv2.imread(image_path)

# 1. Noise Reduction (Denoising)
denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

# 2. Image Sharpening
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # Kernel untuk sharpening
sharpened = cv2.filter2D(denoised, -1, kernel)

# 3. Contrast Adjustment (Histogram Equalization)
# Ubah ke YUV untuk meningkatkan kontras
yuv = cv2.cvtColor(sharpened, cv2.COLOR_BGR2YUV)
yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # Histogram equalization pada channel Y
contrast_adjusted = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

# Tampilkan hasil
show_images(
    [image, denoised, sharpened, contrast_adjusted],
    ["Original", "Denoised", "Sharpened", "Contrast Adjusted"]
)

# Simpan hasil
cv2.imwrite('denoised.jpg', denoised)
cv2.imwrite('sharpened.jpg', sharpened)
cv2.imwrite('contrast_adjusted.jpg', contrast_adjusted)
print("Hasil gambar telah disimpan.")

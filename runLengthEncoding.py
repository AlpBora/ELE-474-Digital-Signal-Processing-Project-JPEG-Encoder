import cv2
import matplotlib.pyplot as plt
from collections import defaultdict

def run_length_encode(image):
    encoded_pixels = []
    height, width = image.shape
    count = 1
    for i in range(height):
        for j in range(width - 1):
            if image[i, j] == image[i, j + 1]:
                count += 1
            else:
                encoded_pixels.append(image[i, j])
                encoded_pixels.append(count)
                count = 1
        encoded_pixels.append(image[i, width - 1])
        encoded_pixels.append(count)
        count = 1
    return encoded_pixels

# Read the image as grayscale
image = cv2.imread("lena_gray.bmp", cv2.IMREAD_GRAYSCALE)

# Step 1: Run-length encoding
encoded_pixels = run_length_encode(image)

# Step 2: Create a frequency table
frequency_table = defaultdict(int)

# Step 3: Update frequencies based on symbol occurrences
i = 0
while i < len(encoded_pixels):
    symbol = encoded_pixels[i]
    count = int(encoded_pixels[i+1])
    frequency_table[symbol] += count
    i += 2

# Extract symbols and frequencies from the frequency table
symbols = list(frequency_table.keys())
frequencies = list(frequency_table.values())

# Create a histogram graph
plt.bar(symbols, frequencies)
plt.xlabel('Symbols')
plt.ylabel('Frequencies')
plt.title('Histogram')
plt.show()
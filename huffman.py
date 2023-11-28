from collections import defaultdict
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import heapq
from runLengthEncoding import run_length_encode

class HuffmanNode:
    def __init__(self, value, frequency):
        self.value = value
        self.frequency = frequency
        self.left = None
        self.right = None
    
    def __lt__(self, other):
        return self.frequency < other.frequency

def build_huffman_tree(frequencies):
    # Step 2: Create leaf nodes
    leaf_nodes = [HuffmanNode(pixel_value, frequency) for pixel_value, frequency in enumerate(frequencies) if frequency > 0]

    # Step 3: Initialize priority queue
    priority_queue = leaf_nodes.copy()
    heapq.heapify(priority_queue)

    # Step 5: Build Huffman tree
    while len(priority_queue) > 1:
        # Dequeue two nodes with lowest weights
        node1 = heapq.heappop(priority_queue)
        node2 = heapq.heappop(priority_queue)

        # Create new internal node
        internal_node = HuffmanNode(None, node1.frequency + node2.frequency)
        internal_node.left = node1
        internal_node.right = node2

        # Enqueue new internal node
        heapq.heappush(priority_queue, internal_node)

    # Return the root node of the Huffman tree
    return priority_queue[0]

def traverse_huffman_tree(node, code, code_table):
    if node is None:
        return

    # Leaf node
    if node.value is not None:
        code_table[node.value] = code
        return

    # Internal node
    traverse_huffman_tree(node.left, code + '0', code_table)
    traverse_huffman_tree(node.right, code + '1', code_table)

image = cv.imread("lena_gray.bmp", cv.IMREAD_GRAYSCALE) 

# Run-length encoding
encoded_pixels = run_length_encode(image)

# Create a frequency table
frequency_table = defaultdict(int)

# Update frequencies based on symbol occurrences
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

huffman_tree = build_huffman_tree(frequencies)

# Step 6: Traverse tree and assign codes
code_table = {}
traverse_huffman_tree(huffman_tree, '', code_table)

# Print the Huffman codes
for pixel_value, code in code_table.items():
    print("Pixel Value:", pixel_value, "Code:", code)

# Initialize variables for mismatch count and total symbols
mismatch_count = 0
total_symbols = len(code_table)

# Validate the Huffman codes
for symbol, code in code_table.items():
    frequency = frequency_table[symbol]
    code_length = len(code)
    for other_symbol, other_code in code_table.items():
        other_frequency = frequency_table[other_symbol]
        other_code_length = len(other_code)
        if other_frequency < frequency and other_code_length > code_length:
            print("Mismatch for symbol:", symbol)
            mismatch_count += 1
            break

# Calculate mismatch percentage
mismatch_percentage = (mismatch_count / total_symbols) * 100

# Print the total mismatch count and percentage
print("Total mismatch count:", mismatch_count)
print("Mismatch percentage:", mismatch_percentage, "%")
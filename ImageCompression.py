from PIL import Image
import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import defaultdict
import heapq

class ImageCompression :
    def __init__(self,image):
        self.image = image
        self.QMatrix= np.zeros((8,8))
        self.QMatrix = [[16,11,10,16,24,40,51,61],
                               [12,12,14,19,26,58,60,55],
                               [14,13,16,24,40,57,69,56],
                               [14,17,22,29,51,87,80,62],
                               [18,22,37,56,68,109,103,77],
                               [24,35,55,64,81,104,113,92],
                               [49,64,78,87,103,121,120,101],
                               [72,92,95,98,112,100,103,99]]
        self.code_table = {}

    def __getitem__(self):
        image_copy = self.getImage()
        zigzag_image = self.takeDCT(image=image_copy) # takes dct, makes quaintizaion and performs zigzag scanning
        encoded_pixels = self.run_length_encode(pixels=zigzag_image) 
        frequencies = self.extract_frequencies(encoded_pixels)
        huffman_tree = self.build_huffman_tree(frequencies)
        self.traverse_huffman_tree(huffman_tree, '', self.code_table)
        return self.code_table

    def getImage(self):
        image_copy = self.image.copy()
        height, width, _= self.image.shape
        print(height,width)
        return image_copy
    
    def ZigZag_Scan(self, matrix):
        rows, cols = len(matrix), len(matrix[0])
        result = []
        
        for i in range(rows + cols - 1):
            if i % 2 == 0:  # Çift indeksli satırlar (aşağı doğru)
                for j in range(min(i, rows - 1), max(0, i - cols + 1) - 1, -1):
                    result.append(matrix[j][i - j])
            else:  # Tek indeksli satırlar (yukarı doğru)
                for j in range(max(0, i - cols + 1), min(i, rows - 1) + 1):
                    result.append(matrix[j][i - j])
        
        return result

    def takeDCT(self,image):
        avarage_num=128
        x_count, y_count = 1 , 1
        finish = 0
        for i in range (0,511):
            for j in range (0,511):
                image[i, j] -=avarage_num
        
        
        encoding_image = np.zeros((512,512))
        #Dikkat eksenler farklı
        while finish != 1:
            matrix = image[(x_count-1)*8:x_count*8,(y_count-1)*8:y_count*8]
            dct_matrix = np.zeros((8,8))
            for u in range(8):
                for v in range(8):
                    sum_dct = 0.0
                    for x in range(8):
                        for y in range(8):
                            cu = 1.0 if u == 0 else (math.sqrt(2) / 2.0)
                            cv = 1.0 if v == 0 else (math.sqrt(2) / 2.0)
                            
                            cos_term = math.cos(((2*x+1)*u*math.pi)/(2*8))* math.cos(((2*y+1)*v*math.pi)/(2*8))
                            sum_dct += matrix[x, y] * cu * cv * cos_term * 0.25

                    dct_matrix[u, v] = sum_dct
                    Qimage = np.round(dct_matrix[u, v] / self.QMatrix)
                    zigzag_image = self.ZigZag_Scan(Qimage)
                    print(zigzag_image)
            #     encoding_image[(x_count-1)*8:x_count*8,(y_count-1)*8:y_count*8] = zigzag_image
            if(y_count == 64 and x_count != 64):
                y_count = 1
                x_count +=1
            elif (y_count == 64 and x_count == 64):
                finish = 1
            else :
                y_count +=1
        # print(zigzag_image.shape)
        # print(zigzag_image)
        # plt.imshow(zigzag_image, cmap='gray')
        # plt.axis("off")
        # plt.show()

    def run_length_encode(self,pixels):
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
    
    def extract_frequencies(self,encoded_pixels):

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
        return frequencies

    def build_huffman_tree(frequencies):
        # Create leaf nodes
        leaf_nodes = [(pixel_value, frequency) for pixel_value, frequency in enumerate(frequencies) if frequency > 0]

        # Initialize priority queue
        priority_queue = leaf_nodes.copy()
        heapq.heapify(priority_queue)

        # Build Huffman tree
        while len(priority_queue) > 1:
            # Dequeue two nodes with lowest weights
            node1 = heapq.heappop(priority_queue)
            node2 = heapq.heappop(priority_queue)

            # Create new internal node
            internal_node = (None, node1[1] + node2[1], node1, node2)

            # Enqueue new internal node
            heapq.heappush(priority_queue, internal_node)

        # Return the root node of the Huffman tree
        return priority_queue[0]

    def traverse_huffman_tree(node, code_table):
        def traverse(node, code=''):
            if node is None:
                return

            # Leaf node
            if node[0] is not None:
                code_table[node[0]] = code
                return

            # Internal node
            traverse(node[2], code + '0')
            traverse(node[3], code + '1')

        # Call the nested helper function
        traverse(node)

    
    
if __name__ == "__main__":
    image = cv.imread("lena_gray.bmp", cv.IMREAD_GRAYSCALE)
    codes = ImageCompression(image)
    print(codes)

    # Print the Huffman codes
    for pixel_value, code in codes:
        print("Pixel Value:", pixel_value, "Code:", code)


#cv.waitKey(0)
#cv.destroyAllWindows()

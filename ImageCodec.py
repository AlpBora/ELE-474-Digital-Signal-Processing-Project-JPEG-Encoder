from PIL import Image
import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import Counter
import time
from tqdm import tqdm

class Node:
    def __init__(self, symbol=None, frequency=0, left=None, right=None):
        self.symbol = symbol
        self.frequency = frequency
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.frequency < other.frequency

class ImageCodec :
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

    def run(self):
        image = self.getImage()
        encoded_image, huffman_codes = self.encode_image(image)
        print(encoded_image)
        decoded_image = self.decode_image(encoded_image,huffman_codes)
        print(decoded_image)
        return decoded_image
    

    def getImage(self):
        image_copy = self.image.copy()
        height, width= self.image.shape
        print(height,width)
        print(self.image.dtype)
        return image_copy
    

    def get_8x8_block(self, image, x_count, y_count):
        return image[(x_count - 1) * 8:x_count * 8, (y_count - 1) * 8:y_count * 8]


    def takeDCT(self,matrix): #MATLAB dc2 function formula
        dct_matrix = np.zeros((8,8))
        for u in range(8):
            for v in range(8):
                sum_dct = 0.0
                cu = 1/math.sqrt(8) if u == 0 else (math.sqrt(2/8))
                cv = 1/math.sqrt(8) if v == 0 else (math.sqrt(2/8))
                for x in range(8):
                    for y in range(8):
                        cos_term = math.cos(((2*x+1)*u*math.pi)/(2*8))* math.cos(((2*y+1)*v*math.pi)/(2*8))
                        sum_dct += matrix[x, y] * cu * cv * cos_term

                dct_matrix[u, v] = sum_dct
        return dct_matrix
                

    def Quantization(self,matrix):
        return np.round(matrix / self.QMatrix)
    

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


    def build_huffman_tree(self,frequencies):
        heap = [Node(symbol=symbol, frequency=frequency) for symbol, frequency in frequencies.items()]
        while len(heap) > 1:
            heap.sort()
            left = heap.pop(0)
            right = heap.pop(0)
            internal_node = Node(frequency=left.frequency + right.frequency, left=left, right=right)
            heap.append(internal_node)
        return heap[0]


    def generate_huffman_codes(self, node, code="", mapping=None):
        if mapping is None:
            mapping = {}
        if node.symbol is not None:
            mapping[node.symbol] = code
        if node.left is not None:
            self.generate_huffman_codes(node.left, code + "0", mapping)
        if node.right is not None:
            self.generate_huffman_codes(node.right, code + "1", mapping)
        return mapping


    def huffman_encode(self, vector):
        frequencies = dict(Counter(vector))
        root = self.build_huffman_tree(frequencies)
        codes = self.generate_huffman_codes(root)
        encoded_vector = ''.join(codes[num] for num in vector)
        return encoded_vector, codes
    
    
    def encode_image(self, image):
        print("Starting Image Encoding")
        image = image.astype('double')
        average_num = 128
        x_count, y_count = 1, 1
        finish = 0
        progress_bar = tqdm(total=64*64, desc="Encoding Image")

        for i in range(0, 511):
            for j in range(0, 511):
                image[i, j] -= average_num

        encoded_image = []
        all_huffman_codes = {}
        
        #Dikkat eksenler farklı
        while finish != 1:
            matrix = self.get_8x8_block(image, x_count, y_count)
            dct_matrix = self.takeDCT(matrix)
            #print("DCT",dct_matrix)
            
            Qimage = self.Quantization(dct_matrix)
            #print("Q",Qimage)
            zigzag_scan = self.ZigZag_Scan(Qimage)
            zigzag_scan = list(map(int, zigzag_scan))
            #print("Zigzag",zigzag_scan)
            encoded_vector, huffman_codes = self.huffman_encode(zigzag_scan)
            #print("Huffman",encoded_vector)

            progress_bar.update(1)

            encoded_image.append(encoded_vector)
            # Her bir huffman_codes dictionary'sini birleştir
            for symbol, code in huffman_codes.items():
                if symbol not in all_huffman_codes:
                    all_huffman_codes[symbol] = code
            
            if y_count == 64 and x_count != 64:
                y_count = 1
                x_count += 1
            elif y_count == 64 and x_count == 64:
                finish = 1
            else:
                y_count += 1
        progress_bar.close()
        print("Image Encoding Finished")
        return np.array(encoded_image), all_huffman_codes

    
    def huffman_decode(self, encoded_vector, codes):
        reverse_codes = {code: symbol for symbol, code in codes.items()}
        current_code = ""
        decoded_vector = []

        for bit in encoded_vector:
            current_code += str(bit)
            if current_code in reverse_codes:
                symbol = reverse_codes[current_code]
                decoded_vector.append(symbol)
                current_code = ""

        return decoded_vector

    def huffmanDecode (self,text,dictionary):
        res = ""
        while text:
            for k in dictionary:
                if text.startswith(k):
                    res += dictionary[k]
                    text = text[len(k):]
        return res
    
    def huffman_decoding_func(self,data, tree):
        
        dict = self.get_codes(tree.root)
        reversed_dict = {}
        for value, key in dict.items():
            reversed_dict[key] = value
        start_index = 0
        end_index = 1
        max_index = len(data)
        s = ''

        while start_index != max_index:
            if data[start_index : end_index] in reversed_dict:
                s += reversed_dict[data[start_index : end_index]]
                start_index = end_index
            end_index += 1

        return s
    
    def get_codes(self,root):
        if root is None:
            return {}
        frequency, characters = root.value
        char_dict = dict([(i, '') for i in list(characters)])

        left_branch = self.get_codes(root.get_left_child())

        for key, value in left_branch.items():
            char_dict[key] += '0' + left_branch[key]

        right_branch = get_codes(root.get_right_child())

        for key, value in right_branch.items():
            char_dict[key] += '1' + right_branch[key]

        return char_dict

    def inverse_zigzag_scan(self, vector, rows, cols):
        matrix = np.zeros((rows, cols))
        index = 0

        for i in range(rows + cols - 1):
            if i % 2 == 1:  # Odd diagonal
                for row in range(max(0, i - cols + 1), min(i, rows - 1) + 1):
                    if index < len(vector):
                        matrix[row, i - row] = vector[index]
                        index += 1
            else:  # Even diagonal
                for row in range(min(i, rows - 1), max(0, i - cols + 1) - 1, -1):
                    if index < len(vector):
                        matrix[row, i - row] = vector[index]
                        index += 1

        return matrix
    

    def inverseQuantization(self,Qimage):
        return np.round(Qimage * self.QMatrix)


    def inverseDCT(self,dct_matrix):
        idct_matrix = np.zeros((8, 8))

        for x in range(8):
            for y in range(8):
                sum_idct = 0.0
                for u in range(8):
                    for v in range(8):
                        cu = 1/math.sqrt(8) if u == 0 else (math.sqrt(2/8))
                        cv = 1/math.sqrt(8) if v == 0 else (math.sqrt(2/8))
                        cos_term = math.cos(((2*x+1)*u*math.pi)/(2*8))* math.cos(((2*y+1)*v*math.pi)/(2*8))
                        sum_idct += dct_matrix[u, v] * cu * cv * cos_term

                idct_matrix[x, y] = sum_idct

        return idct_matrix
    

    def decode_image(self,image,huffman_codes):
        print("Starting Image Decoding")
        average_num = 128
        progress_bar = tqdm(total=64*64, desc="Decoding Image")
        decoded_image = []
        decoded_vector = self. huffmanDecode(image,huffman_codes)
        block_size = 8
        num_blocks = len(decoded_vector) // (block_size * block_size)
        
        for i in range(num_blocks):    
            start_index = i * block_size * block_size
            end_index = start_index + block_size * block_size
            block_vector = decoded_vector[start_index:end_index]
            inverse_zigzag_scan = self.inverse_zigzag_scan(block_vector,block_size,block_size)
            unquantized_matrix = self.inverseQuantization(inverse_zigzag_scan)
            idct_matrix = self.inverseDCT(unquantized_matrix)

            progress_bar.update(1)
            decoded_image.append(idct_matrix)
        
        for i in range(0, len(decoded_image)):
            decoded_image[i] += average_num

        progress_bar.close()
        print("Image Encoding Finished")
        #decoded_image = decoded_image.astype(np.uint8)

        return np.array(decoded_image)
    


if __name__ == "__main__":

    image = cv.imread("lena_gray.bmp", cv.IMREAD_GRAYSCALE)
    codec_instance = ImageCodec(image)
    decoded_image = codec_instance.run()
    plt.imshow(decoded_image)
    plt.show()
    
    

    

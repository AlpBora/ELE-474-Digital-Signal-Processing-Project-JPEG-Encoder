from PIL import Image 
import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
class Dct :
    def __init__(self,image):
        self.image = image

    def __getitem__(self):
        image = self.getCopy()
        dct =self.takeDCT(image)
        return dct

    def getCopy(self):
        image_copy = self.image.copy()
        print(self.image.shape)
        height, width = self.image.shape
        print(height,width)
        return image_copy

    
    def takeDCT(self,image):
        avarage_num=128
        x_count, y_count = 1 , 1
        finish = 0
        for i in range (0,511):
            for j in range (0,511):
                image[i, j] -=avarage_num
        
        dct_image = np.zeros((512,512))
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
            dct_image[(x_count-1)*8:x_count*8,(y_count-1)*8:y_count*8] = dct_matrix
            if(y_count == 64 and x_count != 64):
                y_count = 1
                x_count +=1
            elif (y_count == 64 and x_count == 64):
                finish = 1
            else :
                y_count +=1
        print(dct_image.shape)
        print(dct_image)
        plt.imshow(np.log(1 + np.abs(dct_image)))
        plt.axis('off')
        plt.title('Lena\'nın 2D DCT\'si')
        plt.show()
        return dct_image
    
if __name__ == "__main__":
    image = cv.imread("lena_gray.bmp", cv.IMREAD_GRAYSCALE)
    dct_image = Dct(image)
    

    # to convert gray scale
    #gray_image = cv.imread("lena.bmp",cv.IMREAD_GRAYSCALE)
    #gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
   
    
    

 
    


#cv.waitKey(0)
#cv.destroyAllWindows()

import numpy as np
import cv2 as cv
import math



class Image:
    def __init__(self,filename=" "):
            self.filename= filename
            self.array = np.asarray(cv.imread(filename), dtype=np.uint8)
            self.width= self.array.shape[0]
            self.height= self.array.shape[1]
            #If RGB or RGBA
            if len(self.array.shape)>=3:
                self.channels=self.array.shape[2]
            #grayscale img    
            else:
                self.channels=1


    def save_image(self,outfilename) :
        cv.imwrite(outfilename, self.array)

    #Getters
    def get_array(self):
        return self.array

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height
    
    def get_channels(self):
        return self.channels
    
    #Setter
    def set_array(self, arr):
        self.array=arr
        self.width=arr.shape[0]
        self.height=arr.shape[1]
        if len(self.array.shape)>=3:
                self.channels=self.array.shape[2]
            #grayscale img    
        else:
                self.channels=1


    #Changes brigthness. If constant is less than zero darkens the image.
    # Else brightness the image. 
    def change_brightness(self,constant):
    
        
       if -255<=constant<=255: 

          for i in range(self.width):
              for j in range(self.height):
                  for k in range(self.channels):
                    if 0<=self.array[i][j][k] + constant <=255 :
                        self.array[i][j][k]+=constant
                    elif self.array[i][j][k] + constant >255 :
                        self.array[i][j][k]=255  
                    elif self.array[i][j][k] + constant <0 :
                        self.array[i][j][k]=0 
       else: 
            print("The constant must be in between -255 and 255.")
       
        

    #Changes contrast. If alpha is between 1 and 0 lowers contrast.
    # Else increases contrast. 
    def change_contrast(self, alpha):
        if alpha>0: 
          for i in range(self.width):
              for j in range(self.height):
                  for k in range(self.channels):
                    if 0<=self.array[i][j][k] * alpha <=255 :
                        self.array[i][j][k]*=alpha
                    elif self.array[i][j][k] * alpha >255 :
                        self.array[i][j][k]=255  
                    elif self.array[i][j][k] * alpha <0 :
                        self.array[i][j][k]=0 
        else: 
            print("Alpha must be greater than 0")
        
        
    #Linear blending-> Icombined(x,y)=alpha* I0(x,y) + (1-alpha)*I1(x,y) 
    def blend(self,img2,alpha):
        img2_width=img2.width
        img2_height=img2.height
        img2_channels=img2.channels 

        #The width, height and number of channels must be equal on both images to blend
        if (0<alpha<1 and img2_width==self.width and img2_height==self.height and img2_channels==self.channels): 
          for i in range(self.width):
              for j in range(self.height):
                  for k in range(self.channels):
                      self.array[i][j][k]= alpha* self.array[i][j][k] + (1-alpha) * img2.array[i][j][k]
                      
                    
        else: 
            print("Error.")

        

    #if I(x,y)<threshold -> I(x,y)=0 else I(x,y)=255
    def black_and_white(self,threshold=127):
        #If rgb or rgba first convert to grayscale
        if self.channels==3:
            org_image = cv.imread(self.filename)
            self.array = cv.cvtColor(org_image, cv.COLOR_BGR2GRAY)
            self.channels=1
        elif self.channels==4:
            org_image = cv.imread(self.filename)
            self.array = cv.cvtColor(org_image, cv.COLOR_BGRA2BGR)
        
        if(0<=threshold<=255):
            self.array= np.where(self.array<threshold,0,255)
        else:
            print("Invalid threshold value. Must be between 0 and 255.")
    



    #Apply box filter (blur) to image
    def blur(self,kernel_size):
        
        kernel=np.zeros((kernel_size,kernel_size))
        kernel_value= 1 / (kernel_size**2)

        #Initializing the box-filter kernel based on given size 
        for i in range(kernel_size):
            for j in range(kernel_size):
                kernel[i][j] = kernel_value
        
        #Apply kernel to image array
        self.apply_kernel(kernel,kernel_size)
 
    #Applies gaussian filter with given size and sigma value
    def gaussian_blur(self,kernel_size,sigma):
        r= kernel_size // 2
        kernel=np.zeros(shape=(kernel_size,kernel_size))
        sum=0

        #Initializing the gaussian kernel based on given size and sigma value
        for i in range(kernel_size):
            for j in range(kernel_size):
                kernel[i][j] = math.exp(-0.5 * ((((i-r)/sigma) **2.0) + (((j-r)/sigma)**2.0)))/ (2 * math.pi * sigma * sigma)
                sum+=kernel[i][j]
            
        #Normalizing the kernel so that the sum of kernel elements equals one 
        for i in range(kernel_size):
            for j in range(kernel_size):
                kernel[i][j]=kernel[i][j] / sum

        #Apply kernel to image array
        self.apply_kernel(kernel,kernel_size)

    #Applies a given kernel to the image array
    def apply_kernel(self,kernel,kernel_size):
        r =kernel_size // 2
        #Copy the array values to new array
        copy_arr=np.zeros((self.width,self.height,self.channels), dtype="uint8")
        np.copyto(copy_arr,self.array)

        for j in range(self.height):
            for i in range(self.width):
               for k in range(self.channels):
                   val=0
                   for x_dist in range(-r,r+1):
                       x_i=i+x_dist
                       k_x=r+ x_dist
                       for y_dist in range(-r,r+1):
                           y_i=j+y_dist
                           k_y=r+y_dist
                           if(x_i>0 and x_i<self.width and y_i>0 and y_i<self.height):
                               val+=copy_arr[x_i][y_i][k] * kernel[k_x][k_y]
                   if(0<=val<=255):            
                    self.array[i][j][k]=val
                   elif val>255:
                       self.array[i][j][k]=255
                   else:
                        self.array[i][j][k]=0
        


    #Detects edges on the image
    def find_edges(self):
         #Initialize kernels for edge detection

        #Sobel-x kernel for x-derivate of image
        sobel_x=np.array([[-1,0,1],
                        [-2,0,2],
                        [-1,0,1]  ])
        #Sobel-y kerne for y-derivative of image
        sobel_y=np.array([[1,2,1],
                        [0,0,0],
                        [-1,-2,-1]])


        #Copy the array values to new array for reusing self.array before x-derivative
        copy_arr=np.zeros((self.width,self.height,self.channels), dtype="uint8")
        np.copyto(copy_arr,self.array)

        #gaussian blur applied to remove noise
        self.gaussian_blur(3,1)

        self.apply_kernel(sobel_x,3)
        x_derivative=self.array
        #self.array converted to itself before applying sobel-x kernel
        self.array=copy_arr
        self.apply_kernel(sobel_y,3)
        y_derivative=self.array

        #array iniatlized for storing threshold values
        arr = np.zeros((self.width,self.height,self.channels), dtype=self.array.dtype)
        arr=((x_derivative**2) + (y_derivative**2))

        self.array= np.where(arr>=self.array,0,255)

        #img array converted to grayscale
        img_float32 = np.float32(self.array)
        self.array=cv.cvtColor(img_float32,cv.COLOR_BGR2GRAY)
        



            



    


  
  

    


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import os

# 1. change brightness
def brightenImage(img, b):
    img_1d = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
    resImg = [ [] for _ in range (len(img_1d))]
    index = 0 
    for i in img_1d:
        for j in i:
            if(j + b > 255):
                j = 255
            elif (j+b <0):
                j = 0
            else: j += b
            resImg[index].append(j)
        index +=1
    resImg = np.array(resImg)
    resImg = resImg.reshape(img.shape[0], img.shape[1], img.shape[2]).astype(np.uint8)        
    return resImg, "brighten"
    
# 2. change contrast
def constrastImage(img, a):
    img_1d = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
    resImg = [ [] for _ in range (len(img_1d))]
    index = 0
    for i in img_1d:
        for j in i:
            if(j * a > 255):
                j = 255
            else: j *= (a)
            resImg[index].append(j)
        index +=1
    resImg = np.array(resImg)
    resImg = resImg.reshape(img.shape[0], img.shape[1], img.shape[2]).astype(np.uint8)        
    return resImg, "contrast"

# 3. flip image 
def flipImage(img, type):
    resImg = img.copy()
    
    # flip by vertical
    if(type == 'Vertical'):
        index = len(img)-1
        for i in range (len(img)):
            resImg[i] = img[index]
            index -= 1
    
    # flip by horizontal 
    elif (type == 'Horizontal'):
        for i in range(img.shape[0]):
            index = img.shape[1]-1
            for j in range(img.shape[1]):
                resImg[i][j] = img[i][index]
                index -=1
    return resImg, 'flip'+type
          
# 4.1 change RGB image to gray
def grayImg(img):
    img_1d = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
    resImg = list()
    for i in img_1d:
        tmp = i[0] * 0.3 + i[1]*0.59+ i[2]*0.11
        if(tmp >255):
            tmp = 255
        resImg.append(tmp)
    resImg = np.array(resImg)
    resImg = resImg.reshape(img.shape[0], img.shape[1]).astype(np.uint8)        
    return resImg, "gray"     

# 4.2 change RGB image to sepia
def sepiaImage(img):
    img_1d = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
    resImg = [ [] for _ in range (len(img_1d))]
    index = 0
    for i in img_1d:
        tr = 0.393*i[0] + 0.769*i[1] + 0.189*i[2]
        tg = 0.349*i[0] + 0.686*i[1] + 0.168*i[2]  
        tb = 0.272*i[0] + 0.534*i[1] + 0.131*i[2]
        if(tr > 255):
            tr = 255
        if(tg > 255):
            tg = 255
        if(tb > 255):
            tb = 255
        resImg[index] = [tr,tg,tb]
        index +=1
    resImg = np.array(resImg)
    resImg = resImg.reshape(img.shape[0], img.shape[1], img.shape[2]).astype(np.uint8)        
    
    return np.array(resImg), "sepia"     

# 5.1 blur image
def blurImage(img):
    # Gaussian Kernel 3x3
    GaussianKernel = np.array([[1,2,1],
                               [2,4,2],
                               [1,2,1]])/16

    resImg = img.copy()
    resImg.fill(0)
    for i in range (1,img.shape[0]-1):
        for j in range (1,img.shape[1]-1):
            tmpArr = list()
            sum = 0
            for row in range(i-1, i+2):
                for col in range(j-1, j+2):
                    tmpArr.append(img[row][col])
            A = (np.array(tmpArr)).reshape(3,3,3)
            for row in range(0, 3):
                for col in range(0,3):
                    sum += A[row][col]*GaussianKernel[row][col]
            resImg[i][j] = sum
    return np.array(resImg), "blur"

# 5.2 sharpen image
def sharpenImage(img):
    Sharpen = np.array([[0 , -1,  0],
                        [-1,  5, -1],
                        [0 , -1,  0]])

    resImg = img.copy()
    resImg.fill(0)
    for i in range (1,img.shape[0]-1):
        for j in range (1,img.shape[1]-1):
            tmpArr = list()
            sum = 0
            for row in range(i-1, i+2):
                for col in range(j-1, j+2):
                    tmpArr.append(img[row][col])
            A = (np.array(tmpArr)).reshape(3,3,3)
            for row in range(0, 3):
                for col in range(0,3):
                    arr = A[row][col]
                    number = Sharpen[row][col]
                    mul = np.array([number*ai for ai in arr])
                    sum = sum + mul
            for k in range(len(sum)):
                if(sum[k]<0):
                    sum[k] =0
                elif(sum[k]>255):
                    sum[k] = 255
            resImg[i][j] = sum 
                
    return np.array(resImg), "sharpen"
            
# 6. crop image (at center)
def cropImage(img, height, width):
    row = int((img.shape[0] - height) / 2)
    col = int((img.shape[1] - width) / 2)
    resImg = list()
    for i in range(row, row+height):
        for j in range(col, col+width):
            resImg.append(img[i][j])
    resImg = np.array(resImg)
    resImg = np.reshape(resImg,(height, width, img.shape[2]))
    return resImg, 'crop'    

# 7. crop image by circle frame
def circleCropImage(img):
    resImg = img.copy()
    center = np.array([int(img.shape[0]/2), int(img.shape[1]/2)])
    
    radius = int(min(img.shape[0]/2, img.shape[1]/2))
    x, y = np.ogrid[:img.shape[0], :img.shape[1]]
    circleMask = (x - center[0])**2 + (y - center[1])**2 > radius**2
    resImg[circleMask] = np.zeros(img.shape[2])                 
    return np.array(resImg), 'circle'

# 8. main 
def main():
    
    # enter name of image
    while(True):
        imgName = input('Name of image: ')
        if(not os.path.isfile(imgName)):
            print('Error! Not find image')
        else: break
    
    # read image
    img = Image.open(imgName)
    img = np.array(img)
    
    # choose effect 
    choice = int(input('Effect: '))
    if(choice == 0 ):
        flag = 0
    else: flag=1
    while(choice<8):
        if(choice == 1):    #change brighten
            s = time.time()
            resImg, effectName = brightenImage(img, 30)
            e = time.time()
        
        elif (choice ==2):  #change contrast  
            s = time.time()
            resImg, effectName = constrastImage(img, 1.5)
            e = time.time()
        
        elif (choice ==3):  # flip image
            type = input("Horizontal or Vertical: ")
            s = time.time()
            resImg, effectName = flipImage(img, type)
            e = time.time()
        
        elif (choice ==4):  
            s = time.time() 
            resImg, effectName = grayImg(img)   # change image to gray
            e = time.time()
            print(effectName, ":",e-s)
            fileName = f"{imgName.split('.')[0]}_{effectName }.png"
            Image.fromarray(resImg).save(fileName)
                        
            s = time.time()
            resImg, effectName = sepiaImage(img)    # change image to sepia
            e = time.time()
        
        elif (choice ==5):
            s = time.time()
            resImg, effectName = blurImage(img)     #blur image
            e = time.time()
            print(effectName, ":",e-s)
            fileName = f"{imgName.split('.')[0]}_{effectName }.png"
            Image.fromarray(resImg).save(fileName)
            
            s = time.time()
            resImg, effectName = sharpenImage(img)     # sharpen image
            e = time.time()

        elif (choice == 6):
            s = time.time()
            resImg, effectName = cropImage(img, 200, 200)   # crop image (at center)
            e = time.time()

        elif (choice == 7):
            s = time.time()
            resImg, effectName = circleCropImage(img)   # crop image by circle frame
            e = time.time()
        
        if(flag == 0):      # execute all effects
            if(choice == 0):
                choice +=1
                continue
            else:
                choice +=1
                fileName = f"{imgName.split('.')[0]}_{effectName }.png"
                print(effectName, ":",e-s)
                Image.fromarray(resImg).save(fileName)  
        elif(flag != 0):
            fileName = f"{imgName.split('.')[0]}_{effectName }.png"
            print(effectName, ":",e-s)
            Image.fromarray(resImg).save(fileName)        
            break

main()

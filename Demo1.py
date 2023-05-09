import os
import sys
import cv2
import numpy as np
import glob

def convolve2D(image, kernel, padding=0, strides=1):
    # Cross Correlation
    # kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break

    return output

def getImages(path):
    images = []
    for img in glob.glob(path):
        image = cv2.imread(img)
        images.append(image)

    return images


def processImage(image):
    kernel = np.array([ [-1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1],
                    	[-1, -1, 24, -1, -1],
                        [-1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1] ])
    
    blur = np.array(
        [[1/16, 1/8, 1/16],
        [1/8, 1/4, 1/8],
        [1/16, 1/8, 1/16]]
    )

    image = convolve2D(image, blur)
    image = convolve2D(image, kernel)

    return image

def matchShape(image, shape, title, originalImage): 
    for i in range(1, 10):
        _shape = cv2.resize(shape, None, fx = i * .1, fy = i * .1)

        OUTPUT = convolve2D(image, _shape)

        threshold_value = .99999 * np.max(OUTPUT)
        _, thresholded = cv2.threshold(OUTPUT, threshold_value, 255, cv2.THRESH_BINARY)
        nonzero = cv2.findNonZero(thresholded)

        for pt in nonzero:
            x, y = pt[0]
            h, w = _shape.shape
            top_left = (x - w // 2, y - h // 2)
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(originalImage, top_left, bottom_right, 255)
    
    cv2.imshow(title, originalImage)


if __name__ == '__main__':

    tankImage = cv2.imread('CroppedTank.jpg')
    tankShape = processImage(tankImage)
    cv2.imshow("TANK", tankShape)

    pwd = os.path.dirname(__file__)
    dir = pwd+"/Data/*.jpg"
    images = getImages(dir)

    imageShapes = []
    for image in images:
        img = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
        img = processImage(img)
        imageShapes.append(img)

    for i, img in enumerate(imageShapes):
        matchShape(img, tankShape, str(i + 1), images[i])

    cv2.waitKey(0)
    cv2.destroyAllWindows()

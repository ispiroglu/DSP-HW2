import sys
import cv2
import numpy as np


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
        print(imagePadded)
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

def processImage(image): 
  image = cv2.imread(image) 
  image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY) 
  return image

if __name__ == '__main__':
    # kernel = processImage('/home/xpirr/workspace/python/DSP/HW2/CroppedTank.jpg')
    # img = processImage('/home/xpirr/workspace/python/DSP/HW2/Resim6_8.jpg')

    # result = cv2.filter2D(img, -1, kernel)
    # # Find the location of the maximum value in the result


    # Grayscale Image
    

    # Edge Detection Kernel
    kernel = np.array([ [-1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1],
                    	[-1, -1, 24, -1, -1],
                        [-1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1] ])
    
    array = np.ones((3,3))
    blur = np.array(
        [[1/16, 1/8, 1/16],
        [1/8, 1/4, 1/8],
        [1/16, 1/8, 1/16]]
    )
    
    croppedTank = processImage('/home/xpirr/workspace/python/DSP/HW2/CroppedTank2.jpg')
    blurredTank = convolve2D(croppedTank, blur, padding=0)
    tankShape = convolve2D(blurredTank, kernel)

    # cv2.imshow("Cropped Tank", croppedTank)
    # cv2.imwrite("Blurred Tank.jpg", blurredTank)
    cv2.imwrite("HW2/Tank Shape.jpg", tankShape)


    img = processImage('/home/xpirr/workspace/python/DSP/HW2/Resim6_8.jpg')
    blurredImg = convolve2D(img, blur, padding=0)
    bwImg = convolve2D(blurredImg, kernel, padding=0)

    # cv2.imshow("Original Image", img)
    # cv2.imshow("Blurred Image", blurredImg)
    cv2.imwrite("HW2/Shapes of Image.jpg", bwImg)


    # OUTPUT = convolve2D(bwImg, tankShape, padding=0)

    for i in range(1, 10):
        print(i)
        _tankShape = cv2.resize(tankShape, None, fx = i * .1, fy = i * .1)

        OUTPUT = convolve2D(bwImg, _tankShape, padding=int(_tankShape.shape[0]/2))

        threshold_value = .99 * np.max(OUTPUT)
        _, thresholded = cv2.threshold(OUTPUT, threshold_value, 255, cv2.THRESH_BINARY)
        nonzero = cv2.findNonZero(thresholded)

        # Draw a rectangle around the matched area
        for pt in nonzero:
            x, y = pt[0]
            h, w = _tankShape.shape
            top_left = (x - w // 2, y - h // 2)
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(img, top_left, bottom_right, 255)

    print()
    # Show the result
    cv2.imshow('Matched Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

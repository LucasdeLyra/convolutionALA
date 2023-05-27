import numpy as np
from matplotlib import image as img
from math import exp
import cv2
import sys

np.set_printoptions(threshold=sys.maxsize)

def processImage(image): 
    image = img.imread(image)
    return image


def padding_black(image):
    image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
    image_padded[1:-1, 1:-1] = image
    return image_padded

def padding_double(image):
    image = np.insert(image, 0, image[:,0], axis=1)
    image = np.insert(image, -1, image[:,-1], axis=1)
    image = np.insert(image, 0, image[0], axis=0)
    image = np.insert(image, -1, image[-1], axis=0)
    return image

def gaussian(x, mu, sigma):
  return exp(-(((x-mu)/(sigma))**2)/2.0)/(sigma*np.sqrt(np.pi*2))


def gaussian_vector(kernel_radius, sigma):
    hkernel = []
    for x in range(2*kernel_radius+1):
        hkernel.append(gaussian(x,kernel_radius, sigma))
    return np.array(hkernel)


def normalized_gaussian_kernel(vector):
    output = np.zeros((vector.shape[0], vector.shape[0]))
    for i in range(len(vector)):
        for j in range(len(vector)):
            output[i][j] = vector[i]*vector[j]

    normalize = np.sum(output)
    return output/normalize


def to_grayscale(image_rgb):
    image_grayscale = []
    for row in image_rgb:
        grayscale_line = []
        for r, g, b, a in row:
            grayscale = 0.299*r + 0.587*g + 0.114*b
            grayscale_line.append(grayscale)
        image_grayscale.append(grayscale_line)
    return np.array(image_grayscale)


def get_individual_color_channels(image):
    r = []
    g = []
    b = []
    a = []

    for row in image:

        red_line = []
        green_line = []
        blue_line = []
        alpha_line = []

        if row.shape[1] == 4:
            for red, green, blue, alpha in row:
                red_line.append(red)
                green_line.append(green)
                blue_line.append(blue)
                alpha_line.append(alpha)
            r.append(red_line)
            b.append(blue_line)
            g.append(green_line)
            a.append(alpha_line)

        else:
            for red, green, blue in row:
                red_line.append(red)
                green_line.append(green)
                blue_line.append(blue)
            r.append(red_line)
            b.append(blue_line)
            g.append(green_line)

    if row.shape[1] == 4:
        return [np.array(r), np.array(g), np.array(b), np.array(a)]
    else:
        return [np.array(r), np.array(g), np.array(b)]


def unite_color_channels(channels_list):
    final = []
    r = channels_list[0]
    g = channels_list[1]
    b = channels_list[2]
    if len(channels_list)==4:
        a = channels_list[3]
        for i in range(len(r)):
            linha = []
            for j in range(len(r[i])):
                aux = []
                aux.append(r[i][j])
                aux.append(g[i][j])
                aux.append(b[i][j])
                aux.append(a[i][j])
                linha.append(aux)
            final.append(linha)
    else:
        for i in range(len(r)):
            linha = []
            for j in range(len(r[i])):
                aux = []
                aux.append(r[i][j])
                aux.append(g[i][j])
                aux.append(b[i][j])
                linha.append(aux)
            final.append(linha)

    return np.array(final)


def convolution2D(image, kernel, padding=1):
    kernel = np.array(kernel)

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding)) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding)) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        image_padded = padding_double(image)
    else:
        image_padded = image

    for y in range(image.shape[1]):
        for x in range(image.shape[0]):
            try:
                output[x, y] = np.sum(kernel* image_padded[x: x + xKernShape, y: y + yKernShape])
            except:
                break

    return output


def gaussian_blur(image, iterations=1,output_file_path='./', output_file_name = 'gaussian'):
    vector = gaussian_vector(3,2)
    kernel = normalized_gaussian_kernel(vector)
    
    color_channels = get_individual_color_channels(image)
    for i in range(iterations):
        for color in range(len(color_channels)):
            color_channels[color] = convolution2D(color_channels[color], kernel)
        image = unite_color_channels([color_channels[2], color_channels[1], color_channels[0]])
        output_path = output_file_path + '\\' + output_file_name + str(i) + '.jpg'
        cv2.imwrite(output_path, (image*255).astype(np.uint8))
    return image


def medianblur(image, iterations=1, output_file_path='./', output_file_name = 'median'):
    MEDIAN_KERNEL = np.array([[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]])
    color_channels = get_individual_color_channels(image)
    for i in range(iterations):
        for color in range(len(color_channels)):
            color_channels[color] = convolution2D(color_channels[color], MEDIAN_KERNEL)
        image = unite_color_channels([color_channels[2], color_channels[1], color_channels[0]])
        output_path = output_file_path + '\\' + output_file_name + str(i) + '.jpg'
        cv2.imwrite(output_path, (image*255).astype(np.uint8))
    return image


def sobel_outline(image, output_file_path='./', output_file_name = 'sobel'):
    SOBEL_GY_KERNEL = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    SOBEL_GX_KERNEL = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

    image = to_grayscale(image)

    imageGx = convolution2D(image, SOBEL_GX_KERNEL)
    imageGy = convolution2D(image, SOBEL_GY_KERNEL)
    output_path = output_file_path + '\\' + 'gxsobel' + '.jpg'
    cv2.imwrite(output_path, (np.array(imageGx)*255).astype(np.uint8))
    output_path = output_file_path + '\\' + 'gysobel' + '.jpg'
    cv2.imwrite(output_path, (np.array(imageGy)*255).astype(np.uint8))

    final_image = []
    for i in range(len(imageGy)-1):
        final_image_line = []
        for j in range(len(imageGy[i])-1):
            final_image_line.append(np.sqrt(imageGx[i][j]**2+imageGy[i][j]**2))
        final_image.append(final_image_line)
    output_path = output_file_path + '\\' + output_file_name + '.jpg'
    cv2.imwrite(output_path, (np.array(final_image)*255).astype(np.uint8))
    return image


def prewitt_outline(image, output_file_path='./', output_file_name = 'PREWITT1'):
    PREWITT_GY_KERNEL = [[1, 1, 1], [0, 0, 0], [-1, -1, -1]]
    PREWITT_GX_KERNEL = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
    image = to_grayscale(image)

    imageGx = convolution2D(image, PREWITT_GX_KERNEL)
    imageGy = convolution2D(image, PREWITT_GY_KERNEL)
    output_path = output_file_path + '\\' + 'gx' + '.jpg'
    cv2.imwrite(output_path, (np.array(imageGx)*255).astype(np.uint8))
    output_path = output_file_path + '\\' + 'gy' + '.jpg'
    cv2.imwrite(output_path, (np.array(imageGy)*255).astype(np.uint8))
    
    final_image = []
    for i in range(len(imageGy)-1):
        final_image_line = []
        for j in range(len(imageGy[i])-1):
            final_image_line.append(np.sqrt(imageGx[i][j]**2+imageGy[i][j]**2))
        final_image.append(final_image_line)
    output_path = output_file_path + '\\' + output_file_name + '.jpg'
    cv2.imwrite(output_path, (np.array(final_image)*255).astype(np.uint8))
    return image
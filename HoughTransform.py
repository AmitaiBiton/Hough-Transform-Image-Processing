import random
import numpy as np
from cv2.cv2 import CV_8S
from matplotlib.pyplot import figure
from scipy import ndimage

figure(num=None, figsize=(16, 16), dpi=80, facecolor='w', edgecolor='k')
from matplotlib import pyplot as plt
import cv2
def createImage():
    img = np.zeros((400,400))
    i=100
    # create the picture we need
    for i in range(400):
        for j in range(400):
            if i>70 and j>100 and i<330 and j<300:
                img[i][j]=255
    return img
def create_gaussian():
    mean = 0
    gaussian = np.random.normal(mean,150,16000)
    return gaussian
def random_gaussian(img,gaussian):
    # Randomly selects pixels
    for i in range(16000):
        row = random.randint(0,399)
        column = random.randint(0,399)
        img[row][column] = img[row][column]+gaussian[i]
    return img


def filterGS(img , filter):
    temp = np.zeros((404, 404))
    for i in range(397):
        for j in range(397):
            for l in range(len(filter[0])):
                t = 0
                for k in range(len(filter[0])):
                    t += img[i + l][k + j] * filter[k][l]
                temp[i][j] += t
    return temp
def gaussians():
    image = createImage()
    imageNoise = random_gaussian(np.copy(image), create_gaussian())

    padding = np.zeros((404, 404))
    temp = np.copy(imageNoise)
    new_image = padding
    new_image[2:402, 2:402] = temp

    Gaussians = np.array(
        [[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]]) / 256
    """
    clean_image = filterGS(new_image, Gaussians)
    afterClean = clean_image[2:402, 2:402]
    result = afterClean[:402, :402]
    """
    result  = cv2.filter2D(imageNoise ,cv2.CV_64F, Gaussians)
    return image , imageNoise , result



def gradient_intensity(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.int32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.int32)
    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)
    G = np.hypot(Ix, Iy)
    D = np.arctan2(Iy, Ix)
    return (G, D)

def round_angle(angle):
    angle = np.rad2deg(angle)%180
    if (0<=angle<22.5) or (157.5<=angle<180):
        angle=0
    elif 22.5 <= angle < 67.5:
        angle=45
    elif 67.5 <=angle < 112.5:
        angle=90
    elif 112.5<=angle<157.5:
        angle=135
    return  angle

def suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    for i in range(M):
        for j in range(N):
            # find neighbour pixels to visit from the gradient directions
            where = round_angle(D[i, j])
            try:
                if where == 0:
                    if (img[i, j] >= img[i, j - 1]) and (img[i, j] >= img[i, j + 1]):
                        Z[i, j] = img[i, j]
                elif where == 90:
                    if (img[i, j] >= img[i - 1, j]) and (img[i, j] >= img[i + 1, j]):
                        Z[i, j] = img[i, j]

                elif where == 135:
                    if (img[i, j] >= img[i - 1, j - 1]) and (img[i, j] >= img[i + 1, j + 1]):
                        Z[i, j] = img[i, j]
                elif where == 45:
                    if (img[i, j] >= img[i - 1, j + 1]) and (img[i, j] >= img[i + 1, j - 1]):
                        Z[i, j] = img[i, j]

            except IndexError as e:
                pass

    return Z


def threshold(img, t, T):
    # define gray value of a WEAK and a STRONG pixel
    cf = {
        'WEAK': np.int32(170),
        'STRONG': np.int32(255),
    }
    # get strong pixel indices
    strong_i, strong_j = np.where(img > T)
    # get weak pixel indices
    weak_i, weak_j = np.where((img >= t) & (img <= T))
    # get pixel indices set to be zero
    zero_i, zero_j = np.where(img < t)
    # set values
    img[strong_i, strong_j] = cf.get('STRONG')
    img[weak_i, weak_j] = cf.get('WEAK')
    img[zero_i, zero_j] = np.int32(0)
    return (img, cf.get('WEAK'))

def tracking(img, weak, strong=255):
    M, N = img.shape
    for i in range(M-1):
        for j in range(N-1):
            if img[i, j] == weak:
                # check if one of the neighbours is strong (=255 by default)
                try:
                    if ((img[i + 1, j] == strong) or (img[i - 1, j] == strong)
                    or (img[i, j + 1] == strong) or (img[i, j - 1] == strong)
                    or (img[i + 1, j + 1] == strong) or (img[i - 1, j - 1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img

def hough_line(img):
    # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(-90.0, 90.0))
    width, height = img.shape
    diag_len = int( np.ceil(np.sqrt(width * width + height * height)) ) # max_dist
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)
    # Cache some resuable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)
    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, num_thetas))
    y_idxs, x_idxs = np.nonzero(img) # (row, col) indexes to edges
    # Vote in the hough accumulator
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for t_idx in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            rho = int( round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len )
            accumulator[rho, t_idx] += 1
    return accumulator, thetas, rhos

# get the for index of max
def getMax(H1):
    max = []
    for i in range(2):
        maxValue = np.argmax(H1, axis=0)[90]
        max.append( (maxValue, 90))
        H1[maxValue,90] =0
        H1[maxValue-1 ,90] =0
        H1[maxValue + 1, 90] = 0
        maxValue = np.argmax(H1, axis=0)[0]
        max.append( (maxValue, 0))
        H1[maxValue ,0] = 0
        H1[maxValue - 1, 0] = 0
        H1[maxValue + 1, 0] = 0
    return max

def hough_lines_draw(img, indicies, rhos, thetas):
    list =[]
    for i in range(len(indicies)):
        # reverse engineer lines from rhos and thetas
        rho = rhos[indicies[i][0]]
        theta = thetas[indicies[i][1]]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        # these are then scaled so that the lines go off the edges of the image
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        list.append((x1,y1))
    # sort the list to know evry time is the same "no change"
    list.sort()
    cv2.line(img, (list[0][0],list[2][1]),(list[0][0],list[3][1]), (255, 255, 255), 2)
    cv2.line(img, (list[1][0], list[2][1]), (list[1][0], list[3][1]), (255, 255, 255), 2)
    cv2.line(img, (list[0][0], list[2][1]), (list[1][0], list[2][1]), (255, 255, 255), 2)
    cv2.line(img, (list[0][0], list[3][1]), (list[1][0], list[3][1]), (255, 255, 255), 2)
    return  img
if  __name__ == "__main__":

    image, imageNoise , img1 = gaussians()
    img2, D = gradient_intensity(img1)
    img3 = suppression(np.copy(img2), D)
    img4, weak = threshold(np.copy(img3),170,200)
    img5 = tracking(np.copy(img4), weak)
    accumulator, thetas, rhos = hough_line(img5)
    listOfMax = getMax(accumulator)
    img = np.zeros((400,400))
    img  =hough_lines_draw(img, listOfMax , rhos ,thetas)


    plt.subplot(2, 2, 1), plt.imshow(image, cmap="gray"), plt.title('Orginal Image')
    plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(imageNoise, cmap="gray"), plt.title('Gaussian Noise ')
    plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3), plt.imshow(img5, cmap="gray"), plt.title('Canny')
    plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 4), plt.imshow(img, cmap="gray"), plt.title('Final Image')
    plt.xticks([]), plt.yticks([])



    plt.show()
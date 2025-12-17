# -*- coding: utf-8 -*-
"""

TP2 traitement Image CUPGE 2 ESIR

"""

#Bibliothèques nécessaires
import numpy as np  
import matplotlib.pyplot as plt 
from skimage import io  
from skimage.exposure import cumulative_distribution
from scipy.ndimage import convolve

#######################################
### 1-Ajout de bruit dans une image ###
#######################################


"""L'ajout de bruit se réalise par la fonction add_noise. Le bruit est 
choisi aléatoirement via la fonction np.random"""
def add_noise(image, ecarttype=15, intervalle_pixels=(0, 255)):
    noise = np.random.normal(size=image.shape, scale=ecarttype)
    img_noise = image + noise
    img_noise = np.clip(img_noise, a_min=intervalle_pixels[0], a_max=intervalle_pixels[1])
    return img_noise

plt.figure(1)
img = io.imread("boat.512.tiff")
plt.title("Boat")
plt.imshow(img, cmap="gray")
img = np.float32(img)

image_bruité = add_noise(img)
plt.figure(2)
plt.title("boat bruité")
plt.imshow(image_bruité, cmap="gray")


########################
### 2-Calcul du PSNR ###
########################

#im1 : img originale
#im2 : img bruitée

"""Cette fonction calcul l'erreur quadratique moyenne entre les pixels des
deux images."""


def mse(im1, im2):
    m,n = im1.shape
    mse = 0
    for i in range(0, m-1) : 
        for j in range(0, n-1) :
            mse += np.square(im1[i][j] - im2[i][j])
    return mse/(m*n)

#On devrait trouver qqch autour de 24-25
def psnr(im1, im2):
    PSNR = 10*np.log10(255**2/mse(im1, im2))
    return PSNR

#On trouve 24,7 ce qui est cohérent avec ce que l'on attendait
print("psnr image et img bruité: ", psnr(img, image_bruité))


############################
### 3-Réduction du bruit ###
############################

def mean_filter(N):
    return np.ones((N,N))/(N*N)


filtre = mean_filter(3)
convolution = convolve(image_bruité, filtre)
plt.figure(3)
plt.title("Boat filtré")
plt.imshow(convolution, cmap="gray")
plt.show()

print("psnr img et conv : ", psnr(img, convolution))
print("psnr img bruitée et conv : ", psnr(image_bruité, convolution))

#traiter le cas impair
def gaussian_filter(N, sigma):
    x0 = N//2
    y0 = N//2
    k = np.zeros((N,N))
    for i in range(0,N) : 
        for j in range(0, N) : 
            k[i,j] = np.exp(-((i-x0)**2+(j-y0)**2)/2*sigma**2)
    
    k=k/np.sum(k)
    return k
    
print(gaussian_filter(5, 1))
filtre_gaussian = gaussian_filter(5, 1)
convolution2 = convolve(img, filtre_gaussian)
plt.figure(4)
plt.title("Boat filtré gauss")
plt.imshow(convolution2, cmap="gray")
plt.show()

print("psnr gaussian : ", psnr(img, convolution2))
###################################
### 4-Rehaussement des contours ###
###################################

laplacien = [[0,1,0], [1,-4,1], [0,1,0]]

image_moyenne = convolve(image_bruité, filtre)
image_laplacienne1 = convolve(image_moyenne, laplacien)
image_finale1 = image_moyenne-image_laplacienne1
plt.figure(5)
plt.title("Filtre moyen puis Laplacien")
plt.imshow(image_finale1, cmap = "gray")
plt.show()

image_gaussienne = convolve(image_bruité, filtre_gaussian)
image_laplacienne2 = convolve(image_gaussienne, laplacien)
image_finale2 = image_gaussienne - image_laplacienne2
plt.figure(6)
plt.title("Filtre Gaussien puis Laplacien")
plt.imshow(image_finale2, cmap = "gray")
plt.show()

print("psnr image 1, moyen puis laplacien : ", psnr(image_bruité, image_finale1))
print("psnr image 2, gaussien puis laplacien : ", psnr(image_bruité, image_finale2))
"""Le psnr est moins bon pcq on a augmenté le contraste cependant les img sont nettement plus lisibles """
###################################
### 5-Égalisation d'histogramme ###
###################################


#########################################################
### 6-Réduction du bruit et rehaussement des contours ###
#########################################################


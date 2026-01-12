## Chargement du fichier

from scipy.io import wavfile
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from segmentation_fft import segmentation_fft
from supprInter import suppr_inter

plt.close("all")


def load_audio(file_path):
 """ Load audio file and return data and sampling frequency. """
 FS, data = wavfile.read(file_path)
 return data.astype(np.float32) / np.max(np.abs(data)), FS  # Normalize audio data


#Chargement du fichier
signal, Fs = load_audio("crying_beeps.wav")
t = np.linspace(0, len(signal)/Fs, len(signal))  
plt.figure(1)
plt.plot(t, signal)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('pleurs de bébé')

def save_audio(file_path, data, FS):
 """ Save the audio data to a file. """
 wavfile.write(file_path, FS, (data * 32767).astype(np.int16))  # Convert back to int16 for wav format


## 1. Segmentation   partir de l'energie du signal

def suppr_inter(segmentation_vector, size_s, framerate):
    segmentation_vector = np.array(segmentation_vector, dtype=int)

    der_reff = np.diff(segmentation_vector)

    x1 = np.where(der_reff == 1)[0] + 1
    y1 = np.where(der_reff == -1)[0] + 1

    if len(x1) > 0 and len(y1) > 0:
        if y1[0] < x1[0]:
            x1 = np.insert(x1, 0, 0)
        if y1[-1] < x1[-1]:
            y1 = np.append(y1, len(segmentation_vector) - 1)

    tr = y1 - x1

    petit_inter = np.where(tr < size_s * framerate)[0]

    for y in petit_inter:
        segmentation_vector[x1[y]:y1[y] + 1] = 0

    return segmentation_vector.tolist()


def segmentation_energie_glissante(signal_entree, FS, N, seuil):

    signal_energie = np.zeros(len(signal_entree))

    # calcul de l'energie
    for i in np.arange(N//2, len(signal_entree)-N//2):
        signal_energie[i] = np.mean(signal_entree[i-N//2:i+N//2]**2)

    # lissage
    win = 100
    h = np.ones(win) / win
    signal_energie_lisse = np.convolve(signal_energie, h, mode='same')

    # Seuiller
    vecteur_logique = signal_energie_lisse > seuil
   
    # utiliser supprInter
    
    vecteur_logique_corrige = suppr_inter(vecteur_logique, 0.1, Fs)
    signal_sortie = vecteur_logique_corrige*signal_entree
    
    # Creer le signal logique -> "vecteur_logique"

    seuil_signal = np.ones(len(signal_sortie)) * seuil
    plt.figure(figsize=(12, 8))

    plt.subplot(311)
    plt.plot(signal_entree)
    plt.title('Signal brut')

    plt.subplot(312)
    plt.plot(signal_energie_lisse)
    plt.title('Energie glissante lissée et seuil')
    plt.plot(seuil_signal, 'r')

    plt.subplot(313)
    plt.plot(signal_sortie)
    plt.title('Signal segmenté')

    plt.tight_layout()
    plt.show()

    return signal_sortie, vecteur_logique_corrige

signal_sortie, vecteur_logique_corrige = segmentation_energie_glissante(signal,Fs, 30, 0.0021 );
# enregistrement de l'extrait
save_audio("pleurs_bébé.wav", signal_sortie, Fs)


# 2. Segmentation   partir de la FFT

def segmentation_fft(signal_entree, vecteur_logique, FS):
    seuil = 145
    N = 4096
    f = np.arange(N) * FS / N
    signal_sortie = np.copy(signal_entree)

    # Positions des segments
    diff_logical_vector = np.diff(vecteur_logique)
    fronts_montants = np.where(diff_logical_vector == 1)[0] + 1
    fronts_descendants = np.where(diff_logical_vector == -1)[0] + 1
    positions = np.column_stack((fronts_montants, fronts_descendants))

    for i in range(len(positions)):
        print(len(positions))
        segment = signal_entree[positions[i, 0]:positions[i, 1] + 1]
        L = len(segment)

        plt.figure(figsize=(10, 6))

        plt.subplot(211)
        plt.plot(segment)
        plt.title(f'Segment {i + 1}')

        segment_fft = np.fft.fftshift(np.fft.fft(segment, N))
        plt.subplot(212)
        plt.plot(np.abs(segment_fft))
        plt.title('FFT Magnitude')

        moyenne = np.mean(np.abs(segment_fft))

        passage = 0

        #compter le nombre de fois ou le signal segment_fft passe par la moyenne
        for k in np.arange(0,len(segment_fft)):
            if segment_fft[k] >= moyenne:
                passage+=1 

        plt.suptitle(f'Segment {i + 1}, Value {passage}')
        plt.tight_layout()
        plt.show()

        # Definir la valeur qui permet de differencier les deux classes
        if passage < 250 :
            signal_sortie[positions[i,0]:positions[i,1] + 1] = 0
        
    return signal_sortie

signal_final = segmentation_fft(signal_sortie, vecteur_logique_corrige, Fs)

plt.figure(23)
plt.plot(t,signal_final)
plt.title("Signal final (avec seulement les pleurs de bébé")

# enregistrement de l'extrait
save_audio("pleurs_bébé_filtré.wav", signal_final, Fs)

# Calcul de la duree moyenne des pleurs"""
donnees = scipy.io.loadmat("donnees_bebe.mat")

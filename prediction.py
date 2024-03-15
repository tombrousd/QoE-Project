from pyOpenBCI import OpenBCICyton
import numpy as np
from keras.models import load_model
from scipy import signal
from scipy import stats

SCALE_FACTOR = (4500000)/24/(2**23-1) #Multiply uVolts_per_count to convert the channels_data to uVolts.


model = load_model('model_Transformer.h5')

def filtrageNotch(sig, fRemove, sampleFreq, Quality=30):
    # Design
    b, a = signal.iirnotch(fRemove, Quality, sampleFreq)
    # Apply
    signalFiltered = signal.filtfilt(b, a, sig)
    return signalFiltered

def filtragePasseBande(sig, fcHaut, fcBas, sampleFreq):
    # Fréquence d'échantillonnage
    fe = sampleFreq  # Hz
    
    # Fréquence de nyquist
    f_nyq = fe / 2.  # Hz
    
    # Fréquence de coupure
    fc1 = fcBas  # Hz
    fc2 = fcHaut  # Hz
    
    # Préparation du filtre de Butterworth en passe bande
    sos = signal.butter(4, [fc1, fc2], fs=sampleFreq, btype='band',output='sos')
    signal_bp = signal.sosfilt(sos, sig)
    
    return signal_bp

def Filtering_EEG(EEG):
    final_EEG = []
    for x in range(len(EEG[0])): # par electrode
        electrode = []
        for l in range(len(EEG)): # Toute la session
            electrode += [EEG[l][x]]

        # Filtering 
        notched = filtrageNotch(electrode,50,250)
        electrode_bp = filtragePasseBande(notched,47,1,250)

        final_EEG.append(electrode_bp)

    return np.transpose(final_EEG) #shape(15000,8)

def GetWavebands(sig):
    compt = 0
#     max_len = len(sig)
    for k in range(len(sig)):
        if (sig[k]==0):
            compt += 1
        
    if (compt >= 1):
        return [0,0,0,0,0,0,0,0,0,0]
    
    # FFT. Transformée de Fourier discrète. Elle permet de passer du domaine temporel au domaine fréquentiel.
    y = sig

    # yf = scipy.fftpack.fft(y). 
    xf, yf = signal.welch(y,fs=250,nperseg=(250))  #Calcul densité spectrale de puissance (PSD) avec méthode de Welch. fréquence d'échantillogage = 250 Hz. nperseg = 250 points par fenêtre.
    # xf,yf = plt.psd(y,Fs=250). 
    
    # Define delta lower and upper limits
    Dlow, Dhigh = 1, 4
    # Find intersecting values in frequency vector
    idx_delta = np.logical_and(xf >= Dlow, xf <= Dhigh)
    
    # Define theta lower and upper limits
    Tlow, Thigh = 4, 8
    # Find intersecting values in frequency vector
    idx_theta = np.logical_and(xf >= Tlow, xf <= Thigh)
    
    # Define Alpha lower and upper limits
    low, high = 8, 12
    # Find intersecting values in frequency vector
    idx_alpha = np.logical_and(xf >= low, xf <= high)
    
    # Define Beta lower and upper limits
    Blow, Bhigh = 12, 30
    # Find intersecting values in frequency vector
    idx_beta = np.logical_and(xf >= Blow, xf <= Bhigh)
    
    # Define gamma lower and upper limits
    Glow, Ghigh = 30, 60
    # Find intersecting values in frequency vector
    idx_gamma = np.logical_and(xf >= Glow, xf <= Ghigh)
    
    
    # Calcul PSD
    delta = []
    theta = []
    alpha = []
    beta = []
    gamma = []
    
    for k in range(len(xf)):
#         if (len(idx_alpha) == 0) or (len(idx_beta) == 0) or (len(idx_theta) == 0):
#            break
        if (idx_delta[k]==True):
            delta += [float(yf[k])]
        elif (idx_theta[k]==True):
            theta += [float(yf[k])]
        elif (idx_alpha[k]==True):
            alpha += [float(yf[k])]
        elif (idx_beta[k]==True):
            beta += [float(yf[k])]
        elif (idx_gamma[k]==True):
            gamma += [float(yf[k])]
    
    # Calcul DE
    sig_delta = filtragePasseBande(sig,4,1,250)
    DE_Delta = stats.differential_entropy(sig_delta)
    sig_theta = filtragePasseBande(sig,8,4,250)
    DE_Theta = stats.differential_entropy(sig_theta)
    sig_alpha = filtragePasseBande(sig,12,4,250)
    DE_Alpha = stats.differential_entropy(sig_alpha)
    sig_beta = filtragePasseBande(sig,30,12,250)
    DE_Beta = stats.differential_entropy(sig_beta)
    sig_gamma = filtragePasseBande(sig,60,30,250)
    DE_Gamma = stats.differential_entropy(sig_gamma)
    
    return [float(np.mean(delta)),float(np.mean(theta)),float(np.mean(alpha)),float(np.mean(beta)),float(np.mean(gamma)), 
            DE_Delta, DE_Theta, DE_Alpha, DE_Beta, DE_Gamma]

def EEG_feat(EEG, window):
    video_EEG = []
    win = int((window * 250) / 2)

    for l in range(int(len(EEG) / win)):
        features = []
        # Calculate ESD per window
        for x in range(len(EEG[0])):  # Par électrode
            electrode = []
            for x2 in range(len(EEG)):  # Toute la session (à partir de 2(window*250) points)
                electrode += [EEG[x2][x]]

            M_delta, M_theta, M_alpha, M_beta, M_gamma, DE_Delta, DE_Theta, DE_Alpha, DE_Beta, DE_Gamma = GetWavebands(
                electrode[l * win:(l * win) + (window * 250)])
            features += [M_delta, M_theta, M_alpha, M_beta, M_gamma, DE_Delta, DE_Theta, DE_Alpha, DE_Beta, DE_Gamma]

        video_EEG.append(features)

    return np.array([video_EEG])  # Ajout d'une dimension supplémentaire pour créer une sortie de forme (1 ou None, 40, 80)


#def save_data(sample):
    global data
    data.append([i*SCALE_FACTOR for i in sample.channels_data])

    if len(data) == 15000:
        data_brut = np.array(data)
        
        # Filtrage
        data_filtre = Filtering_EEG(data_brut)

        #Exctraction des features
        features = EEG_feat(data_filtre,3)
        
        #Prédiction
        prediction = model.predict(features)
        
        #Obtention QoE
        quality_classes = np.argmax(prediction, axis=1)
        if quality_classes == 2 : 
            print("QoE : Good Experience")
            print("------------------------")
        elif quality_classes == 1 :
            print("QoE : Neutral Experience")
            print("------------------------")
        elif quality_classes == 0 : 
            print('Qoe : Bad Experience')
            print("------------------------")
    
    elif len(data) > 15000:

        data = []
        print("***********")
        print("Nouvelle QoE en cours de calcul...")
        
def predict(data) :
    data_brut = np.array(data)
    data_filtre = Filtering_EEG(data_brut)
    features = EEG_feat(data_filtre,3)

    prediction = model.predict(features)
    quality_classes = np.argmax(prediction, axis=1)
    
    return quality_classes
    





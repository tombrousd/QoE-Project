import sys
import pyqtgraph as pg
import threading
import numpy as np

from PyQt5.QtGui import QPixmap

from pyqtgraph.Qt import QtGui, QtCore
from scipy import signal
from pyOpenBCI import OpenBCICyton
from prediction import predict, SCALE_FACTOR



colors = 'rgbycmwr'

# Set up GUI Layout
app = QtGui.QApplication([])
win = pg.GraphicsWindow(title='Python OpenBCI GUI')

ts_plots = [win.addPlot(row=i, col=0, colspan=2, title='Channel %d' % i, labels={'left': 'uV'}) for i in range(1,9)]
fft_plot = win.addPlot(row=1, col=2, rowspan=4, title='FFT Plot', labels={'left': 'uV', 'bottom': 'Hz'})
fft_plot.setLimits(xMin=1,xMax=125, yMin=0, yMax=1e7)
waves_plot = win.addPlot(row=5, col=2, rowspan=4, title='EEG Bands', labels={'left': 'uV', 'bottom': 'EEG Band'})
waves_plot.setLimits(xMin=0.5, xMax=5.5, yMin=0)
waves_xax = waves_plot.getAxis('bottom')
waves_xax.setTicks([list(zip(range(6), ('', 'Sleep', 'Drowsy', 'Relax', 'Active', 'Focus')))])
QoE = win.addLabel(row=9, col=0, colspan=3, title='QoE')
text_prediction = 'En cours de calcul...'
QoE.setText('Prédiction de la QoE : {}'.format(text_prediction))

data = [[0,0,0,0,0,0,0,0]]
data_pred = [[0,0,0,0,0,0,0,0]]

class UpdateQoESignal(QtCore.QObject):
    updateQoE = QtCore.pyqtSignal(str)

updateQoESignal = UpdateQoESignal()

# Define OpenBCI callback function
def save_data(sample):
    global data
    data.append([i*SCALE_FACTOR for i in sample.channels_data])

    if len(data) == 15000 :
        prediction = predict(data)
        if prediction == 1 :
            text_prediction = 'Neutral Experience'
        elif prediction == 2 :
            text_prediction = 'Good Experience'
        elif prediction == 0 :
            text_prediction = 'Bad Experience'

         # Mettre à jour l'étiquette QoE avec la nouvelle prédiction
        updateQoESignal.updateQoE.emit('Prédiction de la Qoe :  {}'.format(text_prediction))

        data = [[0,0,0,0,0,0,0,0]]

updateQoESignal.updateQoE.connect(QoE.setText)

# Define function to update the graphs
def updater():
    global data, plots, colors
    t_data = np.array(data[-1250:]).T #transpose data
    fs = 250 #Hz

    # Notch Filter
    def notch_filter(val, data, fs=250):
        notch_freq_Hz = np.array([float(val)])
        for freq_Hz in np.nditer(notch_freq_Hz):
            bp_stop_Hz = freq_Hz + 3.0 * np.array([-1, 1])
            b, a = signal.butter(3, bp_stop_Hz / (fs / 2.0), 'bandstop')
            fin = data = signal.lfilter(b, a, data)
        return fin

    # Bandpass filter
    def bandpass(start, stop, data, fs = 250):
        bp_Hz = np.array([start, stop])
        b, a = signal.butter(5, bp_Hz / (fs / 2.0), btype='bandpass')
        return signal.lfilter(b, a, data, axis=0)

    # Applying the filters
    nf_data = [[],[],[],[],[],[],[],[]]
    bp_nf_data = [[],[],[],[],[],[],[],[]]

    for i in range(8):
        nf_data[i] = notch_filter(60, t_data[i])
        bp_nf_data[i] = bandpass(15, 80, nf_data[i])

    # Plot a time series of the raw data
    for j in range(8):
        ts_plots[j].clear()
        ts_plots[j].plot(pen=colors[j]).setData(t_data[j])

    # Get an FFT of the data and plot it
    sp = [[],[],[],[],[],[],[],[]]
    freq = [[],[],[],[],[],[],[],[]]
    
    fft_plot.clear()
    for k in range(8):
        sp[k] = np.absolute(np.fft.fft(bp_nf_data[k]))
        freq[k] = np.fft.fftfreq(bp_nf_data[k].shape[-1], 1.0/fs)
        fft_plot.plot(pen=colors[k]).setData(freq[k], sp[k])


    # Define EEG bands
    eeg_bands = {'Delta': (1, 4),
                 'Theta': (4, 8),
                 'Alpha': (8, 12),
                 'Beta': (12, 30),
                 'Gamma': (30, 45)}

    # Take the mean of the fft amplitude for each EEG band (Only consider first channel)
    eeg_band_fft = dict()
    sp_bands = np.absolute(np.fft.fft(t_data[1]))
    freq_bands = np.fft.fftfreq(t_data[1].shape[-1], 1.0/fs)

    for band in eeg_bands:
        freq_ix = np.where((freq_bands >= eeg_bands[band][0]) &
                           (freq_bands <= eeg_bands[band][1]))[0]
        eeg_band_fft[band] = np.mean(sp_bands[freq_ix])

    # Plot EEG Bands
    bg1 = pg.BarGraphItem(x=[1,2,3,4,5], height=[eeg_band_fft[band] for band in eeg_bands], width=0.6, brush='r')
    waves_plot.clear()
    waves_plot.addItem(bg1)

# Define thread function
def start_board():
    board = OpenBCICyton(daisy=False)
    board.start_stream(save_data)
    
# Initialize Board and graphing update
if __name__ == '__main__':
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        x = threading.Thread(target=start_board)
        x.daemon = True
        x.start()

        timer = QtCore.QTimer()
        timer.timeout.connect(updater)
        timer.start(0)


        QtGui.QApplication.instance().exec_()

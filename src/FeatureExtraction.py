import os
import pickle
import antropy
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sklearn
import numpy as np
from skimage.feature import hog
import scipy
from scipy.signal import find_peaks
from Code.mag_smoothing import mag_smoothing

def histogram_of_gradients(x):
    fd, hog_image = hog(x, orientations=8, pixels_per_cell=(1, 1),
                    cells_per_block=(1, 1), visualize=True, channel_axis=-1)
    return hog_image


def MFCC(x, fs=44100, n_mfcc=20, frame_size=1024, hop_length=512):
    """
    compute MFCCs
        :param x: input signal
        :param fs: sampling rate
        :param n_mfcc: number of mel coefficients
        :param frame_size: size of the frame
        :param hop_length: hop size
    """

    S = librosa.feature.melspectrogram(y=x, sr=fs, hop_length=hop_length, win_length=frame_size)
    log_mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=n_mfcc)
    log_mfcc = sklearn.preprocessing.scale(log_mfcc, axis=1)

    log_mfcc_mean = np.mean(log_mfcc, axis=1)
    log_mfcc_var = np.var(log_mfcc, axis=1)
    log_mfcc_skew = scipy.stats.skew(log_mfcc, axis=1)
    log_mfcc_kur = scipy.stats.kurtosis(log_mfcc, axis=1)

    return log_mfcc_mean, log_mfcc_var, log_mfcc_skew, log_mfcc_kur

def amplitude_envelope(signal, frame_size=1024, hop_length=512):
    """
    compute amplitude envelope
        :param signal: input signal
        :param frame_size: size of the frame
        :param hop_length: hop size
    """
    
    return np.array([max(signal[i:i+frame_size]) for i in range(0, signal.size, hop_length)])


def normalize(x, axis=0):
    """
        Normalising the input
    """
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

def TC(x):
    """
    compute temporal centroid
        :param x: input signal

    """
    rmse, _ = energy(x, 1, 1)
    tot_e = sum(abs(x ** 2))
    tc = sum(rmse*x) / tot_e
    return tc

def spectral_rolloff(x, hop_length=512, win_length=1024):
    """
    compute spectral rolloff
        :param x: input signal
        :param win_length: size of the frame
        :param hop_length: hop size
    """
    sr = librosa.feature.spectral_rolloff(x, sr=44100, hop_length=hop_length, n_fft=win_length)[0]
    sr_mean = np.mean(sr)
    sr_var = np.var(sr)
    sr_skew = scipy.stats.skew(sr)
    sr_kur = scipy.stats.kurtosis(sr)
    return sr_mean, sr_var, sr_skew, sr_kur

def LogAttTime(x, sr=44100):
    """
    compute log attack time
        :param x: input signal
        :param sr: sampling rate
    """
    _startThreshold = 0.2
    _stopThreshold = 0.9

    maxvalue = max(x)

    startAttack = 0.0
    cutoffStartAttack = maxvalue * _startThreshold
    stopAttack = 0.0
    cutoffStopAttack = maxvalue * _stopThreshold

    for i in range(len(x)):
        if (x[i] >= cutoffStartAttack):
            startAttack = i
            break


    for i in range(len(x)):
        if (x[i] >= cutoffStopAttack):
            stopAttack = i
            break

    attackStart = startAttack / sr
    attackStop = stopAttack / sr

    attackTime = attackStop - attackStart

    logAttackTime = np.log10(attackTime)
    return logAttackTime


def Spectral_Kurtosis(x):
    """
    compute Spectral Kurtosis
        :param x: input signal

    """
    return scipy.stats.kurtosis(x)

def spectral_contrast(x, hop_length=512, win_length=1024, n_bands=6):
    """
    compute spectral contrast
        :param x: input signal
        :param win_length: size of the frame
        :param hop_length: hop size
        :param n_bands: hop number of bands
    """
    sc = librosa.feature.spectral_contrast(x, hop_length=hop_length, n_fft=win_length, n_bands=n_bands)
    sc_mean = np.mean(sc, axis=1)
    sc_var = np.var(sc, axis=1)
    sc_skew = scipy.stats.skew(sc, axis=1)
    sc_kur = scipy.stats.kurtosis(sc, axis=1)
    return sc_mean, sc_var, sc_skew, sc_kur

def Spectral_Skewness(x):
    """
    compute Spectral Skewness
        :param x: input signal

    """
    return scipy.stats.skew(x)

def autocorrelation(x):
    """
    compute autocorrelation
        :param x: input signal

    """
    x = np.array(x)
    # Mean
    mean = np.mean(x)
    # Variance
    var = np.var(x)
    # Normalized data
    ndata = x - mean
    acorr = np.correlate(ndata, ndata, 'full')[len(ndata) - 1:]
    acorr = acorr / var / len(ndata)
    return acorr

#Zero-crossing-rate

def ZCR(x):
    """
    compute zero crossing rate
        :param x: input signal

    """
    zcrs = librosa.feature.zero_crossing_rate(x)
    return zcrs.reshape(-1)

#Energy
def energy(x, hop_length = 512, frame_length = 1024):
    """
    compute the energy of the signal
        :param x: input signal
        :param frame_length: size of the frame
        :param hop_length: hop size
    """

    rmse = librosa.feature.rms(x, frame_length=frame_length, hop_length=hop_length, center=True)

    rmse = rmse[0]
    energy = np.array([
        sum(abs(x[i:i+frame_length]**2))
        for i in range(0, len(x), hop_length)])
    return rmse, energy


def spectral_flux(wavedata, win_length=1024, hop_length=512):
    """
    compute Spectral flux
        :param wavedata: input signal
        :param win_length: size of the frame
        :param hop_length: hop size
    """
    # convert to frequency domain
    magnitude_spectrum = librosa.stft(wavedata, n_fft=win_length, hop_length=hop_length)
    timebins, freqbins = np.shape(magnitude_spectrum)
    # when do these blocks begin (time in seconds)?
    #timestamps = (np.arange(0, timebins - 1) * (timebins / float(sample_rate)))
    sf = np.sqrt(np.sum(np.diff(np.abs(magnitude_spectrum)) ** 2, axis=1)) / freqbins
    #return sf[1:]#, np.asarray(timestamps)
    sf_mean = np.mean(sf[1:])
    sf_var = np.var(sf[1:])
    sf_skew = scipy.stats.skew(sf[1:])
    sf_kur = scipy.stats.kurtosis(sf[1:])
    return sf_mean, sf_var, sf_skew, sf_kur

def SpectralCentroid(x, hop_length=512, win_length=1024, fs=44100):
    """
    compute Spectral Centroid
        :param x: input signal
        :param win_length: size of the frame
        :param hop_length: hop size
        :param fs: sampling rate
    """

    cent = librosa.feature.spectral_centroid(x, hop_length=hop_length, n_fft=win_length, sr=fs)
    sc_mean = np.mean(cent[0])
    sc_var = np.var(cent[0])
    sc_skew = scipy.stats.skew(cent[0])
    sc_kur = scipy.stats.kurtosis(cent[0])
    return sc_mean, sc_var, sc_skew, sc_kur


def CQT(x, hop_length=512, n_bins=72, fs=44100):
    """
    compute Constant Q-transform (CQT)
        :param x: input signal
        :param hop_length: hop size
        :param n_bins: number of bins
        :param fs: sampling rate
    """

    #%matplotlib inline
    fmin = librosa.midi_to_hz(36)

    C = np.abs(librosa.cqt(y=x, sr=fs, fmin=fmin, n_bins=n_bins, hop_length=hop_length))
    #display
    #logC = librosa.amplitude_to_db(C)
    #plt.figure(figsize=(15, 5))
    #librosa.display.specshow(logC, sr=fs, x_axis='time', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')
    cqt_mean = np.mean(C, axis=1)
    cqt_var = np.var(C, axis=1)
    cqt_skew = scipy.stats.skew(C, axis=1)
    cqt_kur = scipy.stats.kurtosis(C, axis=1)
    return np.abs(cqt_mean), np.abs(cqt_var), np.abs(cqt_skew), np.abs(cqt_kur)

def energy_bands(S, freq_bins):
    """
    compute energy in two different bands: [150, 750] and [750, -]
        :param S: input signal spectrum
        :param freq_bins: frequency bins of the spectrum

    """
    index1 = np.where(freq_bins > 150)[0][0]
    index2 = np.where(freq_bins < 750)[0][-1] + 1

    e1 = sum(abs(S[index1:index2]**2))#sum(np.power(S[index1:index2], 2))
    e2 = sum(abs(S[index2:]**2))

    return [e1, e2]

def collect_peaks(data_dir, spectra_db, index, height, distance=400):
    """
    compute the first five harmonics of the spectrum
        :param data_dir: path to folder where to save the plot
        :param spectra_db: input signal spectrum in decibel
        :param index: frequency bin from where the search starts
        :param height: minimum height for the peaks
        :param distance: minimum distance between the peaks
    """
    peaks, _ = find_peaks(spectra_db[index:], height=height, distance=distance)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.semilogx(spectra_db)
    ax.semilogx(peaks + index, spectra_db[peaks + index], "x")
    fig.savefig(data_dir)
    plt.close('all')

    peaks_db = spectra_db[peaks[:6] + index]

    return peaks_db

def difference_chord(spectrums):
    """
    compute the difference between the spectrum of the chord and the sum of the notes composing the chords
        :param spectrums: input signal spectrum

    """
    spectra_db0 = 10 * np.log10(np.abs(mag_smoothing(spectrums[0], 64)))
    spectra_db1 = 10 * np.log10(np.abs(mag_smoothing(spectrums[1], 64)))
    spectra_db2 = 10 * np.log10(np.abs(mag_smoothing(spectrums[2], 64)))
    spectra_db3 = 10 * np.log10(np.abs(mag_smoothing(spectrums[3], 64)))

    s = spectra_db0 + spectra_db1 + spectra_db2 - spectra_db3

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.semilogx(s)
    s = s[100:40100]
    return [s[:8000], s[8000:8000*2], s[8000*2:8000*3], s[8000*3:8000*4], s[8000*4:8000*5]]

def difference_rep(spectrums):
    """
    compute the difference between the spectrum of the repeated note and the same note played isolated
        :param spectrums: input signal spectrum

    """
    spectra_db0 = 10 * np.log10(np.abs(mag_smoothing(spectrums[0], 64)))
    spectra_db1 = 10 * np.log10(np.abs(mag_smoothing(spectrums[1], 64)))
    s = spectra_db1 - spectra_db0
    s = s[100:40100]
    return [s[:8000], s[8000:8000 * 2], s[8000 * 2:8000 * 3], s[8000 * 3:8000 * 4], s[8000 * 4:8000 * 5]]

def bandwidth(x, hop_length, win_length):
    """
    compute the bandwidth
        :param x: input signal
        :param win_length: size of the frame
        :param hop_length: hop size
    """
    return librosa.feature.spectral_bandwidth(x, sr=44100, n_fft=win_length, hop_length=hop_length)[0]

def entro(x, nperseg):
    """
    compute the entropy
        :param x: input signal
        :param nperseg: Length of each FFT segment

    """
    return antropy.spectral_entropy(x, sf=44100, method='fft', nperseg=nperseg, normalize=False, axis=-1)

def rms(x):
    """
    compute the root mean squared energy of the signal
        :param x: input signal

    """
    return np.sqrt(np.mean(np.power(x, 2)))

def findAttack(x, sr=44100):
    """
    retrieve the attack time in samples
        :param x: input signal
        :param sr: sampling rate
    """
    _startThreshold = 0.2
    maxvalue = max(x)
    startAttack = 0.0
    cutoffStartAttack = maxvalue * _startThreshold
    for i in range(len(x)):
        if (x[i] >= cutoffStartAttack):
            startAttack = i
            break
    return startAttack

def extract_all_features(Z, type, win_length=2048, hop_length=2048, n_bands=6, n_bins=20, n_mfcc=20):
    """
    extract all features and build the feature vectors
        :param Z: pickle data containing all the recordings
        :param type: if real, digital, of diskalvier
        :param win_length: frame size
        :param hop_length: hop size
        :param n_bands: number of bins to bands (for spectral contrast)
        :param n_bins: number of bins to compute (for CQT)
        :param n_mfcc: number of mel coefficients to compute
    """
    piano_info, features = [], []
    data_dir = '../../../Analysis/Note_collector'
    data = None
    lag = 0#5000
    sustain = 60000
    # retrive the precomputed harmonics
    if type == 'real':
        data = open(os.path.normpath('/'.join([data_dir, 'all_peaks_real_piano.pickle'])), 'rb')
    elif type == 'disk':
        data = open(os.path.normpath('/'.join([data_dir, 'all_peaks_disk_piano.pickle'])), 'rb')
    elif type == 'digital':
        data = open(os.path.normpath('/'.join([data_dir, 'all_peaks_digital_piano.pickle'])), 'rb')
    Z1 = pickle.load(data)
    ind = 0
    for i in range(len(Z)):
        for n in range(len(Z[i]['vel'])):

            name = Z[i]['piano']
            time = Z[i]['time'][n]
            s = findAttack(time)
            time = time[s-lag:s+sustain]
            tukey = scipy.signal.windows.tukey(len(time), alpha=0.1)
            time = time*tukey
            time = np.pad(time, (0, 88100-len(time)))
            plt.plot(time)
            note = Z[i]['note']
            vel = Z[i]['vel'][n]

            name_ = Z1['name'][ind]
            note_ = Z1['note'][ind]
            vel_ = Z1['vel'][ind]

            #check if there are misalignment between the recordings and related precomputed harmonics
            if name_ != name or note_ != note or vel_ != vel:
                print('wrong')

            spectrum = np.abs(np.divide(Z[i]['spectrum'][n], len(time)))
            mean_s = np.mean(spectrum) #4
            var_s = np.var(spectrum) #5
            s_skew = np.abs(Spectral_Skewness(spectrum))  # 6
            sk = np.abs(Spectral_Kurtosis(spectrum))  # 7
            s = [mean_s, var_s, s_skew, sk]

            s_roll_m, s_roll_v, s_roll_s, s_roll_k = spectral_rolloff(time, win_length=win_length, hop_length=hop_length)  # 9
            s_roll = [s_roll_m, s_roll_v, s_roll_s, s_roll_k]

            sc_m, sc_v, sc_s, sc_k = spectral_contrast(time, hop_length=hop_length, win_length=win_length, n_bands=n_bands) #8
            sc = [sc_m, sc_v, sc_s, sc_k]

            sf_m, sf_v, sf_s, sf_k = spectral_flux(time)#, hop_length=10, win_length=10)) #10
            sf = [sf_m, sf_v, sf_s, sf_k]

            s_centroid_m, s_centroid_v, s_centroid_s, s_centroid_k = SpectralCentroid(time, hop_length=hop_length, win_length=win_length) #11
            s_centroid = [s_centroid_m, s_centroid_v, s_centroid_s, s_centroid_k]

            rmse, _ = energy(time, hop_length=hop_length, frame_length=win_length) #12

            e1, e2 = energy_bands(spectrum, Z[i]['freq'][n]) #13,14
            peaks_0 = Z1['peaks'][ind][0] #15
            peaks_1 = Z1['peaks'][ind][1] #16
            peaks_2 = Z1['peaks'][ind][2] #17
            peaks_3 = Z1['peaks'][ind][3] #18
            peaks_4 = Z1['peaks'][ind][4] #19
            tc = TC(time)  #0
            log_attack = LogAttTime(time)  #1
            cqt_m, cqt_var, cqt_s, cqt_k = CQT(time, n_bins=n_bins)  #2
            cqt = [cqt_m, cqt_var, cqt_s, cqt_k]

            mfcc_m, mfcc_var, mfcc_s, mfcc_k = MFCC(time, n_mfcc=n_mfcc)  #3
            mfcc = [mfcc_m, mfcc_var, mfcc_s, mfcc_k]

            ind = ind + 1
            rms_s = rms(time)
            entropy = entro(time, win_length)
            #band = bandwidth(time, hop_length=hop_length, win_length=win_length)

            features.append([tc, log_attack, cqt, mfcc, s, sc, s_roll, sf, s_centroid,
                             e1, e2, peaks_0, peaks_1, peaks_2, peaks_3, peaks_4, entropy])#, rms_s])
            piano_info.append([name, note, vel])

    return piano_info, features

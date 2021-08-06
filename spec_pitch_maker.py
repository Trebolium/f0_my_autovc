# use Hyper_params consistently
# use pyin instead
import os, pdb, yaml
import pickle
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.signal import get_window
from librosa.filters import mel
from numpy.random import RandomState
from yin import pitch_calc

with open('Hyper_Parameter.yaml') as f:
    hp_Dict = yaml.load(f, Loader=yaml.Loader)

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
    
    
def pySTFT(x, fft_length=1024, hop_length=256):
    
    x = np.pad(x, int(fft_length//2), mode='reflect')
    
    noverlap = fft_length - hop_length
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//hop_length, fft_length)
    strides = x.strides[:-1]+(hop_length*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)
    
    fft_window = get_window('hann', fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T
    
    return np.abs(result)    


def Pitch_Generate(audio):
    pitch = pitch_calc(
        sig= audio,
        sr= hp_Dict['Sound']['Sample_Rate'],
        w_len= hp_Dict['Sound']['Frame_Length'],
        w_step= hp_Dict['Sound']['Frame_Shift'],
        confidence_threshold= hp_Dict['Sound']['Confidence_Threshold'],
        gaussian_smoothing_sigma = hp_Dict['Sound']['Gaussian_Smoothing_Sigma']
        )
    # pdb.set_trace()
    pitch_normed = (pitch - np.min(pitch)) / (np.max(pitch) - np.min(pitch) + 1e-7)
    
    # normalize F0 between 0 and 1
    return pitch_normed  
    
mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
min_level = np.exp(-100 / 20 * np.log(10))
b, a = butter_highpass(30, 16000, order=5)


# audio file directory
rootDir = './wavs'
# spectrogram directory
targetDirSpec = './spmel'
targetDirPitch = './pitch'

if not os.path.exists(targetDirSpec):
    os.makedirs(targetDirSpec)
if not os.path.exists(targetDirPitch):
    os.makedirs(targetDirPitch)

dirName, subdirList, fileList = next(os.walk(rootDir))
print('Found directory: %s' % dirName)

prng = RandomState(225) 
for fileName in sorted(fileList):
    print(fileName)
    # Read audio file
    x, fs = sf.read(os.path.join(dirName,fileName))
    # Remove drifting noise
    y = signal.filtfilt(b, a, x)
    # Ddd a little random noise for model roubstness
    wav = y * 0.96 + (prng.rand(y.shape[0])-0.5)*1e-06
    # Compute pitch contours
    pitch_contour = Pitch_Generate(wav)
    # Compute spect
    D = pySTFT(wav).T
    # Convert to mel and normalize
    D_mel = np.dot(D, mel_basis)
    D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
    S = np.clip((D_db + 100) / 100, 0, 1)    
    # save pitch_contour
    np.save(os.path.join(targetDirPitch, fileName[:-4]),
            pitch_contour.astype(np.float32), allow_pickle=False) 
    # save spect    
    np.save(os.path.join(targetDirSpec, fileName[:-4]),
            S.astype(np.float32), allow_pickle=False)    

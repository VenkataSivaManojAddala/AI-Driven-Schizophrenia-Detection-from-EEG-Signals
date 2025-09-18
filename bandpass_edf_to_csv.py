import pyedflib
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import pandas as pd

def band_pass_filter(sig):
    freq = 128
    lowcut = 0.2
    highcut = 60
    nyquist = 0.5 * freq
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(4, [low, high], btype='band')
    return lfilter(b, a, sig, axis=0)


file_name = "s01.edf"
file_path="s_edf\\"+file_name
f = pyedflib.EdfReader(file_path)
n = f.signals_in_file
signal_labels = f.getSignalLabels()
sampling_rates = f.getSampleFrequencies()
sigbufs = np.zeros((n, f.getNSamples()[0]))
sig_bp = np.zeros((n, f.getNSamples()[0]))

nrows=f.getNSamples()[0]
ncol=list(map(lambda x:str(x),range(1,20)))
df=pd.DataFrame(columns=ncol)
for i in np.arange(n):
    sigbufs[i, :] = f.readSignal(i)
    sig_bp[i, :]=band_pass_filter(f.readSignal(i))
    df[str(i+1)]=sig_bp[i,:]

df.to_csv("C:\\Users\\vpoor\\Desktop\\H\\"+file_name[0:3]+".csv",index=False)
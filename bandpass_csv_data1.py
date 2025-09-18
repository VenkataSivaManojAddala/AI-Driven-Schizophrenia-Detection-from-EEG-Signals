import pyedflib
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import pandas as pd
import os

def band_pass_filter(sig):
    freq = 128
    lowcut = 0.2
    highcut = 60
    nyquist = 0.5 * freq
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(4, [low, high], btype='band')
    return lfilter(b, a, sig, axis=0)


csv_files = [f for f in os.listdir("sch\\") if f.endswith('.csv')]
for fi in csv_files:
    file_name = fi
    file_path = "sch\\" + file_name
    f = pd.read_csv(file_path)
    n = len(f.columns)
    nrows = f.shape[0]
    ncol = list(map(lambda x: str(x), range(1, 17)))
    df = pd.DataFrame(columns=ncol)
    for i in range(1,17):
        df[str(i)] = band_pass_filter(f[str(i)])
    df.to_csv("C:\\Users\\vpoor\\Desktop\\S2\\" + file_name[0:4] + ".csv", index=False)
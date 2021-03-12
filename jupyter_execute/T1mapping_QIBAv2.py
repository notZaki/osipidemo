# QIBA v2

import sys
sys.path.insert(0, '../src')

import original.T1_mapping.MJT_EdinburghUK.t1 as edinburgh
import original.T1_mapping.ST_SydneyAus.VFAT1mapping as sydney
import original.T1_mapping.McGill.vfa as mcgill
import matplotlib.pyplot as plt
import os
import urllib.request
import mat73
import numpy as np
from numpy.linalg import norm

from matplotlib import animation
from IPython.display import HTML

plt.style.use('https://gist.github.com/notZaki/8bfb049230e307f4432fd68d24c603a1/raw/c0baa2a1c55afdf1764b26ee2ebeb1cbf26d8d98/pltstyle')

# Get the file
datafolder = "./data"
if not os.path.isdir(datafolder): os.makedirs(datafolder)

t1file = os.path.join(datafolder, "t1data_v2.mat")
if not os.path.isfile(t1file):
    urllib.request.urlretrieve("https://osf.io/3hwtc/download", t1file)

# Open file

t1data = mat73.loadmat(t1file)
t1data.keys()

def show_error_maps(fittedmaps, sigmaidx = 0, title = "", showcbar = False, truthdata = t1data, clim = (-100, 100)):
    fittedM0, fittedT1 = fittedmaps
    fittedM0 = fittedM0[:,:,sigmaidx]
    fittedT1 = fittedT1[:,:,sigmaidx]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,4))
    ax1.imshow(percenterror(fittedM0, truthdata.M0), cmap = "PuOr_r", clim = clim)
    ax1.grid(False)
    plt.sca(ax1)
    plt.title("%error in M0")
    plt.xticks(np.arange(5, 150, 10), np.round((np.unique(truthdata.T1))).astype(int), rotation=45, ha="right")
    plt.yticks(np.arange(5, 70, 10), np.round((np.unique(truthdata.M0))).astype(int))
    
    im = ax2.imshow(percenterror(fittedT1, truthdata.T1), cmap = "PuOr_r", clim = clim)
    ax2.grid(False)
    plt.sca(ax2)
    plt.title("%error in T1")
    plt.xticks(np.arange(5, 150, 10), np.round((np.unique(truthdata.T1))).astype(int), rotation=45, ha="right")
    plt.yticks(np.arange(5, 70, 10), np.round((np.unique(truthdata.M0))).astype(int))
    
    fig.suptitle(title)
    if showcbar:
        fig.colorbar(im, ax=[ax1,ax2], orientation='horizontal', shrink = 0.5)
    else:
        fig.tight_layout()
    
    return fig


def percenterror(estimated, truth):
    return 100 * (estimated - truth) / truth

def vfa_fit(_signal, _fa, _tr, author, fittype = "linear", mask = None):
    # Signal: numpy array, last dimension must be flip angle
    # fa: flip angles, in radian
    # tr: repetition time, in ms
    
    spatialdims = _signal.shape[:-1]
    signal = _signal.reshape(-1, _signal.shape[-1])
    if mask != None:
        signal = signal[mask[:] > 0, :]
        
    if author == "edinburgh":
        fa = _fa
        tr = _tr
        if fittype == "nonlinear":
            _M0, _T1 = edinburgh.fit_vfa_nonlinear(signal, fa, tr)
        else: # linear
            _M0, _T1 = edinburgh.fit_vfa_linear(signal, fa, tr)
    elif author == "sydney":
        fa = np.rad2deg(_fa)
        tr = _tr
        numvox = np.prod(spatialdims)
        _M0 = np.zeros(numvox)
        _T1 = np.zeros(numvox)
        for idx in range(numvox):
            _M0[idx], _T1[idx] = sydney.VFAT1mapping(fa, signal[idx, :], tr, method = fittype)
    elif author == "mcgill":
        fa = _fa
        tr = _tr
        if fittype == "nonlinear":
            _M0, _T1 = mcgill.novifast(signal, fa, tr)
        elif fittype == "nonlinear_noniterative":
            _M0, _T1 = mcgill.novifast(signal, fa, tr, doiterative = False)
        else: # linear
            _M0, _T1 = mcgill.despot(signal, fa, tr)
    else:
        print("ERROR: Unexpected author")
        return
    
    if mask != None:
        M0 = np.zeros(spatialdims)
        T1 = np.zeros(spatialdims)
        M0[mask[:] > 0] = _M0
        T1[mask[:] > 0] = _T1
    else:
        M0 = _M0.reshape(spatialdims)
        T1 = _T1.reshape(spatialdims)
    return (M0, T1)

The following tests are on the [QIBA](https://sites.duke.edu/dblab/qibacontent/) v2 phantom[^1]. 
The phantom is identical to QIBA v1 and consists of a 2D grid containing 7 $M_0$ values, and 15 $T_1$ values.
However, unlike QIBA v1, the v2 phantom has noise with $\sigma = \{2, 5, 10, 20, 50, 100\}$

[^1]: This could've been QIBA v3, as v2 and v3 are identical

print("LLS - McGill")
%time mcgill_lls = vfa_fit(t1data.signal, t1data.fa, t1data.TR, author = "mcgill", fittype = "linear")
print(" ")

print("NLS - McGill")
%time mcgill_nls = vfa_fit(t1data.signal, t1data.fa, t1data.TR, author = "mcgill", fittype = "nonlinear")

show_error_maps(mcgill_lls, 0, title = "Linear least squares")
show_error_maps(mcgill_nls, 0, title = "Non-linear least squares", showcbar = True)

show_error_maps(mcgill_lls, 1, title = "Linear least squares")
show_error_maps(mcgill_nls, 1, title = "Non-linear least squares", showcbar = True)

show_error_maps(mcgill_lls, 2, title = "Linear least squares")
show_error_maps(mcgill_nls, 2, title = "Non-linear least squares", showcbar = True)

show_error_maps(mcgill_lls, 3, title = "Linear least squares")
show_error_maps(mcgill_nls, 3, title = "Non-linear least squares", showcbar = True)

show_error_maps(mcgill_lls, 4, title = "Linear least squares")
show_error_maps(mcgill_nls, 4, title = "Non-linear least squares", showcbar = True)

show_error_maps(mcgill_lls, 5, title = "Linear least squares")
show_error_maps(mcgill_nls, 5, title = "Non-linear least squares", showcbar = True)

# The above can be repeated with other implementations:

# print("LLS - Edinburgh")
# %time edinburgh_lls = vfa_fit(t1data.signal, t1data.fa, t1data.TR, author = "edinburgh", fittype = "linear")
# print(" ")

# print("LLS - Sydney")
# %time sydney_lls = vfa_fit(t1data.signal, t1data.fa, t1data.TR, author = "sydney", fittype = "linear")
# print(" ")

## The NLS implementation sometimes give an error

# print("NLS - Edinburgh")
# %time edinburgh_nls = vfa_fit(t1data.signal, t1data.fa, t1data.TR, author = "edinburgh", fittype = "nonlinear")
# print(" ")

# print("NLS - Sydney")
# %time sydney_nls = vfa_fit(t1data.signal, t1data.fa, t1data.TR, author = "sydney", fittype = "nonlinear")
# print(" ")


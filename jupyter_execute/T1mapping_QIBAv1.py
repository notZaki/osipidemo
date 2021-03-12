# QIBA v1

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

plt.style.use('https://gist.github.com/notZaki/8bfb049230e307f4432fd68d24c603a1/raw/c0baa2a1c55afdf1764b26ee2ebeb1cbf26d8d98/pltstyle')

# Get the file
datafolder = "./data"
if not os.path.isdir(datafolder): os.makedirs(datafolder)

t1file = os.path.join(datafolder, "t1data.mat")
if not os.path.isfile(t1file):
    urllib.request.urlretrieve("https://osf.io/jzu4k/download", t1file)

# Load file
t1data = mat73.loadmat(t1file)

def show_maps(fittedmaps = None, M0 = None, T1 = None, truthdata = t1data):
    if np.any(M0):
        fittedM0 = M0
        fittedT1 = T1
    else:
        fittedM0, fittedT1 = fittedmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,8))
    ax1.imshow(fittedM0)
    ax1.grid(False)
    plt.sca(ax1)
    plt.title("M0")
    plt.xticks(np.arange(5, 150, 10), np.round((np.unique(truthdata.T1))).astype(int), rotation=45, ha="right")
    plt.yticks(np.arange(5, 70, 10), np.round((np.unique(truthdata.M0))).astype(int))
    
    ax2.imshow(fittedT1)
    ax2.grid(False)
    plt.sca(ax2)
    plt.title("T1")
    plt.xticks(np.arange(5, 150, 10), np.round((np.unique(truthdata.T1))).astype(int), rotation=45, ha="right")
    plt.yticks(np.arange(5, 70, 10), np.round((np.unique(truthdata.M0))).astype(int))
    
    fig.tight_layout()
    return

def show_error_maps(fittedmaps, title = "", showcbar = False, truthdata = t1data, clim = (-100, 100)):
    fittedM0, fittedT1 = fittedmaps
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
    return

def percenterror(estimated, truth):
    return 100 * (estimated - truth) / truth

# Wrapper --- should move this to library
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

The following tests are on the [QIBA](https://sites.duke.edu/dblab/qibacontent/) v1 phantom. 
The phantom consists of a 2D grid containing 7 $M_0$ values, and 15 $T_1$ values.

The ground truth is show below:

show_maps(M0 = t1data.M0, T1 = t1data.T1)

## Linear least-squares

### Edinburgh

%%time 
edinburgh_lls = fittedM0, fittedT1 =  vfa_fit(t1data.signal, t1data.fa, t1data.TR, author = "edinburgh", fittype = "linear")
show_maps(edinburgh_lls); 

### Sydney

%%time
sydney_lls = vfa_fit(t1data.signal, t1data.fa, t1data.TR, author = "sydney", fittype = "linear")
show_maps(sydney_lls)

### McGill

%%time
mcgill_lls = vfa_fit(t1data.signal, t1data.fa, t1data.TR, author = "mcgill", fittype = "linear")
show_maps(mcgill_lls)

# Non-linear least squares

### Edinburgh

%%time
edinburgh_nls =  vfa_fit(t1data.signal, t1data.fa, t1data.TR, author = "edinburgh", fittype = "nonlinear")
show_maps(edinburgh_nls)

### Sydney

%%time 
sydney_nls = vfa_fit(t1data.signal, t1data.fa, t1data.TR, author = "sydney", fittype = "nonlinear")
show_maps(sydney_nls)

### McGill

%%time
mcgill_nls = vfa_fit(t1data.signal, t1data.fa, t1data.TR, author = "mcgill", fittype = "nonlinear")
show_maps(mcgill_nls)

## Percent error maps

### Linear least-squares

clim = (-20, 20)
show_error_maps(edinburgh_lls, title = "Edinburgh LLS", clim = clim);
show_error_maps(sydney_lls, title = "Sydney LLS", clim = clim);
show_error_maps(mcgill_lls, title = "McGill LLS", clim = clim, showcbar = True);

### Non-linear least squares

clim = (-20, 20)
show_error_maps(edinburgh_nls, title = "Edinburgh NLS", clim = clim)
show_error_maps(sydney_nls, title = "Sydney NLS", clim = clim)
show_error_maps(mcgill_nls, title = "McGill NLS", clim = clim, showcbar = True);

### Linear vs Non-linear fitting

clim = (-20, 20)
show_error_maps(edinburgh_lls, title = "Linear least squares", clim = clim)
show_error_maps(edinburgh_nls, title = "Non-linear least squares", clim = clim, showcbar = True);
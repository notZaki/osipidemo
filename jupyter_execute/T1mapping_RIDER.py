# RIDERneuro T1 mapping

import sys
sys.path.insert(0, '../src')

import original.T1_mapping.MJT_EdinburghUK.t1 as edinburgh
import original.T1_mapping.ST_SydneyAus.VFAT1mapping as sydney
import original.T1_mapping.McGill.vfa as mcgill
import matplotlib.pyplot as plt
import os
import json
import re
import urllib.request
import nibabel as nib
import mat73
import numpy as np
from numpy.linalg import norm

plt.style.use('https://gist.github.com/notZaki/8bfb049230e307f4432fd68d24c603a1/raw/c0baa2a1c55afdf1764b26ee2ebeb1cbf26d8d98/pltstyle')

def remove_leading_slash(pathstring):
    if pathstring[0] == "/":
        return pathstring[1:]
    else:
        return pathstring


def make_folder_for(file):
    folder = os.path.dirname(file)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    return


def get_manifest(manifestfile, manifesturl = "https://osf.io/nx5jw/download"):
    if not os.path.isfile(manifestfile):
        urllib.request.urlretrieve(manifesturl, manifestfile)
    with open(manifestfile) as iostream:
        manifest = json.load(iostream)
    return manifest


def get_rider(subjectid, session, filetype, datafolder = "./data"):
    # subjectid in ["01" to "19"]
    # session in [1 or 2]
    # filetype in [VFA, GRE, perf, mask]

    # Download/get manifest
    if not os.path.isdir(datafolder): 
        os.makedirs(datafolder)
    manifestfile = os.path.join(datafolder, "RIDERmanifest.json")
    manifest = get_manifest(manifestfile)

    # Find key in manifest
    matchedkeys = []
    for key in manifest.keys():
        if (f"sub-{subjectid}" in key) and (f"ses-{session}" in key) and (filetype in key):
            matchedkeys.append(key)

    # Download/get files
    files = []
    for key in matchedkeys:
        fileurl = manifest[key]
        filepath = os.path.join(datafolder, remove_leading_slash(key))
        make_folder_for(filepath)
        if not os.path.isfile(filepath):
            urllib.request.urlretrieve(fileurl, filepath)
        files.append(filepath)
        
    return files

def get_flip_angle(filename):
    return int(re.search("flip(.*?)_VFA", filename).group(1))

def get_nifti_data(filename):
    return nib.load(filename).get_fdata()

def get_tr(filename):
    return nib.load(filename).header.get_zooms()[-1]

def show_maps(fittedmaps, title = ""):
    fittedM0, fittedT1 = fittedmaps
    
    climM0 = np.nanquantile(fittedM0, [0.1, 0.98])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))
    imM0 = ax1.imshow(fittedM0, clim = climM0)
    ax1.grid(False)
    plt.sca(ax1)
    plt.title("M0")
    fig.colorbar(imM0, ax=ax1, orientation='vertical', shrink = 0.5)
    
    climT1 = np.nanquantile(fittedT1[fittedT1>0], [0.1, 0.98])
    imT1 = ax2.imshow(fittedT1, clim = climT1)
    ax2.grid(False)
    plt.sca(ax2)
    plt.title("T1")
    fig.colorbar(imT1, ax=ax2, orientation='vertical', shrink = 0.5)
    
    fig.suptitle(title)
    fig.tight_layout()
    return

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
            _M0, _T1 = mcgill.novifast(signal, fa, tr, initialvalues=[10000, 3000])
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

This demo will use a single scan from the [RIDER neuro MRI collection](https://wiki.cancerimagingarchive.net/display/Public/RIDER+NEURO+MRI).

vfafiles = get_rider(subjectid = "01", session = "1", filetype = "VFA.nii.gz")
signal = np.stack([get_nifti_data(file) for file in vfafiles], axis = -1)
flipangles = np.array([get_flip_angle(file) for file in vfafiles])
flipangles = np.deg2rad(flipangles)
tr = 4.43

# Use single slice for faster fitting
signal = signal[:,:,8,:]

## McGill

print("LLS - McGill")
%time mcgill_lls = vfa_fit(signal, flipangles, tr, author = "mcgill", fittype = "linear")
print("")

print("NLS - McGill")
%time mcgill_nls = vfa_fit(signal, flipangles, tr, author = "mcgill", fittype = "nonlinear")

show_maps(mcgill_lls, title = "Linear least squares")
show_maps(mcgill_nls, title = "Nonlinear least squares")

## Edinburgh

print("LLS - Edinburgh")
%time edinburgh_lls = vfa_fit(signal, flipangles, tr, author = "edinburgh", fittype = "linear")
print("")

# Skipping NLS because of error
# print("NLS - Edinburgh")
# %time edinburgh_nls = vfa_fit(signal, flipangles, tr, author = "edinburgh", fittype = "nonlinear")

show_maps(edinburgh_lls, title = "Linear least squares")

print("LLS - Sydney")
%time sydney_lls = vfa_fit(signal, flipangles, tr, author = "sydney", fittype = "linear")
print("")

# Skipping NLS because of error
# print("NLS - Sydney")
# %time sydney_nls = vfa_fit(signal, flipangles, tr, author = "sydney", fittype = "nonlinear")

show_maps(sydney_lls, title = "Linear least squares")
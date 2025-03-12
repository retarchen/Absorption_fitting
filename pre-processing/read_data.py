import numpy as np
import matplotlib.pyplot as plt
from astropy.io.votable import parse
from scipy.interpolate import make_interp_spline, BSpline
from astropy.convolution import Gaussian1DKernel, convolve
from scipy.optimize import curve_fit
from astropy.io import fits
from astropy import wcs
from astropy import units as u
from astropy.coordinates import SkyCoord, CIRS, AltAz
from astropy.coordinates.builtin_frames.utils import get_polar_motion
#from astropy import _erfa as erfa
from astropy.coordinates.representation import UnitSphericalRepresentation
from astropy.coordinates import ICRS, LSR
from scipy import interpolate
from scipy import integrate
from peakutils.peak import indexes
from scipy.signal import find_peaks
import glob
import os
from astropy.coordinates import Angle
import math
import itertools
import scipy.stats as stats
from scipy.integrate import trapz
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.stats import linregress

def interpo_spec(xemi,yemi):
    
    x= np.linspace(xemi.min()-1,xemi.max()+1,200)
    bins = np.digitize(xemi, x)
    averaged_yemi = np.zeros(len(x))
    for i in range(len(x)):
        indices = np.where(bins == i)[0]
        if len(indices) > 0:
            averaged_yemi[i] = np.mean(yemi[indices])
    g = Gaussian1DKernel(stddev=1)
    averaged_yemi = convolve(averaged_yemi, g)
    #averaged_yemi = savgol_filter(averaged_yemi, window_length=5, polyorder=3)
    return averaged_yemi

def emi_remove(spectra):
    num_to_remove = int(spectra.shape[1] // 4)  # Number of spectra to remove (1/4 of total)
    for _ in range(num_to_remove):
        mean_spectrum = np.mean(spectra, axis=1)
        deviations = np.sum(np.abs(spectra - mean_spectrum[:, np.newaxis]), axis=0)
        index_to_remove = np.argmax(deviations)
        spectra = np.delete(spectra, index_to_remove, axis=1)
    return spectra

def emi_clean_baseline(spectra,v):
    #s=[]
    num_to_remove = int(spectra.shape[1] // 1)  # Number of spectra to remove (1/4 of total)
    for _ in range(num_to_remove):
        mean_spectrum = np.mean(spectra, axis=1)
        deviations = np.sum(np.abs(spectra - mean_spectrum[:, np.newaxis]), axis=0)
        index= np.argmax(deviations)
        spectra[:,index]=baseline_fitting(v,spectra[:,index])
       #spectra = np.delete(spectra, index_to_remove, axis=1)
       
    '''
    for i in range(spectra.shape[1]):
        spec=spectra[:,i]
        #print(spec.shape, v.shape)
        s.append(baseline_fitting(v,spec))
    s=np.transpose(s)
   # print(s.shape)
   '''
    return spectra

def baseline_fitting(velocity, temperature, threshold=1):
    mask = np.ones_like(temperature, dtype=bool)  # Start with all data points included
    for _ in range(10):  # Limit number of iterations to prevent infinite loops
        # Fit baseline only to data points included in the mask
        slope, intercept, _, _, _ = linregress(velocity[mask], temperature[mask])
        fitted_line = slope * velocity + intercept
        
        # Calculate residuals and standard deviation of the residuals
        residuals = temperature - fitted_line
        std_residual = np.std(residuals[mask])
        
        # Update mask to exclude points outside the threshold
        new_mask = np.abs(residuals) < threshold * std_residual
        # Check if the mask has changed; if not, break the loop
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask

    return temperature-fitted_line


def read(ra,dec,filep='',R_out=8,R_in=4,emi='ring',avg_mode='raw',spec='baseline',smooth=True):
    fit = fits.open(filep)
    TB=fit[0].data
    velocity_start = fit[0].header["CRVAL3"]
    velocity_pixel = fit[0].header["CRPIX3"]
    velocity_inc = fit[0].header["CDELT3"]
    channel=len(TB)
    indices = np.arange(channel)  # Array of indices from 0 to total_length - 1
    velocity = velocity_start + (indices - velocity_pixel+1) * velocity_inc           
    a=wcs.WCS(fit[0].header)
    pix=a.all_world2pix([[ra,dec,190000]], 1)
    xe=velocity/1000
    p1_=round(pix[0,1])
    p2_=round(pix[0,0])
    if emi=='ring':
        indices = np.indices(TB.shape[1:])
       # print(indices,p1_,p2_)
        distances = np.sqrt((indices[0] - p1_)**2 + (indices[1] - p2_)**2)
        outer_mask = distances <= R_out
        inner_mask = distances <= R_in
        ring_mask = np.logical_xor(outer_mask, inner_mask)
        p=np.where(ring_mask)
       # print(p)
        #print(ring_mask.shape,TB.shape)
        
        #print(len(p),np.all(~ring_mask))
        if np.all(~ring_mask):
            ye=np.zeros(len(xe))
            yerr=np.zeros(len(xe))
        else:
            ye=TB[:, ring_mask]
            yerr=np.std(ye,axis=1)
        #print(ye.shape)
        if avg_mode=='raw':
            ye=np.average(ye,axis=1)
        elif avg_mode=='base':
            #ym=np.mean(ye,axis=1)
            #ynew=ye-np.repeat(ym[:, np.newaxis], len(ye[0,:]), axis=1)
            #err=np.std(ynew,axis=0)
            #p=int(len(err)/4*3)
            #err_=np.sort(err)[p]
            #exclude_indices = np.where(err > err_)[0]
            #include_indices = list(set(range(ye.shape[1])) - set(exclude_indices))
            #ye=ye[:,include_indices]
            
           # print(len(ye), ye)
            if np.all(~ring_mask)==False:
                #ye=emi_remove(ye)
                if ye.max()==0.:
                    ye=ye
                else:
                    if spec=='baseline':
                        ye=emi_clean_baseline(ye,xe)
                    elif spec=='remove':
                        ye=emi_remove(ye)
                    else:
                        ye=ye
                #print(np.median(ye)==np.nan, np.median(ye),np.nan)
                if smooth:
                    a = np.apply_along_axis(lambda y: interpo_spec(xe, y), 0, ye)
                    xe= np.linspace(xe.min()-1,xe.max()+1,200)
                else:
                    a=ye
                yerr=np.std(a,axis=1)
                #print(yerr.shape)
                ye=np.mean(a,axis=1)
        
    elif emi=='Non':
       # ye=TB[:,(p1_-avg_pix):(p1_+avg_pix),(p2_-avg_pix):(p2_+avg_pix)]
       # ye=np.average(ye,axis=(1,2))
        ye=TB[:,p1_,p2_]
        yerr=np.average(ye,axis=1)-np.average(ye,axis=1)
    elif emi=='square':
        ye=TB[:,(p1_-R_out):(p1_+R_out),(p2_-R_out):(p2_+R_out)]
        #print(ye.shape)
        le=int(len(ye[0,:,:])/2)
        indices = np.indices(ye.shape[1:])
        a=abs(indices[0]-le)
        b=abs(indices[1]-le)
        c=a<R_in
        d=b<R_in
        e=np.logical_and(c, d)
        e=np.logical_not(e)
        #print(e)
        ye=ye[:,e]
        yerr=np.std(ye,axis=1)
        if avg_mode=='raw':
            ye=np.average(ye,axis=1)
        elif avg_mode=='base':
            ym=np.mean(ye,axis=1)
            ynew=ye-np.repeat(ym[:, np.newaxis], len(ye[0,:]), axis=1)
            err=np.std(ynew,axis=0)
            p=int(len(err)/4*3)
            err_=np.sort(err)[p]
            exclude_indices = np.where(err > err_)[0]
            include_indices = list(set(range(ye.shape[1])) - set(exclude_indices))
            ye=ye[:,include_indices]
            yerr=np.std(ye,axis=1)
            ye=np.mean(ye,axis=1)
            
   # ye=baseline_fitting(xe, ye)
    return xe, ye,yerr

    

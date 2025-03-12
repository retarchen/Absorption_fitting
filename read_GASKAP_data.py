import numpy as np
import matplotlib.pyplot as plt
from astropy.convolution import Gaussian1DKernel, convolve
from scipy.signal import savgol_filter
from astropy.io.votable import parse
from astropy.io import fits
from astropy import wcs
from scipy.interpolate import interp1d
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D, NoOverlapError


datapathbase='/d/bip5/hchen'

def read_GASKAP_emi(ra,dec,mode='SMC',R_out=8,R_in=4,emi='ring',avg_mode='raw'):
    #print(mode,emi,R_out,R_in,avg_mode)
    if mode=='SMC':
        fit = fits.open(datapathbase+'/SMC/SMC_askap_parkes_PBC_K.fits')
        TB=fit[0].data
        velocity_start = fit[0].header["CRVAL3"]
        velocity_inc = fit[0].header["CDELT3"]
        channel=len(TB)
        velocity=np.linspace(velocity_start,velocity_start+(channel-1)*velocity_inc,channel)            
        a=wcs.WCS(fit[0].header)
        pix=a.all_world2pix([[ra,dec,190000]], 1)
        xe=velocity/1000
        p1_=round(pix[0,1])
        p2_=round(pix[0,0])
        
    elif mode=='LMC':
        fit = fits.open(datapathbase+'/LMC_ATCA+PKS_K.fits')
        TB=fit[0].data
        velocity_start = fit[0].header["CRVAL3"]
        velocity_inc = fit[0].header["CDELT3"]
        channel=120
        velocity=np.linspace(velocity_start,velocity_start+(channel-1)*velocity_inc,channel)            
        a=wcs.WCS(fit[0].header)
        pix=a.all_world2pix([[ra,dec,190000]], 1)
        xe=[]
        ye=[]
        if round(pix[0,1])<2230 and round(pix[0,0])<1997:
            xe=velocity/1000
            p1_=round(pix[0,1])
            p2_=round(pix[0,0])
        else:
            return [0],[0],[0]
    elif mode=='LMC_gaskap':
        fit = fits.open(datapathbase+'/LMC_gaskap_emi/LMC-3_askap_parkes_PBC_K.fits')
        TB=fit[0].data
        velocity_start = fit[0].header["CRVAL3"]
        velocity_inc = fit[0].header["CDELT3"]
        channel=len(TB)
        velocity=np.linspace(velocity_start,velocity_start+(channel-1)*velocity_inc,channel)            
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
        ye=TB[:, ring_mask]
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
            
    elif emi=='ring18':
        indices = np.indices(TB.shape[1:])
       # print(indices,p1_,p2_)
        distances = np.sqrt((indices[0] - p1_)**2 + (indices[1] - p2_)**2)
        outer_mask = distances <= R_out
        inner_mask = distances <= R_in
        ring_mask = np.logical_xor(outer_mask, inner_mask)
        p=np.where(ring_mask)
       # print(p)
        #print(ring_mask.shape,TB.shape)
        ye=TB[:, ring_mask]
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
    return xe, ye,yerr

def read_GASKAP_abs(name_abs,mode='SMC',emi=False):
    if mode=='SMC':
        a=parse(name_abs)
        spectra_abs=a.get_first_table().to_table(use_names_over_ids=True)
        x=np.array(spectra_abs['velocity']/1000)
        y=np.array(spectra_abs['opacity'])
        yerr=np.array(spectra_abs['sigma_opacity'])
        yemi=np.array(spectra_abs['em_mean'])
        yemi_err=np.array(spectra_abs['em_std'])
    elif mode=='LMC':
        a=parse(name_abs)
        spectra_abs=a.get_first_table().to_table(use_names_over_ids=True)
        x=np.array(spectra_abs['velocity']/1000)
        #y=np.array(spectra_abs['optical_depth'])
        y=np.array(spectra_abs['smoothed_od'])
        yerr=np.array(spectra_abs['sigma_smoothed_od'])

    if emi:
        return x, y,yerr,yemi,yemi_err
    else:
        return x, y,yerr
    
def slope_corr(x,y,err,n=3):
    y=1-y
    yerr=n*err
    ye=yerr-y
    p=np.argwhere(ye>0).flatten()
    xnew=x[p]
    ynew=y[p]   
    pol=np.polyfit(xnew,ynew,1)
    ypol=pol[0]*x+pol[1]
    ycor=y-ypol
    return 1-ycor 
    
def parse_coords(coord_string):
        """
        Convert a coordinate string in the format 'JHHMMSS±DDMMSS' to RA and Dec.

        Parameters:
        coord_string (str): The coordinate string, e.g., 'J053344-721624'.

        Returns:
        tuple: RA (in decimal degrees), Dec (in decimal degrees)
        """
        # Add colons between the components to make it parseable
        ra = coord_string[1:3] + 'h' + coord_string[3:5] + 'm' + coord_string[5:7] + 's'
        dec = coord_string[7:10] + 'd' + coord_string[10:12] + 'm' + coord_string[12:] + 's'
        
        # Create a SkyCoord object with formatted RA and Dec
        coords = SkyCoord(ra + ' ' + dec, frame='icrs')
        
        # Get RA and Dec in decimal degrees
        return coords.ra.degree, coords.dec.degree
    
def read_synchro_emi(name,mode='LMC',radius =120):
    ra,dec=parse_coords(name)
    if mode=='SMC':
        fit = fits.open(datapathbase+'/MC_synchrotron_emi/non_thermal_map_14_projected_smc.fits')
        TB=fit[0].data/88.4
    elif mode=='LMC':
        fit = fits.open(datapathbase+'/MC_synchrotron_emi/non_thermal_map_14_tau_eff_5sigma_lmc.fits')
        TB=fit[0].data/78.2
        
    wcs1 =wcs.WCS(fit[0].header)
    x, y = wcs1.wcs_world2pix(ra, dec, 0)
   # Check if the coordinates are within the image bounds
    if (x < 0 or y < 0 or x >= TB.shape[1] or y >= TB.shape[0]):
        return 0, 0  # Return zeros if outside bounds

    try:
        # Create a cutout around the position
        size = (radius / 3600) / np.abs(wcs1.wcs.cdelt[0]) * 2  # Diameter in pixels
        cutout = Cutout2D(TB, (x, y), size, wcs=wcs1, mode='partial', fill_value=np.nan)

        # Calculate the mean and std deviation, ignoring NaNs
        values = cutout.data
        values = values[~np.isnan(values)]  # Remove NaN values

        if values.size == 0:
            return 0, 0  # Return zeros if all are NaN
        else:
            return np.nanmean(values), np.nanstd(values)
    except NoOverlapError:
        return 0, 0  # Return zeros if no overlap with the image
    
def get_emi_abs_data(name,name_abs,ra,dec,mode='SMC',R_out=8,R_in=4,emi='ring',avg_m='raw',v_move=0.,save_saturated=False):
    if mode=='LMC':
        a = np.loadtxt(f"{datapathbase}/LMC_GASKAP_emi_all_abs/{name}.txt")
        x, y,yerr=read_GASKAP_abs(name_abs,mode='LMC',emi=False)
        #x=x-1
        xemi=a[:,0]
        yemi=a[:,1]
        yemi_err=a[:,2]
        p=np.argwhere(yemi_err>0.).flatten()
        xemi,yemi,yemi_err=xemi[p],yemi[p],yemi_err[p]
        #xemi=x
       # g = Gaussian1DKernel(stddev=1.5)
       # yemi = convolve(yemi, g)
       # yemi_err=convolve(yemi_err, g)
        xl=xemi.min()-1
        xh=xemi.max()+1

        
    elif mode=='SMC':
        x, y,yerr=read_GASKAP_abs(name_abs,mode='SMC',emi=False)
        xemi,yemi,yemi_err=read_GASKAP_emi(ra,dec,mode=mode,emi=emi,R_out=R_out,R_in=R_in,avg_mode=avg_m)
        #xemi=x
        xl=60
        xh=250
       
    elif mode=='LMC_gaskap_txt':
        a=np.loadtxt(datapathbase+'/LMC_gaskap_emi/LMC_GASKAP_position_emi_2/LMC_GASKAP_emi_n{:02d}.txt'.format(number_source-1))
        x, y,yerr=read_GASKAP_abs(name_abs,mode='LMC',emi=False)
        
        #x=x-1
        xemi=a[:,0]
        yemi=a[:,1]
        yemi_err=a[:,2]
    
    #y=1-y
    #x=x-4
    
    p=np.argwhere((x>xl)&(x<xh)).flatten()
    x=x[p]
    y=y[p]
    yerr=yerr[p]
    y=slope_corr(x,y,yerr)
    
    if y.min()<=0:
        #print('Satuated')
        if save_saturated:
            with open(datapathbase + '/output_data/LMC_fitting/saturated_spectra.txt', 'a') as f:
                f.write(f"{name}\n")
    #    y=1-y
    #    p=np.argwhere(y>=0.99).flatten()
    #    y[p]=0.99
    #    #y=y/y.max()-np.exp(-3)
    #    y=1-y
    x=x+v_move
    #y=-np.log(y)
    y=1-y
    #yerr=-np.log(yerr)
    
    
   # y=1-y1
    p=np.argwhere((xemi>=xl)&(xemi<=xh)).flatten()
    xemi,yemi=xemi[p],yemi[p]
    yemi_err=yemi_err[p]
    #p=np.argwhere(yemi_err>0.).flatten()
    #xemi,yemi=xemi[p],yemi[p]
    #yemi_err=yemi_err[p]
   # print(xemi.shape,x.shape)
    if mode != 'a':
        common_x = np.linspace(min(xemi.min(), x.min()), max(xemi.max(), x.max()), max(len(xemi), len(x)))
        interp_emi = interp1d(xemi, yemi, kind='linear', fill_value="extrapolate")
        yemi = interp_emi(common_x)
        interp_emi = interp1d(xemi, yemi_err, kind='linear', fill_value="extrapolate")
        yemi_err = interp_emi(common_x)
        interp_abs = interp1d(x, y, kind='linear', fill_value="extrapolate")
        y= interp_abs(common_x)
        interp_abs = interp1d(x, yerr, kind='linear', fill_value="extrapolate")
        yerr= interp_abs(common_x)
        x=common_x
        xemi=common_x

        #xemi,yemi,yemi_err=interpo_spec(x,xemi,yemi,yemi_err)
        #p=np.where(yemi_err==0.)
        #yemi_err[p]=np.max(yemi_err)/100
    
    #yemi_selferr = yemi-savgol_filter(yemi, window_length=51, polyorder=3)
    data_dict = {
    "velocity_tau": x,
    "1_e_tau": y,
    "1_e_tau_err": yerr,
    "velocity_TB": xemi,
    "TB": yemi,
    "TB_err": yemi_err
    }

    
    return data_dict
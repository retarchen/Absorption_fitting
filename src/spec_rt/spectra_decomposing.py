import numpy as np
import matplotlib.pyplot as plt
from astropy.convolution import Gaussian1DKernel, convolve
from scipy.optimize import curve_fit
from astropy.coordinates import SkyCoord
from scipy.signal import find_peaks, peak_widths
import itertools
import scipy.stats as stats
from scipy.signal import savgol_filter
from itertools import product
from gaussFitSpec import fit_spectrum
from gaussFitSpec.fitting import gaussian, multi_gaussian
from .spectra_decomposing_io import (
    load_six_column_spectrum,
    validate_absorption_input,
    write_table_outputs,
)
from .spectra_decomposing_plotting import create_legacy_axes, plot_fit_panels
from .spectra_decomposing_utils import align_spectra_grids, filter_positive_error_rows
from .spectra_decomposing_search import build_emission_model, generate_cnm_orderings


class _GaussianFitSpecAdapter:
    """Thin compatibility bridge from legacy code to ``gaussFitSpec``."""

    def __init__(self, x, y, y_err):
        self.x = x
        self.y = y
        self.y_err = y_err
        self.CF_limit = 0.97
        self.x_peak = []
        self.fit_mode = "BIC"
        self.nGaussianMax = 8
        self.bic_weight = 10
        self.num_cold = 0
        self.last_result = None
        self.fit_label = "absorption"
        self.verbose = True
        self.initial_center_window = None
        self.filter_components = False

    @staticmethod
    def gaussian_func(x, *params):
        """Evaluate one Gaussian component using ``gaussFitSpec``."""
        return gaussian(x, *params)

    def gaussian_func_multi(self, x, *gausf):
        """Evaluate summed Gaussian components using ``gaussFitSpec``."""
        return multi_gaussian(x, *gausf)

    def fitting(self):
        """Fit with ``gaussFitSpec.fit_spectrum`` and return legacy arrays."""
        method = "f_test" if self.fit_mode.lower() in {"f-test", "f_test"} else "bic"
        kwargs = {
            "name": self.fit_label,
            "method": method,
            "max_components": int(self.nGaussianMax),
            "bic_weight": float(self.bic_weight),
            "positive_amplitudes": True,
            "filter_components": bool(self.filter_components),
            "verbose": bool(self.verbose),
        }
        if len(self.x_peak) > 0:
            kwargs["initial_centers"] = np.asarray(list(self.x_peak), dtype=float)
            kwargs["initial_center_window"] = self.initial_center_window
        elif self.num_cold > 0:
            kwargs["fixed_n_components"] = int(self.num_cold)

        result = fit_spectrum(self.x, self.y, self.y_err, **kwargs)
        self.last_result = result

        popt = np.asarray(result.parameters, dtype=float)
        covariance = result.covariance
        if covariance is None or np.ndim(covariance) != 2:
            pcov = np.full_like(popt, np.nan, dtype=float)
        else:
            diagonal = np.diag(covariance)
            pcov = np.where(diagonal >= 0, np.sqrt(diagonal), np.nan)
        return popt, pcov


class SpectraDecomposing:
    """Legacy radiative-transfer decomposition workflow."""

    load_six_column_spectrum = staticmethod(load_six_column_spectrum)
    create_legacy_axes = staticmethod(create_legacy_axes)

    def __init__(self, x,y,y_err,xemi,yemi,yemi_err):
        self.x = x
        self.y = y
        self.y_err = y_err
        self.xemi = xemi
        self.yemi = yemi
        self.yemi_err = yemi_err
        self.CF_limit=0.97
        self.Tsmin=2.73
        self.F=[0,0.5,1]
        self.Tsky=2.73 #CMB
        self.v_shift=0.00001  # mannually adjust shift
        self.nGaussian=0
        self.nGaussianMax=8
        self.bic_weight=20
        self.bic_max_value=10000
        self.num_cold=0  # number of cold components assigned, if zero, means no assignment and fitting is fully automatic.
       # self.bic_weight=20
        self.peak_abs=[]
        self.peak_emi=[]
        self.fit_mode='BIC'
        self.ax=None
        self.savecsv=False
        self.name='source'
        self.datapath='./'
        self.renew=False
        self.align_data=False
        self.max_auto_warm_components=6
        self.absorption_center_window=5.0
        self.cnm_order_strategy="overlap"
        self.cnm_overlap_sigma=2.5
        self.max_cnm_orderings=720
        self.use_analytic_jacobian=True
        self.search_stats={}

    def _prepare_inputs(self):
        """Filter invalid rows, validate absorption format, and align grids if needed."""
        x, y, yerr = filter_positive_error_rows(self.x, np.copy(self.y), self.y_err)
        xemi, yemi, yemi_err = filter_positive_error_rows(self.xemi, self.yemi, self.yemi_err)
        validate_absorption_input(y, yerr)
        if self.align_data:
            x, y, yerr, xemi, yemi, yemi_err = align_spectra_grids(x, y, yerr, xemi, yemi, yemi_err)
        return x, y, yerr, xemi, yemi, yemi_err
        
    @staticmethod
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

    @staticmethod
    def calculate_noise(y,yerr,n=1):
        ye=n*yerr-y
        p=np.argwhere(ye>0).flatten()
        ye_n=yerr[p]
        
        return np.nanmean(ye_n)

    @staticmethod
    def calculate_fwhm(wavelength, intensity):
        half_max = np.max(intensity) / 2.0
        above_half_max = intensity >= half_max
        half_max_wavelengths = wavelength[above_half_max]
        fwhm = np.max(half_max_wavelengths) - np.min(half_max_wavelengths)
        return fwhm

    @staticmethod
    def sigma1_data(x,rang=30):
        err=[]
        for i in range(len(x)):
            if i<rang:
                st=0
                ed=i+rang
            elif i>(len(x)-rang):
                st=i-rang
                ed=len(x)-1
            else:
                ed=i+rang
                st=i-rang
            err.append(np.std(x[st:ed]))
        return np.array(err)
        
        
    def gaussian_func_multi(self,x, *gausf):
        """Evaluate the sum of Gaussian components via ``gaussFitSpec``."""
        return multi_gaussian(x, *gausf)
    
    @staticmethod
    def gaussian_func(x, *params):
        """Evaluate one Gaussian component via ``gaussFitSpec``."""
        return gaussian(x, *params)

    @staticmethod
    def F_test(x,y_0, y_fit1, y_fit2, sigma_rms, x_1, x_2):
        '''
        x_1,x_2: degree of freedom
        '''
        chi_square1=np.sum((y_0-y_fit1)**2/sigma_rms)/x_1
        chi_square2=np.sum((y_0-y_fit2)**2/sigma_rms)/x_2
        F=chi_square1/chi_square2*x_2/x_1
        CF=stats.f.cdf(F, x_1, x_2)
        return CF
    
    @staticmethod
    def fwhm_x_range(x, y):
        peaks, _ = find_peaks(y)
        if len(peaks) == 0:
            return np.nan  # No peaks found

        results_half = peak_widths(y, peaks, rel_height=0.3)
        left_ips = results_half[2]
        right_ips = results_half[3]

        x_left = x[left_ips.astype(int)]
        x_right = x[right_ips.astype(int)]

        x_min = np.min(x_left)
        x_max = np.max(x_right)

        return x_max - x_min

    def Gaussian_fit(self):
        """Run the full absorption plus emission decomposition workflow."""
        self.search_stats = {
            "emission_fit_attempts": 0,
            "emission_fit_successes": 0,
            "emission_fit_failures": 0,
        }
        x, y, yerr, xemi, yemi, yemi_err = self._prepare_inputs()

        peak_emi=self.peak_emi
        F=self.F
        Tsmin=self.Tsmin
        v_sh=self.v_shift
        yemi_selferr=yemi-savgol_filter(yemi, window_length=51, polyorder=3)
        #y is 1-e^(-tau)
        
        bic_limit=-1000
        ynew=np.copy(y)
        g = Gaussian1DKernel(stddev=0.5)
        ynew = convolve(ynew, g)
        
        print('start absorption fitting')
        gf=_GaussianFitSpecAdapter(x,ynew,yerr)
        gf.fit_label="smoothed"
        gf.x_peak=self.peak_abs
        gf.fit_mode=self.fit_mode
        gf.bic_weight=self.bic_weight
        gf.num_cold=self.num_cold
        gf.initial_center_window=self.absorption_center_window if len(self.peak_abs) > 0 else None
        gf.filter_components=len(self.peak_abs) > 0
        popt_,pcov_=gf.fitting()

        #print(popt_[1::3])
        if y.max()>=1.:
            print('Satuated')
            p=np.argwhere(y>=0.99).flatten()
            y[p]=y[p]/y.max()*0.99
            #y=y/y.max()-np.exp(-3)
        y=-np.log(1-y)
        
        gf=_GaussianFitSpecAdapter(x,y,yerr)
        gf.fit_label="tau"
        gf.x_peak=popt_[1::3]
        gf.fit_mode=self.fit_mode
        gf.bic_weight=self.bic_weight
        gf.num_cold=self.num_cold
        gf.initial_center_window=self.absorption_center_window
        gf.filter_components=True
        popt,pcov=gf.fitting()
        
        #popt[1]=popt[1]-3
        popt_ori=np.copy(popt)
        print('Absorption fitting finished')
        ncold=int(len(popt)/3)
        p0=np.zeros(ncold*2)+self.Tsmin
        for _ in range(ncold):
            p0[_]=popt[3*_+1]
        
        Ts_low_lim=[]
        for i in range(ncold):
            j=i*3
            _popt=popt[j:j+3]
            p=np.argmin(abs(_popt[1]-xemi))
            Ts_low_lim.append(yemi_err[p]/(1-np.exp(-y[p])))
        #print('Ts_low limit', Ts_low_lim)

        #print(x,xemi)
        if len(peak_emi)>0:
            #print(xemi[peak_emi],peak_emi)
            nwarm=len(peak_emi)
            for i in range(len(peak_emi)):
                p0=np.append(p0,[np.max(yemi), peak_emi[i], 1])
            auto=False
        else:
            p=np.argmax(yemi)
            nwarm=1
            p0=np.append(p0,[np.max(yemi), xemi[p], 1]) 
            auto=True
            
        print('start emission fitting')
        def loop(p0,ncold, xemi, yemi,yemi_err,popt,F=F,nwarm=nwarm,Tsmin=Tsmin):
            #print(p0)
            popt2_=[]
            res=[]
            wf=[]
            sigma_Tsf,all_Tsf=[],[]
            funTexp=[]
            order=[]
            fit_err=[]
            v_shift=[]
            lowbound=np.array([0. for _ in range(len(p0))])
            #lowbound=np.array([-np.inf for _ in range(len(p0))])
            for _ in range(ncold):
                lowbound[_]=p0[_]-v_sh
            _=2*ncold
            lowbound[ncold:_]=Tsmin
            for _ in range(nwarm):
                lowbound[2*ncold+_*3+1]=xemi.min()
                ind=np.argmin(np.abs(x - p0[2*ncold+_*3+1]))
                lowbound[2*ncold+_*3]=np.mean(yemi_err[max(ind-20,0):min(ind+20,len(xemi)-1)])
                #lowbound[2*ncold+_*3]=np.max(yemi_err)*0.5
                p0[2*ncold+_*3]=np.mean(yemi_err[max(ind-20,0):min(ind+20,len(xemi)-1)])*1+1
                #p0[2*ncold+_*3]=np.max(yemi_err)*0.5+1
                #lowbound[2*ncold+_*3]=calculate_noise(yemi,yemi_err,n=3)*2
            highbound=np.array([np.inf for _ in range(len(p0))])
            for _ in range(ncold):
                highbound[_]=p0[_]+v_sh
            for _ in range(ncold):
                __=_*3
                _popt=popt[__:__+3]
                fwhm_=2.35482*_popt[2]
                max_y2 = np.max(yemi[(xemi >= xemi[np.argmax(y)] - v_sh) & (xemi <= xemi[np.argmax(y)] + v_sh)])
                _1=max_y2/(1-np.exp(-_popt[0]))-Tsmin*(np.exp(-_popt[0])-1)
                cons=min(_1+_1*0.3,21.866*fwhm_**2)
                if cons>Tsmin:
                    highbound[ncold+_]=cons
                else:
                    highbound[ncold+_]=Tsmin+100
            for _ in range(nwarm):
                highbound[2*ncold+_*3+1]=xemi.max()
                highbound[2*ncold+_*3]=np.max(yemi)+ 3*np.max(yemi_err)
                highbound[2*ncold+_*3+2]=max(np.ptp(xemi[yemi > 4 * yemi_err]) if np.any(yemi > 4 * yemi_err) else 0,
                                             self.fwhm_x_range(xemi,yemi))

            mask = (p0 > highbound) | (p0 < lowbound)
            p0[mask] = (highbound[mask] + lowbound[mask]) / 2
           # print(highbound,lowbound,p0)
            CNMsequences = generate_cnm_orderings(
                popt,
                strategy=self.cnm_order_strategy,
                overlap_sigma=self.cnm_overlap_sigma,
                max_orderings=self.max_cnm_orderings,
            )
            Fsequences = np.array(list(itertools.product(F, repeat=nwarm)))
            self.search_stats["cold_components"] = ncold
            self.search_stats["warm_components"] = nwarm
            self.search_stats["cnm_orderings"] = len(CNMsequences)
            self.search_stats["fraction_combinations"] = len(Fsequences)
            self.search_stats["fits_per_loop"] = len(CNMsequences) * len(Fsequences)
            for cn in range(len(CNMsequences)):
                for i in range(len(Fsequences)):
                    T_exp, T_jac = build_emission_model(
                        xemi,
                        popt,
                        CNMsequences[cn],
                        Fsequences[i],
                        self.Tsky,
                    )
                    try:
                        self.search_stats["emission_fit_attempts"] += 1
                        pop_, pcov = curve_fit(
                            T_exp,
                            xemi,
                            yemi,
                            p0=p0,
                            bounds=(lowbound,highbound),
                            jac=T_jac if self.use_analytic_jacobian else None,
                            maxfev=120000,
                        )
                    except RuntimeError as e:
                        self.search_stats["emission_fit_failures"] += 1
                        print(f"Fit failed: {e}")
                        continue
                    self.search_stats["emission_fit_successes"] += 1
                    pcov_=np.sqrt(np.diag(pcov))
                # print(pcov_)
                    popt2_.append(pop_)
                    funTexp.append(T_exp(xemi, *pop_))
                    wf.append(1/np.std((yemi-T_exp(xemi, *pop_)))**2)
                    v_ori=np.array([popt_ori[3*_+1] for _ in range(ncold)])
                    v_shift.append(pop_[:ncold]-v_ori)
                    _=2*ncold
                    sigma_Tsf.append(pcov_[ncold:_])
                    all_Tsf.append(pop_[ncold:_])
                    order.append(CNMsequences[cn])
                    fit_err.append(pcov_)
                    
                    
                    residuals=yemi-T_exp(xemi, *pop_)
                    chi2 = np.nansum((residuals / yemi_err) ** 2)
                    k = len(pop_)
                    n = len(yemi)
                    #print(k,n,chi2,yemi_err.min())
                    bic = k * np.log(n) + chi2
                    
                    _pc=pcov_
                    #print('pcov',_pc)
                    count = np.sum(np.array(_pc)>50)
                    p=np.argwhere(_pc>50).flatten()
                    
                    _b=bic/3+np.sum(_pc[p]/10)
                    bic+=_b*count
                    #if len(p)>0:
                    #    print('large error bars:',_pc[p])
                    #    print('bic',bic)
                    #bic +=np.sum(_pc[p]/2)
                    
                    count = np.sum(np.array(_pc)/np.array(pop_)>0.5)
                   # if count>0:
                   #     print(bic,count)
                    _b=bic/2
                    bic+=_b*count

                    a=pop_[_:]
                    #count = np.sum(a[0::3]<self.calculate_noise(yemi,yemi_err)*3)
                    count = np.sum(a[0::3]<max(np.max(yemi_err),self.calculate_noise(yemi,yemi_err)*3))
                    #print(a[0::3],calculate_noise(yemi,yemi_err),count)
                    #print(count)
                    #if count>0:
                    #    print(count)
                    _b=20
                    bic +=_b* count
                    #print(k,n,chi2,bic)
                    #print(pcov_,bic)
                    res.append(bic)
                # print(i,cn)
                    #res.append(np.sqrt(np.sum((yemi-T_exp(xemi, *pop_))**2)))
            #print(np.array(res),p0)
                    
            return (res,popt2_,funTexp,Fsequences,wf,np.array(sigma_Tsf),
                    np.array(all_Tsf),np.array(order),np.array(fit_err),np.array(v_shift))
        
        if auto:
            if self.fit_mode=='F_test':
                CF=1
                p0_1=p0
                n_=1
                res1p=1000
                resye=np.sqrt(np.sum(yemi_selferr**2))
                #print('1,',CF,C_Flim, res1p,resye)
                print(resye)
                amp2=np.max(yemi)
                restd2=np.std(yemi_selferr)
                while CF>self.C_Flim and n_<self.nGaussianMax:
            # while CF>C_Flim and res1p>resye and n_<7:
                #while CF>C_Flim and amp2>2*restd2 and n_<7:
                    res1,popt2_1,funTexp1,Fsequences1,wf1,sigma_Tsf1,all_Tsf1,order1,fit_err1,v_shift1=loop(p0_1,ncold, xemi, 
                                                                                                            yemi,
                                                                                yemi_err,popt,nwarm=nwarm)
                    _=2*ncold
                    Ts_1 = [popt2_1[i][ncold:_] for i in range(len(popt2_1))]
                    score_1=[]
                    for j in range(len(res1)):
                        score_=0
                        score_+=res1[j]
                        for i in range(ncold):
                            if Ts_1[j][i]>10:
                                Ts_score=-res1[j]/100
                            else:
                                Ts_score=0
                            score_+=Ts_score
                        score_1.append(score_)
                    p=np.argmin(res1)
                    res1p=res1[p]
                    funT1=funTexp1[p]
                    g = Gaussian1DKernel(stddev=3)
                    _res=yemi-funT1
                    _res= convolve(_res, g)
                    p_=np.argmax(_res)
                    p0_2=np.append(p0_1,[1, xemi[p_], 1]) 
                    #print('p:',p0_2)
                    nwarm=nwarm+1
                    res2,popt2_2,funTexp2,Fsequences2,wf2,sigma_Tsf2,all_Tsf2,order2,fit_err1,v_shift1=loop(p0_2,ncold, xemi, yemi,
                                                                                yemi_err,popt,nwarm=nwarm)
                    _=2*ncold
                    Ts_2 = [popt2_2[i][ncold:_] for i in range(len(popt2_2))]
                    score_2=[]
                    for j in range(len(res2)):
                        score_=0
                        score_+=res2[j]
                        for i in range(ncold):
                            if Ts_2[j][i]>10:
                                Ts_score=-res2[j]/100
                            else:
                                Ts_score=0
                            score_+=Ts_score
                        score_2.append(score_)
                    p=np.argmin(res2)
                    funT2=funTexp2[p]
                    CF=self.F_test(xemi,yemi, funT1, funT2, yemi_err, len(xemi)-len(p0_1), len(xemi)-len(p0_2))
                    n_+=1
                    p0_1=p0_2
                    restd2=np.sqrt(res2[p]**2/(len(yemi-1)))
                    amp2 = min(popt2_2[p][ncold + i * 3] for i in range(nwarm))
                    print(CF,n_,res1p)
                res,popt2_,funTexp,Fsequences=res1,popt2_1,funTexp1,Fsequences1
                wf,sigma_Tsf,all_Tsf=wf1,sigma_Tsf1,all_Tsf1
                order=order1
                fit_err=fit_err1
                v_shift=v_shift1
            elif self.fit_mode=='BIC':
                print('Fitting mode: BIC')
                def fit_and_calculate_bic(x, y, y_error):
                    p0_1=np.zeros(ncold*2)+Tsmin
                    for _ in range(ncold):
                        p0_1[_]=popt[3*_+1]
                    pe, _ =find_peaks(y, height=np.max(y)/7, distance=5)
                    #print('_pe',x[pe])
                    index=np.argsort(y[pe])[::-1]
                    pe=pe[index]
                    _pe = min(6, len(pe))
                    #print('peak',len(pe),x[pe],y[pe])
                    res1,popt2_1,funTexp1,Fsequences1,wf1,sigma_Tsf1,all_Tsf1,order1,fit_err1,v_shift1=loop(p0_1,ncold, x, 
                                                                                    y,y_error,popt,nwarm=0)
                    _=2*ncold
                    Ts_1 = [popt2_1[i][ncold:_] for i in range(len(popt2_1))]
                    score_1 = [ res1[j] + sum(-5 if Ts_1[j][i] > 10 else 0 for i in range(ncold))
                                            for j in range(len(res1))]
                    p=np.argmin(score_1)
                    _bbic=score_1[p]
                    best_bic=score_1[p]
                    best_mean_score=np.mean(score_1)
                    _mmean_score=np.mean(score_1)
                    b_pos=-1
                    print('BIC ',_bbic,'Mean_score ', _mmean_score,'nemi=',0)
                    improving = True
                    nwarm=1
                    best_p0_1 = np.copy(p0_1)
                    best_nwarm = 0
                    y_res=y
                    num=0
                    lim=1
                    #while improving or num<=lim:
                    bic_limit=500
                    while improving and (
                        self.max_auto_warm_components is None
                        or nwarm <= self.max_auto_warm_components
                    ):
                        #print(improving, num,lim)
                        _bbic=self.bic_max_value
                        _mmean_score=self.bic_max_value
                        try:
                            #print('_pe',x[pe])
                            for pos in range(_pe):
                                _pp=pe[pos]
                                _p0_1=np.append(p0_1,[np.max(y_error), x[_pp], 1])
                                #print('_p0_1',_p0_1)
                                res1,popt2_1,funTexp1,Fsequences1,wf1,sigma_Tsf1,all_Tsf1,order1,fit_err1,v_shift1=loop(_p0_1,
                                                                                                                        ncold, x, 
                                                                                            y,y_error,popt,nwarm=nwarm)
                                _=2*ncold
                                Ts_1 = [popt2_1[i][ncold:_] for i in range(len(popt2_1))]
                                score_1 = [ res1[j] + sum(-10 if Ts_1[j][i] > 10 else 0 for i in range(ncold))
                                            for j in range(len(res1))]

                                #print('score_1',np.array(score_1))
                                # identify large mean bic but small minimum bic
                                score_1=np.array(score_1)
                                p=np.argwhere(score_1<self.bic_max_value).flatten()
                                if len(p)>0:
                                    score_1=score_1[p]
                                else:
                                    print('too large BIC value')
                                
                               # print('sc',score_1)
                                #if np.mean(score_1)-np.min(score_1)>100 and np.min(score_1)<best_bic:
                                if np.min(score_1)<best_bic and np.mean(score_1)-np.min(score_1)>100:
                                    #print('True')
                                    #print('>300')
                                    #print(np.min(score_1),np.max(score_1),np.mean(score_1))
                                    bic_limit=best_mean_score+40
                                    _p=np.argwhere(score_1<bic_limit).flatten()
                                    score_1=score_1[_p]
                                    #print(np.min(score_1),np.max(score_1),np.mean(score_1))
                                    #print(score_1[score_1<np.mean(score_1)])

                                #print('sc',score_1)
                                p=np.argmin(score_1)
                                _bic=score_1[p]
                                #print('_bic',_bic)
                                _mean_score=np.mean(score_1)

                                #if _bic-_bbic<.5 and _mean_score-_mmean_score<10:
                                #print(_bic-_bbic,_mean_score-_mmean_score)
                                if _bic-_bbic<.1 and _mean_score-_mmean_score<10:
                                    #print('BIC ',_bic, 'Mean_score ', _mean_score)
                                    #if np.sum(fit_err1/popt2_1>0.5)>0:
                                    #    print('BIC ',_bic, 'Mean_score ', _mean_score,'fit_err',fit_err1)
                                    _bbic = _bic
                                    b_pos=pos
                                    _mmean_score=_mean_score
                                    funT1=funTexp1[p]
                                    g = Gaussian1DKernel(stddev=2)
                                    _res=yemi-funT1
                                    y_res= convolve(_res, g)
                                    

                        except RuntimeError as e:
                            print(f"Fit failed for Gaussians: {e}")

                        if b_pos < 0:
                            print("No acceptable warm-component candidate; stopping automatic search")
                            improving = False
                            break

                        nwarm+=1
                        p0_1=np.append(p0_1,[np.max(y_error), x[pe[b_pos]], 1])
                        bic=_bbic
                        mean_score=_mmean_score
                        pe0, _ =find_peaks(y_res, height=np.max(y_res)/5, distance=5)
                        pe=np.append(pe,pe0)
                        pe=np.unique(pe)
                        index=np.argsort(y[pe])[::-1]
                        pe=pe[index]
                        _pe = min(7, len(pe))
                        b_pos=-1
                        
                        print('BIC ',bic, 'Mean_score ', mean_score,'nemi=',nwarm-1)
                       #print('bestBIC ',best_bic, 'Mean_score ', best_mean_score)
                        
                        
                        if (bic-best_bic)<.1 and (mean_score-best_mean_score)<10:
                           # a=np.array(fit_err1)
                           # b=np.array(popt2_1)
                            #print('BIC ',_bic, 'Mean_score ', _mean_score,'fit_err',a/b)
                        #if bic< best_bic+.1:
                            best_bic = bic
                            best_mean_score=mean_score
                            best_p0_1 = np.copy(p0_1)
                            best_nwarm = nwarm - 1
                            if not improving:
                                num=0
                                lim=0
                        else:
                            # If the BIC did not improve, stop fitting additional Gaussians
                            num+=1
                        # if num>=lim and lim<2:
                        #     lim+=1
                            improving = False
                            
                       # print(improving)

                            
                        

                    #print(p0_1)
                # print(p0_1,nwarm)
                    #back=-3*(lim+1)
                    final_p0_1 = best_p0_1
                    final_nwarm = best_nwarm
                    if final_nwarm == 0 and len(p0_1) > 2 * ncold:
                        final_p0_1 = p0_1[:-3]
                        final_nwarm = max(nwarm - 2, 0)
                    back=-3
                # print(back,p0_1[:back])
                # res1,popt2_1,funTexp1,Fsequences1,wf1,sigma_Tsf1,all_Tsf1,order1,fit_err1,v_shift1=loop(p0_1[:back],ncold, x, 
                #                                                                        y,y_error,popt,nwarm=nwarm-2-lim)
                    res1,popt2_1,funTexp1,Fsequences1,wf1,sigma_Tsf1,all_Tsf1,order1,fit_err1,v_shift1=loop(final_p0_1,ncold, x, 
                                                                                        y,y_error,popt,nwarm=final_nwarm)
         
                # print('res1',np.array(res1))
                #  print(res1)

                    return res1,popt2_1,funTexp1,Fsequences1,wf1,sigma_Tsf1,all_Tsf1,order1,fit_err1,v_shift1,bic_limit
                res,popt2_,funTexp,Fsequences,wf,sigma_Tsf,all_Tsf,order,fit_err,v_shift,bic_limit=fit_and_calculate_bic(xemi, yemi, yemi_err)

            
        else:
            print('Fitting mode: mannual')
            resye=np.sqrt(np.sum(yemi_selferr**2))
            #print(resye)
            
            p0_=p0[2*ncold:]
            t0=p0[:2*ncold]
            num = len(p0_) // 3
            #modifications = [
            #    [p0_[i * 3 + 1] - 4, p0_[i * 3 + 1] + 4] for i in range(num)
            #]
            
            modifications = [
                [p0[i * 3 + 1] - 4, 
                 p0[i * 3 + 1], 
                 p0[i * 3 + 1] + 4] for i in range(num)
            ]
            # Generate all combinations of the modified second elements
            all_combinations = list(product(*modifications))
            original_p0_combination = tuple(p0_[i * 3 + 1] for i in range(num))
            all_combinations.append(original_p0_combination)
        # print(all_combinations)
            # Evaluate all combinations to find the minimum BIC
            results = []
            for combination in all_combinations:
                new_p0 = p0_.copy()
                for i in range(num):
                    new_p0[i * 3 + 1] = combination[i]  # Update second elements
                new_p0=np.concatenate((t0, new_p0))
                results.append(loop(new_p0,ncold, xemi, yemi,yemi_err,popt))

            res,popt2_,funTexp,Fsequences,wf,sigma_Tsf,all_Tsf,order,fit_err,v_shift= min(results, key=lambda r: r[0])
            #print('BIC=',bic)
            
            #res,popt2_,funTexp,Fsequences,wf,sigma_Tsf,all_Tsf,order,fit_err,v_shift=loop(p0,ncold, xemi, yemi,yemi_err,popt)
        
        
        _=2*ncold
        Ts_1 = [popt2_[i][ncold:_] for i in range(len(popt2_))]
        #print(Ts_1)
        
        score_1=[]
        for j in range(len(res)):
            score_=0
            score_+=res[j]
            for i in range(ncold):
                if Ts_1[j][i]>10:
                    Ts_score=-10
                else:
                    Ts_score=0
                score_+=Ts_score
            score_1.append(score_)
        res=np.array(res)
        
        

        score_1,popt2_,funTexp,Fsequences,wf,sigma_Tsf,all_Tsf=(np.array(score_1),np.array(popt2_),
                                                                np.array(funTexp),np.array(Fsequences),
                                                                np.array(wf),np.array(sigma_Tsf),np.array(all_Tsf))
        #print(score_1.shape,popt2_.shape,funTexp.shape,Fsequences.shape,wf.shape,sigma_Tsf.shape,all_Tsf.shape)
        
        p=np.argwhere(score_1<self.bic_max_value).flatten()
        if len(p)>0:
            score_1,popt2_,funTexp,wf,sigma_Tsf,all_Tsf,order,fit_err,v_shift,res=score_1[p],popt2_[p],funTexp[p],wf[p],sigma_Tsf[p],all_Tsf[p],order[p],fit_err[p],v_shift[p],res[p]
            
        if len(score_1)>3:
            
            if np.mean(score_1)-np.min(score_1)>100 and bic_limit>0:
                _p=np.argwhere(score_1<(bic_limit)).flatten()
            else:
                _p=np.argwhere(score_1<(np.mean(score_1)+3*np.std(score_1))).flatten()
            #print(score_1,_p,np.mean(score_1)+np.std(score_1))
            score_1,popt2_,funTexp,wf,sigma_Tsf,all_Tsf,order,fit_err,v_shift,res=(score_1[_p],popt2_[_p],funTexp[_p],wf[_p],sigma_Tsf[_p],
                                                                                   all_Tsf[_p],order[_p],fit_err[_p],v_shift[_p],res[_p])
            #print(score_1)
        #print(all_Tsf.shape)
        #for i in range(ncold):
        #    ___p=np.argwhere(all_Tsf[:,i]>3.77).flatten()
        #    print('Ts>3.77', ___p.shape)
        #    if len(___p)>0:
        #        score_1,popt2_,funTexp,wf,sigma_Tsf,all_Tsf,order,fit_err,v_shift,res=(score_1[___p],popt2_[___p],funTexp[___p],
        #                                                                wf[___p],sigma_Tsf[___p],all_Tsf[___p],
        #                                                                order[___p],fit_err[___p],v_shift[___p],res[___p])
        #    else:
        #        ___p=np.argwhere(all_Tsf[:,i]>3.77).flatten()
        #        print('Ts>3.77', ___p.shape)
        #        if len(___p)>0:
        #            score_1,popt2_,funTexp,wf,sigma_Tsf,all_Tsf,order,fit_err,v_shift,res=(score_1[___p],popt2_[___p],funTexp[___p],
        #                                                                    wf[___p],sigma_Tsf[___p],all_Tsf[___p],
        #                                                                    order[___p],fit_err[___p],v_shift[___p],res[___p])
        #        else:
        #            ___p=np.argwhere(all_Tsf[:,i]>0).flatten()
        #           print('Ts>0', ___p.shape)
        #            if len(___p)>0:
        #                score_1,popt2_,funTexp,wf,sigma_Tsf,all_Tsf,order,fit_err,v_shift,res=(score_1[___p],popt2_[___p],funTexp[___p],
        #                                                                        wf[___p],sigma_Tsf[___p],all_Tsf[___p],
        #                                                                        order[___p],fit_err[___p],v_shift[___p],res[___p])
            #print('Ts>10 num', ___p.shape)
        p=np.argmin(score_1)
        print('final BIC ', score_1[p],'mean score ', np.mean(score_1) ,'nemi=',len(popt2_[p][2*ncold:])//3)
        #print(score_1[p])
        print('velocity shift',v_shift[p])
    # print('all_Tsf',all_Tsf)
    
        mean_Ts,sigma_meanTsf=[],[]
        for i in range(ncold):
            if len(all_Tsf[:,i])>1:
                m_=np.sum(wf*all_Tsf[:,i])/np.sum(wf)
                mean_Ts.append(m_)
                sigma_meanTsf.append(np.sqrt(np.sum(wf*(all_Tsf[:,i]-m_)**2+wf*sigma_Tsf[:,i]**2)/np.sum(wf)*len(wf)/(len(wf)-1)))
            else:
                mean_Ts.append(float(np.ravel(all_Tsf[:, i])[0]))
                sigma_meanTsf.append(float(np.ravel(sigma_Tsf[:, i])[0]))
            
        v_shi=v_shift[p]
        popt2=popt2_[p]
        _=2*ncold
        Ts=popt2[ncold:_]
        Ts_err=sigma_Tsf[p]
        gausf=popt2[_:]
        funT=funTexp[p]
        _F=Fsequences[p%len(Fsequences)]
        Tfit_err=self.sigma1_data(yemi-funT)
        Or=order[p]
        fit_e=fit_err[p]
        fit_e=fit_e[_:]
        #print('fiterr',fit_err)

        #if self.selfabs:
        #    p1=np.argwhere(y==y.max()).flatten()
        #    a_=abs(xemi-x[p1])
        #    p=np.argmin(a_)
        #    Ton=yemi[p]
        #    Toff=funT[p]
        #    p2=[]
        #    for i in range(int(len(gausf)/3)):
        #        p2.append((i+1)*3-2)
        #    p3=np.argmin(abs(gausf[p2]-xemi[p]))
        #    Ts=(Ton-Toff)/y[p1]+gausf[(p3+1)*3-3]
            
        return popt_ori,pcov,popt2,Ts,Ts_err,gausf,funT,Tfit_err,Or,fit_e,mean_Ts,sigma_meanTsf,nwarm,v_shi,_F

    def fit_and_plot(self):
        """Fit the spectra and draw the legacy four-panel diagnostic figure."""
        if self.ax is None:
            _, self.ax = create_legacy_axes()

        x, y, yerr, xemi, yemi, yemi_err = self._prepare_inputs()
        name = self.name
        fit_output = self.Gaussian_fit()
        (
            popt,
            pcov,
            popt2,
            Ts,
            Ts_err,
            gausf,
            funT,
            Tfit_err,
            Or,
            fit_e,
            mean_Ts,
            sigma_meanTsf,
            nwarm,
            v_shift,
            F_values,
        ) = fit_output
        popt_ori = np.copy(popt)

        if self.savecsv:
            write_table_outputs(
                self,
                name=name,
                popt=popt,
                pcov=pcov,
                gausf=gausf,
                funT=funT,
                xemi=xemi,
                Or=Or,
                fit_e=fit_e,
                mean_Ts=mean_Ts,
                sigma_meanTsf=sigma_meanTsf,
                v_shift=v_shift,
                F_values=F_values,
            )

        plot_fit_panels(
            self,
            ax=self.ax,
            x=x,
            y=y,
            yerr=yerr,
            xemi=xemi,
            yemi=yemi,
            yemi_err=yemi_err,
            name=name,
            popt_ori=popt_ori,
            popt=popt,
            Ts=Ts,
            gausf=gausf,
            funT=funT,
            mean_Ts=mean_Ts,
            sigma_meanTsf=sigma_meanTsf,
            nwarm=nwarm,
            v_shift=v_shift,
            F_values=F_values,
        )
        return fit_output

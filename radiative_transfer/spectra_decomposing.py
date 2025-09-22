import numpy as np
import matplotlib.pyplot as plt
from astropy.convolution import Gaussian1DKernel, convolve
from scipy.optimize import curve_fit
from astropy.coordinates import SkyCoord
from scipy.signal import find_peaks, peak_widths
import itertools
import scipy.stats as stats
from scipy.integrate import trapz
from scipy.signal import savgol_filter
from .Gaussian_fitting import GaussianFitting as Gf  
import pandas as pd
import os
from itertools import product

linestyle_plot=['--','dotted','-.','solid','--','dotted','-.','solid']
#datapathbase='/d/bip5/hchen'

class SpectraDecomposing:
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
        fun=np.zeros(len(x))
        for i in range(int(len(gausf)/3)):
            j=i*3
            fun+= self.gaussian_func(x,*gausf[j:j+3])
        return fun
    
    @staticmethod
    def gaussian_func(x, *params):
        a,mu, sigma = params
        return a * np.exp(-(x - mu)**2 / (2 * sigma**2))

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
  #  def Gaussian_fit(x,y,yerr,xemi,yemi,yemi_err,peak_abs=[],peak_emi=[],F=[0,0.5,1],selfabs=False,
  #                          C_Flim=0.97,Tsmin=9,iteration_max=7,fit_mode='F_test',v_sh=4.):
        x,y,yerr,xemi,yemi,yemi_err=self.x,np.copy(self.y),self.y_err,self.xemi,self.yemi,self.yemi_err
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
        
        gf=Gf(x,ynew,yerr)
        gf.x_peak=self.peak_abs
        gf.bic_weight=self.bic_weight
        gf.num_cold=self.num_cold
        popt_,pcov_=gf.fitting()

       # popt_,pcov_=Gf.fitting(x,ynew,yerr,x_peak=self.peak_abs)
        print(popt_[1::3])
        if y.max()>=1.:
            print('Satuated')
            p=np.argwhere(y>=0.99).flatten()
            y[p]=y[p]/y.max()*0.99
            #y=y/y.max()-np.exp(-3)
        y=-np.log(1-y)
        
        gf=Gf(x,y,yerr)
        gf.x_peak=popt_[1::3]
        gf.bic_weight=self.bic_weight
        gf.num_cold=self.num_cold
        popt,pcov=gf.fitting()
        
        #popt,pcov=Gf.fitting(x,y,yerr,x_peak=popt_[1::3])
        #popt[1]=popt[1]-3
        popt_ori=np.copy(popt)
        print('popt_ori',popt_ori)
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
                lowbound[2*ncold+_*3]=np.max(yemi_err[max(ind-20,0):min(ind+20,len(xemi)-1)])
                #lowbound[2*ncold+_*3]=np.max(yemi_err)*0.5
                p0[2*ncold+_*3]=np.max(yemi_err[max(ind-20,0):min(ind+20,len(xemi)-1)])*1+1
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
            values = np.arange(0, ncold)
           # print(p0,lowbound,highbound)
            CNMsequences = np.array(list(itertools.permutations(values, ncold)))
            Fsequences = np.array(list(itertools.product(F, repeat=nwarm)))
            for cn in range(len(CNMsequences)):
                for i in range(len(Fsequences)):
                    def T_exp(x,*pa):
                        _vp=pa[:ncold]
                        _=2*ncold
                        Ts=pa[ncold:_]
                        para=pa[_:]
                        T_CNM = np.zeros_like(x)
                        CN_=CNMsequences[cn]
                        for j in range(len(CN_)):
                            index=CN_[j]*3
                            if j==0:
                                tau_=0
                            else:
                                tau_=np.zeros_like(x)
                                for l in range(j):
                                    index2=CN_[l]*3
                                    #tau_+=np.log(1-gaussian_func(x,*popt[index2:index2+3]))
                                    tau_+=self.gaussian_func(x,*popt[index2:index2+3])
                            #T_CNM+=gaussian_func(x,*popt[index:index+3])*Ts[CN_[j]]*np.exp(tau_)
                            _popt_=popt[index:index+3]
                            _popt_[1]=_vp[CN_[j]]
                            #print(_popt_,x,gaussian_func(x,*_popt_))
                            T_CNM+=(1-np.exp(-self.gaussian_func(x,*_popt_)))*(Ts[CN_[j]])*np.exp(-tau_)
                        T_WNM = np.zeros_like(x)
                        F_=Fsequences[i]
                        for m in range(len(F_)):
                            i_=m*3
                            #T_WNM+=(F_[m] + (1 - F_[m]) *(1-gaussian_func_multi(x,*popt))) * gaussian_func(x, *para[i_:i_+3])
                            T_WNM+=(F_[m] + (1 - F_[m]) *np.exp(-self.gaussian_func_multi(x,*popt))) * self.gaussian_func(x, *para[i_:i_+3])
                        #print(np.exp(-gaussian_func_multi(x,*popt)))
                        T_WNM=T_WNM+self.Tsky*(np.exp(-self.gaussian_func_multi(x,*popt))-1)
                        return T_CNM+T_WNM
                    try:
                        pop_, pcov = curve_fit(T_exp, xemi, yemi,p0=p0,bounds=(lowbound,highbound),maxfev=120000)
                    except RuntimeError as e:
                        print(f"Fit failed: {e}")
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
                    count = np.sum(np.array(_pc)>50)
                    p=np.argwhere(_pc>50).flatten()
                    _b=bic/3+np.sum(_pc[p]/10)
                    bic+=_b*count
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
                    score_1 = [ res1[j] + sum(-5 if Ts_1[j][i] > 3.77 else 0 for i in range(ncold))
                                            for j in range(len(res1))]
                    p=np.argmin(score_1)
                    _bbic=score_1[p]
                    best_bic=score_1[p]
                    best_mean_score=np.mean(score_1)
                    _mmean_score=np.mean(score_1)
                    b_pos=-1
                    print('origin BIC ',_bbic,'Original_Mean_score ', _mmean_score)
                    improving = True
                    nwarm=1
                    y_res=y
                    num=0
                    lim=1
                    #while improving or num<=lim:
                    bic_limit=500
                    while improving:
                        #print(improving, num,lim)
                        _bbic=100000
                        _mmean_score=1000000
                        try:
                            for pos in range(_pe):
                                _pp=pe[pos]
                                _p0_1=np.append(p0_1,[np.max(y_error), x[_pp], 1])
                                #print('_p0_1',_p0_1)
                                res1,popt2_1,funTexp1,Fsequences1,wf1,sigma_Tsf1,all_Tsf1,order1,fit_err1,v_shift1=loop(_p0_1,
                                                                                                                        ncold, x, 
                                                                                            y,y_error,popt,nwarm=nwarm)
                                _=2*ncold
                                Ts_1 = [popt2_1[i][ncold:_] for i in range(len(popt2_1))]
                                score_1 = [ res1[j] + sum(-5 if Ts_1[j][i] > 10 else 0 for i in range(ncold))
                                            for j in range(len(res1))]

                                #print('score_1',np.array(score_1))
                                # identify large mean bic but small minimum bic
                                score_1=np.array(score_1)
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

                        nwarm+=1
                        p0_1=np.append(p0_1,[np.max(y_error), x[pe[b_pos]], 1])
                        bic=_bbic
                        mean_score=_mmean_score
                        pe0, _ =find_peaks(y_res, height=np.max(y_res)/5, distance=5)
                        pe=np.append(pe,pe0)
                        index=np.argsort(y[pe])[::-1]
                        pe=pe[index]
                        _pe = min(7, len(pe))
                        b_pos=-1
                        
                        print('BIC ',bic, 'Mean_score ', mean_score)
                       #print('bestBIC ',best_bic, 'Mean_score ', best_mean_score)
                        
                        
                        if (bic-best_bic)<.1 and (mean_score-best_mean_score)<10:
                           # a=np.array(fit_err1)
                           # b=np.array(popt2_1)
                            #print('BIC ',_bic, 'Mean_score ', _mean_score,'fit_err',a/b)
                        #if bic< best_bic+.1:
                            best_bic = bic
                            best_mean_score=mean_score
                            if not improving:
                                num=0
                                lim=0
                        else:
                            # If the BIC did not improve, stop fitting additional Gaussians
                            num+=1
                        # if num>=lim and lim<2:
                        #     lim+=1
                            if best_bic<700:
                                improving = False
                            
                       # print(improving)

                            
                        

                    #print(p0_1)
                # print(p0_1,nwarm)
                    #back=-3*(lim+1)
                    back=-3
                # print(back,p0_1[:back])
                # res1,popt2_1,funTexp1,Fsequences1,wf1,sigma_Tsf1,all_Tsf1,order1,fit_err1,v_shift1=loop(p0_1[:back],ncold, x, 
                #                                                                        y,y_error,popt,nwarm=nwarm-2-lim)
                    res1,popt2_1,funTexp1,Fsequences1,wf1,sigma_Tsf1,all_Tsf1,order1,fit_err1,v_shift1=loop(p0_1[:back],ncold, x, 
                                                                                        y,y_error,popt,nwarm=nwarm-2)
         
                # print('res1',np.array(res1))
                #  print(res1)

                    return res1,popt2_1,funTexp1,Fsequences1,wf1,sigma_Tsf1,all_Tsf1,order1,fit_err1,v_shift1,bic_limit
                res,popt2_,funTexp,Fsequences,wf,sigma_Tsf,all_Tsf,order,fit_err,v_shift,bic_limit=fit_and_calculate_bic(xemi, yemi, yemi_err)

            
        else:
            resye=np.sqrt(np.sum(yemi_selferr**2))
            print(resye)
            
            p0_=p0[2*ncold:]
            t0=p0[:2*ncold]
            num = len(p0_) // 3
            modifications = [
                [p0_[i * 3 + 1] - 4, p0_[i * 3 + 1] + 4] for i in range(num)
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
                    Ts_score=-5
                else:
                    Ts_score=0
                score_+=Ts_score
            score_1.append(score_)
        res=np.array(res)
        score_1,popt2_,funTexp,Fsequences,wf,sigma_Tsf,all_Tsf=(np.array(score_1),np.array(popt2_),
                                                                np.array(funTexp),np.array(Fsequences),
                                                                np.array(wf),np.array(sigma_Tsf),np.array(all_Tsf))
        #print(score_1.shape,popt2_.shape,funTexp.shape,Fsequences.shape,wf.shape,sigma_Tsf.shape,all_Tsf.shape)
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
        print('final BIC ', res[p],'mean score ', np.mean(res) )
        print(score_1[p])
        print('velocity shift',v_shift[p])
    # print('all_Tsf',all_Tsf)
    
        mean_Ts,sigma_meanTsf=[],[]
        for i in range(ncold):
            if len(all_Tsf[:,i])>1:
                m_=np.sum(wf*all_Tsf[:,i])/np.sum(wf)
                mean_Ts.append(m_)
                sigma_meanTsf.append(np.sqrt(np.sum(wf*(all_Tsf[:,i]-m_)**2+sigma_Tsf[:,i]**2)/np.sum(wf)*len(wf)/(len(wf)-1)))
            else:
                mean_Ts.append(all_Tsf[:,i])
                sigma_meanTsf.append(sigma_Tsf[:,i])
            
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
    #def fit_and_plot(x,y,yerr,xemi,yemi,yemi_err,ax,name='J003037-742901',peak_abs=[],peak_emi=[],F=[0,0.5,1],selfabs=False,
    #                    C_Flim=0.97,Tsmin=9,savetxt=True,iteration_max=7,
    #                        fit_mode='F_test',v_sh=0.0001):
        '''
        x:velocity
        y:1-e^(-tau)
        '''
        ax=self.ax
        x,y,yerr,xemi,yemi,yemi_err=self.x,np.copy(self.y),self.y_err,self.xemi,self.yemi,self.yemi_err
        peak_emi=self.peak_emi
        F=self.F
        Tsmin=self.Tsmin
        v_sh=self.v_shift
        name=self.name
        
        #ra, dec = self.parse_coords(name) # attach it later
        popt,pcov,popt2,Ts,Ts_err,gausf,funT,Tfit_err,Or,fit_e,mean_Ts,sigma_meanTsf,nwarm,v_shift,_F=self.Gaussian_fit()
        popt_ori=np.copy(popt)
        ncold=int(len(popt)/3)
        #if y.max()>=1.:
        #    p=np.argwhere(y>=0.99).flatten()
        #    y[p]=y[p]/y.max()*0.99
        #    if self.savecsv:
        #        with open(datapathbase + '/output_data/LMC_fitting/saturated_spectra.txt', 'a') as f:
        #            f.write(f"{name}\n")

        if self.savecsv:
            #properties for full LOS
            NHI_c=0
            NHI_w=0
            sigma_NHIc=0
            sigma_NHIw=0
            K=1.823e18/1e20
            
            if self.renew:
                file_path=self.datapath+'CNMonlydata.csv'
                if os.path.exists(file_path):
                    df_existing = pd.read_csv(file_path)
                    df_existing=df_existing[df_existing['Name'] != name]
                    df_existing.reset_index(drop=True, inplace=True)
                    df_existing.to_csv(file_path, mode='w', index=False, header=True)
                    
                file_path=self.datapath+'Fulldata.csv'
                if os.path.exists(file_path):
                    df_existing = pd.read_csv(file_path)
                    df_existing=df_existing[df_existing['Name'] != name]
                    df_existing.reset_index(drop=True, inplace=True)
                    df_existing.to_csv(file_path, mode='w', index=False, header=True)
                
                file_path=self.datapath+'WNMonlydata.csv'
                if os.path.exists(file_path):
                    df_existing = pd.read_csv(file_path)
                    df_existing=df_existing[df_existing['Name'] != name]
                    df_existing.reset_index(drop=True, inplace=True)
                    df_existing.to_csv(file_path, mode='w', index=False, header=True)
                    
            for i in range(ncold):
                j=i*3
                _popt=popt[j:j+3]
                _pcov=pcov[j:j+3]
                if mean_Ts[i]<=1000:
                    NHI_c+=K*mean_Ts[i]*_popt[0]*np.sqrt(2*np.pi)*_popt[2]
                    _d=((K*sigma_meanTsf[i]*_popt[0]*np.sqrt(2*np.pi)*_popt[2])**2+(K*mean_Ts[i]*_pcov[0]*np.sqrt(2*np.pi)*_popt[2])**2+
                        (K*mean_Ts[i]*_popt[0]*np.sqrt(2*np.pi)*_pcov[2])**2)
                    sigma_NHIc+=_d
                else:
                    NHI_w+=K*mean_Ts[i]*_popt[0]*np.sqrt(2*np.pi)*_popt[2]
                    _d=((K*sigma_meanTsf[i]*_popt[0]*np.sqrt(2*np.pi)*_popt[2])**2+(K*mean_Ts[i]*_pcov[0]*np.sqrt(2*np.pi)*_popt[2])**2+
                        (K*mean_Ts[i]*_popt[0]*np.sqrt(2*np.pi)*_pcov[2])**2)
                    sigma_NHIw+=_d
            sigma_NHIc=np.sqrt(sigma_NHIc)
            print('nhi_c:',NHI_c,'error:',sigma_NHIc)
            for i in range(int(len(gausf)/3)):
                j=i*3
                _popt=gausf[j:j+3]
                _pcov=fit_e[j:j+3]
                #print(gausf.shape,fit_e.shape)
                #NHI_w=abs(1.823e18*trapz(gaussian_func_multi(xemi,*gausf), xemi)/1e20)
                NHI_w+=K*_popt[0]*np.sqrt(2*np.pi)*_popt[2]
                _d=(K*_popt[0]*np.sqrt(2*np.pi)*_pcov[2])**2+(K*_pcov[0]*np.sqrt(2*np.pi)*_popt[2])**2
                #print('errors:', _popt,_pcov,_d)
                sigma_NHIw+=_d
            sigma_NHIw=np.sqrt(sigma_NHIw)
            print('nhi_w:',NHI_w,'error:',sigma_NHIw)
            f_c=NHI_c/(NHI_c+NHI_w)
            sigma_fc=np.sqrt(((NHI_c/(NHI_c+NHI_w)**2*sigma_NHIw))**2+((NHI_w/(NHI_c+NHI_w)**2*sigma_NHIc))**2)
            NHI_uncorr=K*np.trapz(funT,xemi)
            print('fc:',f_c,'error:',sigma_fc)
            data = {
                "Name": name,
                "NHI_c": NHI_c,
                "Sigma_NHIc": sigma_NHIc,
                "NHI_w": NHI_w,
                "Sigma_NHIw": sigma_NHIw,
                "f_c": f_c,
                "Sigma_fc": sigma_fc,
                "Tsky":self.Tsky,
                "NHI_uncorr_fit":NHI_uncorr
            }
            df = pd.DataFrame([data])
            file_path=self.datapath+'Fulldata.csv'
            #file_path=datapathbase + '/output_data/LMC_fitting/Fulldata.csv'
        
            write_header = not os.path.exists(file_path)
            df.to_csv(file_path, mode='a', index=False,header=write_header)
            #a=np.c_[name,NHI_c,sigma_NHIc,NHI_w,sigma_NHIw,f_c,sigma_fc]
            #with open(datapathbase+'/output_data/LMC_fitting/Fulldata.txt', 'a') as file:
            #    np.savetxt(file, a)
            warm_i=[]
            for i in range(ncold):
                j=i*3
                _popt=popt[j:j+3]
                _pcov=pcov[j:j+3]
                #fwhm_=2*np.sqrt(-2*_popt[2]**2*np.log((1-np.sqrt(1-_popt[0]))/_popt[0]))
                fwhm_=2.35482*_popt[2]
                
                NHI_c=K*mean_Ts[i]*_popt[0]*np.sqrt(2*np.pi)*_popt[2]
                NHI_c_err=np.sqrt((K*sigma_meanTsf[i]*_popt[0]*np.sqrt(2*np.pi)*_popt[2])**2+(K*mean_Ts[i]*_pcov[0]*np.sqrt(2*np.pi)*_popt[2])**2+
                    (K*mean_Ts[i]*_popt[0]*np.sqrt(2*np.pi)*_pcov[2])**2)
                #NHI_w=abs(1.823e18*trapz(gaussian_func_multi(xemi,*gausf), xemi)/1e20)
                #NHI_all=abs(1.823e18*trapz(funT, xemi)/1e20)
                T_K=21.866*fwhm_**2
                
                if i==0:
                    #latex_code = r'''%d &%s & %.4f & %.4f& %.2f & %.2f $\pm$ %.2f& %d & %.2f $\pm$ %.2f  & %.1f $\pm$ %.1f  & %.2f $\pm$ %.2f  & %d & %.2f  \\  '''%(number_source,name_abs[-23:-9],ra,dec,f_c,mean_Ts[i],sigma_meanTsf[i],Or[i]
                #      ,_popt[0],_pcov[0],_popt[1],_pcov[1],fwhm_,2.355*_pcov[2],T_K,
                #      NHI_c)
                    latex_code = r'''%s & %.0f$\pm$ %.0f & %.0f $\pm$ %.0f& %d & %.2f $\pm$ %.2f  & %.1f $\pm$ %.1f  & %.2f $\pm$ %.2f & %d & %.2f $\pm$ %.2f \\  '''%(name,f_c*100,sigma_fc*100,mean_Ts[i],sigma_meanTsf[i],Or[i]
                        ,_popt[0],_pcov[0],_popt[1],_pcov[1],fwhm_,2.355*_pcov[2],T_K,
                        NHI_c, NHI_c_err)
                else:
                    #latex_code = r'''
                    #& &  & & & %.2f $\pm$ %.2f & %d & %.2f $\pm$ %.2f  & %.1f $\pm$ %.1f  & %.2f $\pm$ %.2f & %d & %.2f  \\ 

                    #     NHI_c)
                    latex_code = r'''
                      & & & %.0f $\pm$ %.0f & %d & %.2f $\pm$ %.2f  & %.1f $\pm$ %.1f  & %.2f $\pm$ %.2f & %d & %.2f $\pm$ %.2f \\ 

                    '''%(mean_Ts[i],sigma_meanTsf[i],Or[i],_popt[0],_pcov[0],_popt[1],_pcov[1],fwhm_,2.355*_pcov[2],T_K,
                        NHI_c, NHI_c_err)
            # print('popt:',_popt[0])
                
                data = {
                    "Name": name, "tau": _popt[0], "Sigma_tau": _pcov[0], "velocity_tau": _popt[1],"Sigma_velocity_tau": _pcov[1], "fwhm": fwhm_, 
                    "Sigma_fwhm": 2.355*_pcov[2], 
                    "mean_Ts": mean_Ts[i],"sigma_mean_Ts": sigma_meanTsf[i], "T_k_max": T_K, "NHI_c": NHI_c, "Sigma_NHI_c": NHI_c_err, 
                    "v_shift": v_shift[i], "Order":Or[i]
                }
                df = pd.DataFrame([data])
                file_path=self.datapath+'CNMonlydata.csv'
                write_header = not os.path.exists(file_path)
                df.to_csv(file_path, mode='a', index=False,header=write_header)
                #a=np.c_[name,ra,dec,_popt[0],_popt[1],fwhm_,mean_Ts[i],sigma_meanTsf[i],T_K,
                #     NHI_c,NHI_c_err,v_shift[i]]
                #with open(datapathbase+'/output_data/LMC_fitting/CNMonlydata.txt', 'a') as file:
                #    np.savetxt(file, a)

                #with open(datapathbase+'/output_data/LMC_fitting/CNMoutput.txt', 'a') as file:
                with open(self.datapath+'output.txt', 'a') as file:
                    file.write(latex_code)
            #properties for WNM components
            for i in range(int(len(gausf)/3)):
                j=i*3
                _popt=gausf[j:j+3]
                _popt_err=fit_e[j:j+3]
                fwhm_=2.35482*_popt[2]
                NHI_w=K*_popt[0]*np.sqrt(2*np.pi)*_popt[2]
                NHI_w_err=np.sqrt((K*_popt[0]*np.sqrt(2*np.pi)*_popt_err[2])**2+(K*_popt_err[0]*np.sqrt(2*np.pi)*_popt[2])**2)
                T_K=21.866*fwhm_**2
                #latex_code = r'''
                #    & &  &  & & &%.1f & %.2f $\pm$ %.2f & %.1f $\pm$ %.1f & %.2f $\pm$ %.2f & %d & %.2f  \\ 
                #    '''%(F[i],_popt[0],_popt_err[0],_popt[1],_popt_err[1],fwhm_,2.355*_popt_err[2],T_K, NHI_w)
                latex_code = r'''
                     &  & & &%.1f & %.2f $\pm$ %.2f & %.1f $\pm$ %.1f & %.2f $\pm$ %.2f & %d & %.2f $\pm$ %.2f \\ 
                    '''%(_F[i],_popt[0],_popt_err[0],_popt[1],_popt_err[1],fwhm_,2.355*_popt_err[2], T_K,NHI_w,NHI_w_err)
                #with open(datapathbase+'/output_data/LMC_fitting/CNMoutput.txt', 'a') as file:
                with open(self.datapath+'output.txt', 'a') as file:
                    file.write(latex_code)
                #print('nhi1',NHI_w)
                data = {
                    "Name": name,"TB_WNM": _popt[0], "Sigma_TB_WNM": _popt_err[0], "velocity_TB": _popt[1],"Sigma_velocity_TB": _popt_err[1], "fwhm": fwhm_, 
                    "Sigma_fwhm": 2.355*_popt_err[2], "T_k_max": T_K,
                    'NHI_w':NHI_w,'Sigma_NHI_w':NHI_w_err, "F_value":_F[i]
                }
                df = pd.DataFrame([data])
                file_path=self.datapath+'WNMonlydata.csv'
                #file_path=datapathbase + '/output_data/LMC_fitting/WNMonlydata.csv'
                write_header = not os.path.exists(file_path)
                df.to_csv(file_path, mode='a', index=False,header=write_header)
                #a=np.c_[name,NHI_w,NHI_w_err]
                #with open(datapathbase+'/output_data/LMC_fitting/WNMonlydata.txt', 'a') as file:
                #    np.savetxt(file, a)
            
            latex_code=r'''\hline
            '''
            #with open(datapathbase+'/output_data/LMC_fitting/CNMoutput.txt', 'a') as file:
            with open(self.datapath+'output.txt', 'a') as file:
                file.write(latex_code)
        
        print()
        #ax[0].fill_between(xemi,yemi+ yemi_err,yemi- yemi_err,alpha=0.2)
        #print(np.min(yemi),np.min(xemi))
        ax[0].plot(xemi, yemi, label='GASKAP emission',linewidth=1.2,c='tab:blue')
        ax[0].plot(xemi, funT, label='Best fit',linewidth=3,alpha=0.9,c='tab:orange')
        #ax[0].fill_between(xemi,yemi+yemi_err,yemi-yemi_err,alpha=0.2,facecolor='gray',edgecolor='gray')
        for i in range(ncold):
            j=i*3
            _popt=popt[j:j+3]
            _popt[1]=_popt[1]+v_shift[i]
            #ax[0].plot(xemi, (1-np.exp(-self.gaussian_func(xemi,*_popt)))*Ts[i],
            #        label='CNM, Ts_best=%.2f$\pm$%.2f K, \n Ts_mean=%.2f$\pm$%.2f K, \n v_shift=%.1f km/s'%(Ts[i],Ts_err[i],
            #                                                                            mean_Ts[i],sigma_meanTsf[i],v_shift[i]),
            #        linewidth=1.8,linestyle=linestyle_plot[i],color='green')
            ax[0].plot(xemi, (1-np.exp(-self.gaussian_func(xemi,*_popt)))*Ts[i],
                    label='CNM,Ts=%.0f$\pm$%.0f K'%( mean_Ts[i],sigma_meanTsf[i]),
                       linewidth=1.8,linestyle=linestyle_plot[i],color='green')
        
        #print('gausf',gausf)
        for i in range(int(len(gausf)/3)):
            j=i*3
            _popt=gausf[j:j+3]
            ax[0].plot(xemi, self.gaussian_func(xemi,*_popt), label='WNM,F=%.1f \n [%.2f,%.2f,%.2f]'%(_F[i],_popt[0],_popt[1],
                                                                                                2.35482*_popt[2]),
                    linewidth=1.8,linestyle=linestyle_plot[i],color='gray')
        
        ax[0].set_title(name)
        ax[0].set_ylabel(r'$T_B$')
        nco=2 if nwarm>4 else 1
        ax[0].legend(framealpha=0,fontsize='x-small',ncol=nco)
        plt.subplots_adjust(hspace=0)
        ax[1].plot(xemi,yemi-funT,linewidth=1.2,c='black')
        #ax[1].plot(x_,-np.log(1-gaussian_func_multi(x_,*popt)),linewidth=0.7,c='black')
        #ax[1].fill_between(xemi,Tfit_err,-Tfit_err,alpha=0.2,facecolor='gray',edgecolor='gray')
        ax[1].fill_between(xemi,yemi_err,-yemi_err,alpha=0.2,facecolor='gray',edgecolor='gray')
        ax[1].set_ylabel('Residual')
        ax[1].set_xlabel('velocity (km/s)')
        zero=np.array([0]*np.shape(xemi)[0])
        ax[1].plot(xemi,zero,linewidth=1,c='black')
        plt.subplots_adjust(hspace=0)
        #y=1-np.exp(-y)
        #yerr=1-np.exp(-yerr)
        a=1-np.exp(-self.gaussian_func_multi(x,*popt_ori))
        ax[2].plot(x, y, label='GASKAP absorption',linewidth=1.2,c='tab:blue')
        ax[2].plot(x, a, label='Best fit',linewidth=3,alpha=0.9,c='tab:orange')
        
        for i in range(ncold):
            j=i*3
            _popt=popt_ori[j:j+3]
            ax[2].plot(x, 1-np.exp(-self.gaussian_func(x,*_popt)), 
                    label=r'$\tau$ Fitting parameters:'+'\n [%.2f,%.2f,%.2f]'%(_popt[0],_popt[1],2.35482*_popt[2])
                    ,linewidth=1.8,linestyle=linestyle_plot[i],c='green')
            #ax[2].fill_betweenx([0,y.max()],_popt[1]-_popt[2],_popt[1]+_popt[2],alpha=0.2,color='red')
        ax[2].set_ylabel(r'$1-e^{-\tau}$')
        ax[2].legend(framealpha=0,fontsize='small')
        plt.subplots_adjust(hspace=0)
        #err=sigma1_data(y-gaussian_func_multi(x,*popt))
        ax[3].plot(x,y-a,linewidth=1.2,c='black')
    # ax[3].fill_between(x,err,-err,alpha=0.2,facecolor='gray',edgecolor='gray')
        ax[3].fill_between(x,yerr,-yerr,alpha=0.2,facecolor='gray',edgecolor='gray')
        ax[3].set_ylabel('Residual')
        ax[3].set_xlabel('velocity (km/s)')
        zero=np.array([0]*np.shape(xemi)[0])
        ax[3].plot(xemi,zero,linewidth=1,c='black')
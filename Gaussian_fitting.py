import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.stats as stats
from astropy.convolution import Gaussian1DKernel, convolve
from scipy.signal import find_peaks
from itertools import product

#This is a single Gaussian function
def gaussian_func(x, *params):
    a,mu, sigma = params
    return a * np.exp(-(x - mu)**2 / (2 * sigma**2))


#This is multiple Gaussian function
def gaussian_func_multi(x, *gausf):
    fun=np.zeros(len(x))
    for i in range(int(len(gausf)/3)):
        j=i*3
        fun+= gaussian_func(x,*gausf[j:j+3])
    return fun

def calculate_noise(y,yerr,n=1):
    ye=n*yerr-y
    p=np.argwhere(ye>0).flatten()
    ye_n=yerr[p]
    
    return np.nanmean(ye_n)


#This is to calculate CF in section 3.3 of https://arxiv.org/pdf/1503.01108.pdf     
def F_test(x,y_0, y_fit1, y_fit2, sigma_rms, x_1, x_2):
    '''
    x_1,x_2: degree of freedom
    '''
    chi_square1=np.sum((y_0-y_fit1)**2/sigma_rms)/x_1
    chi_square2=np.sum((y_0-y_fit2)**2/sigma_rms)/x_2
    F=chi_square1/chi_square2*x_2/x_1
    CF=stats.f.cdf(F, x_1, x_2)
    return CF

#This is the main fitting function
def fitting(x,y,y_err,CF_limit=0.97,x_peak=[],fit_mode='BIC'):
    if len(x_peak)==0:
        #the original guess of the three parameters of the single Gaussian function
        p=np.argmax(y)
        p0=np.array([1, x[p], 1])
        auto=True
    else:
        p0=[]
        for i in range(len(x_peak)):
            p0=np.append(p0,[1, x_peak[i], 1])
        auto=False
        
    
    #mianly use curve_fit to fit the data
    def loop(p0,x,y,y_err):
        y1=np.copy(y)
        #p=np.argwhere(y1<calculate_noise(y,y_err,n=2)*3).flatten()
        #g = Gaussian1DKernel(stddev=3) 
        #y1[p] = convolve(y1[p], g)
        lowbound=np.array([0. for _ in range(len(p0))])
        num=int(len(p0)/3)
        for _ in range(num):
            lowbound[_*3+1]=x.min()
            ind=np.argmin(np.abs(x - p0[_*3+1]))
            lowbound[_*3]=np.mean(y_err[(ind-4):(ind+4)])*3
            p0[_*3]=np.mean(y_err[(ind-4):(ind+4)])*3+1
            #lowbound[_*3]=calculate_noise(y,y_err,n=2)*3
            #p0[_*3]=calculate_noise(y,y_err,n=2)*3+1
        highbound=np.array([np.inf for _ in range(len(p0))])
       #print(calculate_noise(y,y_err,n=1),p0,lowbound,highbound)
        #f=gaussian_func_multi(x,*p0)
        #print(p0,f)
        pop_, pcov = curve_fit(gaussian_func_multi, x, y1,p0=p0,bounds=(lowbound,highbound),
                               maxfev = 10000)
        pcov_=np.diag(pcov)
        residuals=y-gaussian_func_multi(x,*pop_)
        chi2 = np.nansum((residuals / y_err) ** 2)
        k = len(pop_)
        n = len(y)
        bic = k * np.log(n) + chi2
        a=calculate_noise(y,y_err,n=10)*3.3
        bic  += 20 if np.any(pop_[0::3] < a) else 0

        return pop_,bic,pcov_
    
    if auto:
        if fit_mode=='F_test':
            #if CF> 0.97, adding another Gaussian function is needed and then do the iteration again.
            CF=1
            p0_1=p0
            n_=1
            while CF>CF_limit:
                popt_1,_,pcov_1=loop(p0_1,x, y,y_err)
                function_1=gaussian_func_multi(x,*popt_1)
                p_=np.argmax(y-function_1)
                p0_2=np.append(p0_1,[1, x[p_], 1]) 

                popt_2,_,pcov_2=loop(p0_2,x, y,y_err)
                function_2=gaussian_func_multi(x,*popt_2)
                CF=F_test(x,y, function_1, function_2, y_err, len(x)-len(p0_1), len(x)-len(p0_2))
                n_+=1
                p0_1=p0_2
                print(CF,n_)
            popt=popt_1
            pcov=pcov_1
        elif fit_mode=='BIC':
            def fit_and_calculate_bic(x, y, y_err):
                p0_1=p0
                pe, _ =find_peaks(y, height=np.max(y)/5, distance=5)
                index=np.argsort(y[pe])[::-1]
                pe=pe[index]
                pe=pe[(x[pe] > (x.min()+10)) &(x[pe] < (x.max()-10))]
                _pe = min(5, len(pe))
                popt_1,score_1,pcov_1=loop(p0_1,x, y,y_err)
                _bbic=score_1
                best_bic=_bbic
                b_pos=-1
                print('BIC ',_bbic,'n=1')
                improving = True
                y_res=y
                k=1
                while improving and k<6:
                    _bbic=100000
                    try:
                        for pos in range(_pe):
                            _pp=pe[pos]
                            _p0_1=np.append(p0_1,[1, x[_pp], 1])
                            popt_1,score_1,pcov_1=loop(_p0_1,x, y,y_err)
                            _bic=score_1
                            #print('_bic',_bic,'popt',popt_1)
                           

                            if _bic< _bbic:
                                _bbic = _bic
                                b_pos=pos
                                funT1=gaussian_func_multi(x,*popt_1)
                                g = Gaussian1DKernel(stddev=2)
                                _res=y-funT1
                                y_res= convolve(_res, g)

                    except RuntimeError as e:
                        print(f"Fit failed for Gaussians: {e}")

                    p0_1=np.append(p0_1,[1, x[pe[b_pos]], 1])
                    bic=_bbic
                    pe, _ =find_peaks(y_res, height=np.max(y_res)/5, distance=5)
                    index=np.argsort(y[pe])[::-1]
                    pe=pe[index]
                    pe=pe[(x[pe] > (x.min()+10)) &(x[pe] < (x.max()-10))]
                    _pe = min(6-k, len(pe))
                    b_pos=-1
                    k+=1
                    print('BIC ',bic,'n=',k)
                    if bic< best_bic-5:
                        best_bic = bic

                    else:
                        # If the BIC did not improve, stop fitting additional Gaussians
                        improving = False
                #print(best_bic,p0_1)    
                popt_1,score_1,pcov_1=loop(p0_1[:-3],x, y,y_err)
                print('final n=',k-1)
                #print(popt_1)    
               
                return popt_1,pcov_1
            popt,pcov= fit_and_calculate_bic(x, y, y_err)
            

          
    else:
        num = len(p0) // 3
        modifications = [
            [p0[i * 3 + 1] - 4, p0[i * 3 + 1] + 4] for i in range(num)
        ]
        # Generate all combinations of the modified second elements
        all_combinations = list(product(*modifications))
        # Evaluate all combinations to find the minimum BIC
        results = []
        for combination in all_combinations:
            new_p0 = p0.copy()
            for i in range(num):
                new_p0[i * 3 + 1] = combination[i]  # Update second elements
            results.append(loop(new_p0, x, y, y_err))

        popt, bic, pcov = min(results, key=lambda r: r[1])
        print('BIC=',bic)

    return popt,pcov


def fitting_plot(x,y,y_err,x_peak=[],fit_mode='BIC'):
    #if you want to smooth the data:
    #g = Gaussian1DKernel(stddev=0.5) # you can adjust the stddev value to prevent over smooth (smaller stddev, less smooth)
    #y = convolve(y, g)
    popt,_=fitting(x,y,y_err,x_peak=x_peak,fit_mode=fit_mode)
    print(popt)
    y_fit=gaussian_func_multi(x, *popt)
    plt.figure(dpi=200)
    plt.subplot(211)
    plt.plot(x,y)  # original data
    plt.plot(x,y_fit)     #total fitting data
    #print(int(len(popt)/3))
    for i in range(int(len(popt)/3)):   #plot individual Gaussian components
        j=i*3
        _popt=popt[j:j+3]
        plt.plot(x,gaussian_func_multi(x, *_popt),c='gray',linewidth=1)
    plt.subplot(212)
    plt.plot(x,y-y_fit,c='black',linewidth=1)
    plt.axhline(y=0, color='black', linestyle='--')
    plt.fill_between(x,y_err,-y_err,alpha=0.2,facecolor='gray',edgecolor='gray')
    plt.subplots_adjust(hspace=0)



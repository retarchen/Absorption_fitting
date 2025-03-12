from radiative_transfer import Gf,sd
import read_GASKAP_data as rd
import numpy as np
import matplotlib.pyplot as plt

#.....
#spectra=rd.get_emi_abs_data(nlist[j],nabs[j],1,1,mode='LMC',v_move=0.)
x,y,yerr,xemi,yemi,yemi_err=spectra['velocity_tau'],spectra['1_e_tau'],spectra['1_e_tau_err'],spectra['velocity_TB'],spectra['TB'],spectra['TB_err']

fig = plt.figure(figsize=(12, 8), dpi=300)

ax1_first = plt.subplot2grid((8, 2), (0, 0), rowspan=3, fig=fig)

ax = [ax1_first,
       plt.subplot2grid((8, 2), (3, 0), rowspan=1, fig=fig, sharex=ax1_first),
       plt.subplot2grid((8, 2), (4, 0), rowspan=3, fig=fig, sharex=ax1_first),
       plt.subplot2grid((8, 2), (7, 0), fig=fig, sharex=ax1_first)]

'''
sd.fit_and_plot(x,y,yerr,xemi,yemi,yemi_err,ax,name=nlist[j],savetxt=False,
                peak_abs=[],peak_emi=[],Tsmin=3.77,
                 fit_mode='BIC',v_sh=4)

'''
tsyn,_=rd.read_synchro_emi(nlist[j])
spec_fit=sd(x,y,yerr,xemi,yemi,yemi_err)
spec_fit.v_shift=4.
spec_fit.peak_abs=[]
spec_fit.peak_emi=[]
spec_fit.Tsmin=2.73+tsyn
spec_fit.Tsky=2.73+tsyn
spec_fit.savecsv=True
spec_fit.fit_mode='BIC'
spec_fit.ax=ax
#spec_fit.datapath=datapath #default is the current
spec_fit.fit_and_plot()
#!/usr/bin/env python
# encoding: utf-8
"""
untitled.py

Created by Marc-André on 2012-10-20.
Copyright (c) 2012 IGBMC. All rights reserved.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
from frQRd import rQRd
from sane import sane
import Cadzow

def mfft(v):
    "utility that returns the modulus of the fft of v"
    import scipy.fftpack as fft
    n = len(v)
    v2 = np.zeros(4*n,dtype=complex)   # do 4 time zero-filling for nicer spectrum.
    v2[:n] = v[:]
    s0 = fft.fft(v2)
    s0 = np.real(np.sqrt(s0*s0.conj()))   # ref spectrum
    return s0

def SNR(noisy,target):
    "computes and return SNR value, in dB"
    return 10*np.log10(sum(abs(target)**2)/sum(abs(noisy - target)**2))

def gene(lendata = 10000, noise = 20.0, noisetype = "additive"):
        """
        generate synthetic interferograms
        """
        t = np.arange(lendata*1.0)/lendata          #  time series

        Freq = np.array([ 0.10523667,  0.28469171,  0.17548498,  0.72540314,  0.05720856,
                0.78487534,  0.68200948,  0.50657686,  0.95773274,  0.442715  ,
                0.05376049,  0.55894902,  0.80293117,  0.2885081 ,  0.45760559,
                0.57798204,  0.51470708,  0.91344429,  0.81348799,  0.93070597])*2j*np.pi*lendata
        Freq = np.array([ 0.06523667,  0.28469171,  0.10548498,  0.72540314,  0.05720856,
                0.78487534,  0.68200948,  0.50657686,  0.05603274,  0.442715  ,
                0.05376049,  0.55894902,  0.80293117,  0.2885081 ,  0.45760559,
                0.57798204,  0.51470708,  0.91344429,  0.81348799,  0.93070597])*2j*np.pi*lendata
        nbpeaks = len(Freq)
        Amp = [(i+1)*20 for i in range(nbpeaks)]    # amplitudes
        LB = 1.0      # linewidth

        data0 = np.zeros(lendata, dtype=complex)

        if noisetype == "additive":
            for i in range(nbpeaks):
                data0 +=  Amp[i] * np.exp(Freq[i]*t) * np.exp(-LB*t)

            dataadd = data0 + noise*(np.random.randn(t.size)+1j*np.random.randn(t.size))  # additive complex noise
            data=dataadd

        elif noisetype == "scintillation":
            for i in range(nbpeaks):
                data0 +=  Amp[i] * np.exp(Freq[i]*t) * np.exp(-LB*t)

            data = np.zeros(lendata, dtype=complex)
            Anoise = noise/2
            Fnoise = noise/200
            for i in range(nbpeaks):
                nAmp = Amp[i] + Anoise*np.random.randn(t.size)
                nFreq = Freq[i] + Fnoise*np.random.randn(t.size)
                data +=  nAmp * np.exp(nFreq*t) * np.exp(-LB*t)
            
        elif noisetype == "sampling":
            tn = t + 0.3*np.random.randn(t.size)/lendata          #  time series with noisy jitter
            for i in range(nbpeaks):
                data0 +=  Amp[i] * np.exp(Freq[i]*t) * np.exp(-LB*t)
            data = np.zeros(lendata, dtype=complex)
            for i in range(nbpeaks):
                data +=  Amp[i] * np.exp(Freq[i]*tn) * np.exp(-LB*tn)
            
        elif noisetype == "missing points":
            for i in range(nbpeaks):
                data0 +=  Amp[i] * np.exp(Freq[i]*t) * np.exp(-LB*t)
            miss = np.random.randint(2, size=len(t))
            
            dataadd = data0*miss
            data=dataadd
        else:
            raise Exception("unknown noise type")

        return (data,data0)
    
def main(meth='rQRd', zoom=False):
    t0 = time.time()
    lendata = 100000
    noisetype = "additive"
    noise = 300.0
    if meth == 'rQRd':
        method = rQRd
        orda = 15000
        rank = 40
    elif meth == 'SANE':
        method = sane
        orda = 15000
        rank = 24
    plt.figure()
    for i,noisetype in enumerate(("additive","scintillation",)): # "sampling", "missing points")):
        data,data0 = gene(lendata, noise, noisetype)        
        iSNR = SNR(data,data0)
        print ("\nInitial Noisy Data SNR: %.2f dB - noise type : %s"%(iSNR,noisetype))

        print ("=== Running %s algo ==="%(meth,),)
        print ("lendata:",lendata,)
        print (" orda:",orda,)
        print (' rank:',rank)
        datarqrd = method(data, k=rank, orda=orda) # denoise signal with rQRd
        fSNR = SNR(datarqrd,data0)
        print ("Final Data SNR: %.2f dB - noise type : %s"%(fSNR,noisetype))
        if i < 2 :
            pos = i+3
        else:
            pos = i+5
        sub = plt.subplot(5,2,pos)
        plt.plot( mfft(data) )
        plt.ylabel("noisy data")
        sub.xaxis.set_major_locator(plt.NullLocator())
        sub.yaxis.set_major_locator(plt.NullLocator())
        if zoom:
            sub.set_xbound(*zoom)    # pour zoomer au début
        
        sub = plt.subplot(5,2,pos+2)
        plt.plot( mfft(datarqrd) )
        plt.ylabel("%s filtered"%(meth,))
        plt.title("noise-type : %s"%noisetype)
        sub.xaxis.set_major_locator(plt.NullLocator())
        sub.yaxis.set_major_locator(plt.NullLocator())

        if zoom:
            sub.set_xbound(*zoom)    # pour zoomer au début


    sub = plt.subplot(5,2,1)
    plt.plot( data0 )
    plt.ylabel("noise-free data-set")
    sub.xaxis.set_major_locator(plt.NullLocator())
    sub.yaxis.set_major_locator(plt.NullLocator())
    sub = plt.subplot(5,2,2)
    plt.plot( mfft(data0) )
    if zoom:
        sub.set_xbound(*zoom)    # pour zoomer au début

    sub.xaxis.set_major_locator(plt.NullLocator())
    sub.yaxis.set_major_locator(plt.NullLocator())
    plt.ylabel("FT")
    print('elaps:', time.time()-t0)
    plt.show()


if __name__ == '__main__':
    main(meth='SANE')


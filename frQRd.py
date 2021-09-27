"""
rQRd.py

Created by Lionel Chiron and Marc-Andr\'e on 2013-10-13.
Copyright (c) 2012 IGBMC. All rights reserved.

version 1.1 
12/oct/2012 corrected for horizontal H matrix, small Y,  + optimized code

#########
Algorithm rQRd method for  denoising time series, named rQRd (standing for random QR denoising)

main function is 
rQRd(data, rank)
data : the series to be denoised
rank : the rank of the analysis

"""
import sys
import numpy as np
import numpy.linalg as linalg
from scipy.linalg import norm
from numpy.fft import fft, ifft
import functools
import multiprocessing as mproc
import unittest
import itertools
import time

debug = 0 # put to 1 for debuging message
mp = True
def rQRd(data, k, orda=None, iterations=1):
    
    """ 
    rQRd algorithm. Name stands for random QR denoising.
    From a data series return a denoised series denoised
    data : the series to be denoised - a (normally complex) numpy buffer
    k : the rank of the analysis
    orda : is the order of the analysis
        internally, a Hankel matrix (M,N) is constructed, with M = orda and N = len(data)-orda+1
        if None (default) orda = (len(data)+1)/2

    values are such that
    orda <= (len(data)+1)/2
    k < orda
    N = len(data)-orda+1
    Omega is (N x k)
    """
    if np.allclose(data,0.0):   # dont do anything if data is empty
        return data
    if not orda:
        orda = (data.size+1)/2
    if (2*orda > data.size):
        raise(Exception('order is too large'))
    if (k >= orda):
        raise(Exception('rank is too large'))
    N = len(data)-orda+1
    if debug:
        t0 = time.time()
    dd = data
    for i in range(iterations):
        Omega = np.random.normal(size=(N,k))
        Q, QstarH = rQRdCore(dd, orda, Omega)# H=QQ*H
        if debug:
            print( "Q.shape, QstarH.shape ",Q.shape, QstarH.shape)
        if mp:
            dd = Fast_Hankel2dtmp(Q,QstarH)
        else:
            dd = Fast_Hankel2dt(Q,QstarH)
    denoised = dd
    if data.dtype == "float":           # this is a kludge, as a complex data-set is to be passed - use the analytic signal if your data are real
        denoised = np.real(denoised)
    if debug:
        print ("Total time %.2f seconds"%(time.time()-t0))
    return denoised

def rQRdCore(data, orda, Omega):
    '''
    Core of rQRd algorithm
    '''
    if debug:
        t0 = time.time()
        print ("==== computing Y")
    if mp:
        Y =  FastHankel_prod_mat_matmp(data, orda, Omega)
    else:
        Y =  FastHankel_prod_mat_mat(data, orda, Omega)
    if debug:
        t1 = time.time()
        print ("it took %.2f seconds"%(t1-t0))
        print ("==== computing QR")
    Q,r = linalg.qr(Y) # QR decomopsition of Y
    del(r)          # we don't need it any more
    if debug:
        t2 = time.time()
        print ("it took %.2f seconds"%(t2-t1))
        print ("=== computing Q and QstarH")
        print ("orda ", orda)
        print ("Omega.shape[1] ",Omega.shape[1])
    if mp:
        QstarH = FastHankel_prod_mat_matmp(data.conj(), Omega.shape[0], Q).conj().T# 
    else:
        QstarH = FastHankel_prod_mat_mat(data.conj(), Omega.shape[0], Q).conj().T# 
    if debug:
        t3 = time.time()
        print ("it took %.2f seconds"%(t3-t2))
    return Q, QstarH# Approximation of H given by QQ*H    

def vec_mean(M,L):
    '''
    Vector for calculating the mean from the sum
    data = vec_sum*vec_mean
    '''
    vec_prod_diag = [1/float((i+1)) for i in range(M)]
    vec_prod_middle = [1/float(M) for i in range(L-2*M)]
    vec_mean_prod_tot = vec_prod_diag + vec_prod_middle + vec_prod_diag[::-1]
    return np.array(vec_mean_prod_tot)

def FastHankel_prod_mat_mat(gene_vect, M, matrix):
    '''
    Fast Hankel matrix matrix product based on FastHankel_prod_mat_vec
    '''
    if debug:print ("in fast hankel matrix product")
    N,K = matrix.shape # equivalent of U
    data = np.zeros(shape = (M, K), dtype = complex)
    L = M+N-1
    for k in range(K):
        prod_vect = matrix[:,k] # vector line, length = N
        data[:,k] = FastHankel_prod_mat_vec(gene_vect, M, prod_vect) # 
    return data

def fFastHankel_prod_mat_matmp(k):
    global gene_vect, M, matrix
    prod_vect = matrix[:,k] # vector line, length = N
    datak = FastHankel_prod_mat_vec(gene_vect, M, prod_vect) # 
    return datak

def FastHankel_prod_mat_matmp(gene_vect, M, matrix, nbproc=None):
    '''
    Fast Hankel matrix matrix product based on FastHankel_prod_mat_vec
    '''
    if debug:print ("in fast hankel matrix product")
    N,K = matrix.shape # equivalent of U
    data = np.zeros(shape = (M, K), dtype = complex)
    L = M+N-1
    inject(('gene_vect', gene_vect))
    inject(('matrix', matrix))
    inject(('M', M))
    todo = range(K)
    pool = mproc.Pool(nbproc)
    res = pool.imap(fFastHankel_prod_mat_matmp,  todo)
    for k in range(K):
        data[:,k] = res.next()
    pool.close()
    pool.join()
    del(matrix)
    del(gene_vect)
    del(M)
    return data

def FastHankel_prod_mat_vec(gene_vect, M, prod_vect):
    """
    Compute product of Hankel matrix H = dt2Hankel(gene_vect, M) by prod_vect.
    H is not computed
    M is the length of the result
    """
    L = len(gene_vect)
    prod_vect_zero = np.concatenate((prod_vect, np.zeros(M-1)))   # prod_vect is completed with zero to length L
    fft0, fft1 = fft(gene_vect), fft(prod_vect_zero[::-1])      # FFT transforms
    prod = fft0*fft1                          # FFT product for doing convolution product. 
    c = ifft(prod)                        # IFFT for going back 
    return np.roll(c,+1)[:M]

def inject(arg):
    """
    this functions injects in the global dictionnary a new object
    arg is (name, value)
    
    used to set-up global before Pool
    generates many errors for pylint which does not understand the techniques
    """
    (nom, val) = arg
    globals()[nom]=val
    return None


def Fast_Hankel2dt(Q,QH):
    '''
    returning to data from Q and QstarH
    Based on FastHankel_prod_mat_vec.
    '''
    if debug:print ("in Fast Hankel to data ")
    M,K = Q.shape # equivalent of T
    K,N = QH.shape # equivalent of U
    L = M+N-1
    vec_sum = np.zeros((L,), dtype = complex)
    for k in range(K):
        prod_vect = QH[k,:] # length = N
        gene_vect = np.concatenate((np.zeros(N-1), Q[:, k], np.zeros(N-1)))# gene vec for Toeplitz matrix
        vec_k = FastHankel_prod_mat_vec(gene_vect, L, prod_vect[::-1]) # used as fast Toeplitz
        vec_sum += vec_k 
    datadenoised = vec_sum*vec_mean(M,L)# from the sum to the mean
    return datadenoised

def fFast_Hankel2dtmp(k):
    global Q, QstarH
    prod_vect = QstarH[k,:] # length = N
    M,K = Q.shape # equivalent of T
    K,N = QstarH.shape # equivalent of U
    L = M+N-1
    gene_vect = np.concatenate((np.zeros(N-1), Q[:, k], np.zeros(N-1)))# gene vec for Toeplitz matrix
    vec_k = FastHankel_prod_mat_vec(gene_vect, L, prod_vect[::-1]) # used as fast Toeplitz
    return vec_k
def Fast_Hankel2dtmp(Q, QstarH, nbproc=None):
    '''
    returning to data from Q and QstarH
    Based on FastHankel_prod_mat_vec.
    '''
    inject(('Q',Q))
    inject(('QstarH',QstarH))
    if debug:print ("in Fast Hankel to data ")
    M,K = Q.shape # equivalent of T
    K,N = QstarH.shape # equivalent of U
    L = M+N-1
    vec_sum = np.zeros((L,), dtype = complex)
    pool = mproc.Pool(nbproc)
    todo = range(K)
    res = pool.imap(fFast_Hankel2dtmp, todo)
    for i in range(K):
        vec_sum += res.next()
    pool.close()
    pool.join()
    # print ("vec_mean")
    datadenoised = vec_sum*vec_mean(M,L)# from the sum to the mean
    return datadenoised

class rQRd_Tests(unittest.TestCase):
    def test_rQRd(  self,
                    lendata = 10000,
                    rank=100,
                    orda=4000,
                    nbpeaks = 20,
                    noise = 200.0,
                    noisetype = "additive"):
        """
        ============== example of use of rQRd on a synthetic data-set ===============
        """
        import time
        import matplotlib.pyplot as plt

        ###########
        print ("=== Running rQR algo ===",)
        print ("lendata:",lendata,)
        print (" orda:",orda,)
        print (' rank:',rank)
           
        def mfft(v):
            "utility that returns the modulus of the fft of v"
            import scipy.fftpack as fft
            s0 = fft.fft(v)
            s0 = np.real(np.sqrt(s0*s0.conj()))   # ref spectrum
            return s0
        def SNR_dB(noisy,target):
            "computes and return SNR value, in dB"
            return 10*np.log10(sum(abs(target)**2)/sum(abs(noisy - target)**2))
        ########--------
        # Data built for tests
        ################################################ Create the data
        nbpeaks = 8     # number of simulated signals
        LB = 1.11       # linewidth
        Freq = [(i+1+np.sqrt(10))*np.pi*500.0j for i in range(nbpeaks)]  # frequencies
        Amp = [(i+1)*20 for i in range(nbpeaks)]    # amplitudes

        data0 = np.zeros(lendata,dtype=complex)

        if noisetype == "additive":
            x = np.arange(lendata*1.0)/lendata          #  time series
            for i in range(nbpeaks):
                data0 +=  Amp[i] * np.exp(Freq[i]*x) * np.exp(-LB*x)

            dataadd = data0 + noise*(np.random.randn(x.size)+1j*np.random.randn(x.size))  # additive complex noise
            data=dataadd

        elif noisetype == "multiplicative":
            x = np.arange(lendata*1.0)/lendata          #  time series
            for i in range(nbpeaks):
                data0 +=  Amp[i] * np.exp(Freq[i]*x) * np.exp(-LB*x)

            data = np.zeros(lendata,dtype=complex)
            Anoise = noise/2
            Fnoise = noise/200
            for i in range(nbpeaks):
                nAmp = Amp[i] + Anoise*np.random.randn(x.size)
                nFreq = Freq[i] + Fnoise*np.random.randn(x.size)
                data +=  nAmp * np.exp(nFreq*x) * np.exp(-LB*x)

        elif noisetype == "sampling":
            x = np.arange(lendata*1.0)/lendata          #  time series
            xn = x + 0.5*np.random.randn(x.size)/lendata          #  time series with noisy jitter
            for i in range(nbpeaks):
                data0 +=  Amp[i] * np.exp(Freq[i]*x) * np.exp(-LB*x)
            data = np.zeros(lendata,dtype=complex)
            for i in range(nbpeaks):
                data +=  Amp[i] * np.exp(Freq[i]*xn) * np.exp(-LB*xn)

        elif noisetype == "missing points":
            x = np.arange(lendata*1.0)/lendata          #  time series
            for i in range(nbpeaks):
                data0 +=  Amp[i] * np.exp(Freq[i]*x) * np.exp(-LB*x)
            miss = np.random.randint(2, size=len(x))

            dataadd = data0*miss
            data=dataadd
        else:
            raise Exception("unknown noise type")
        
        iSNR = SNR_dB(data,data0)
        print ("Initial Noisy Data SNR: %.2f dB - noise type : %s"%(iSNR,noisetype))
        t0 = time.time()
        datarqrd = rQRd(data, k=rank, orda=orda) # denoise signal with rQRd
        trQRd = time.time()-t0

        fdatarqrd  = mfft(datarqrd )# FFT of rQRd denoised signal
        print ("=== Result ===")
        fSNR = SNR_dB(datarqrd, data0)
        print( "Denoised SNR: %.2f dB  - processing gain : %.2f dB"%( fSNR, fSNR-iSNR ))
        print ("processing time for rQRd : %f sec"%trQRd)
        print( fSNR-iSNR)
        #####
        sub = subpl(2, plt)
        sub.next()
        #######################
        sub.plot(data0,'b',label="clean signal")# rQR normal rQR
        sub.title('data series')
        sub.next()
        sub.plot(fdata,'b',label="clean spectrum")# rQR normal rQR
        sub.title('FFT spectrum')
        ######################
        sub.next()
        sub.plot(data,'k', label="noisy signal")# plot the noisy data
        sub.next()
        sub.plot(fdatanoise,'k', label="noisy spectrum")# plot the noisy data
        #######################
        sub.next()
        sub.plot(datarqrd ,'r', label='rQRd filtered signal') # plot the signal denoised with rQRd
        sub.next()
        sub.plot(fdatarqrd ,'r', label= 'rQRd filtered spectrum') # plot the signal denoised with rQRd
        sub.title("Noise type : "+noisetype)
        ############################
        sub.show()
    
if __name__ == '__main__':
    unittest.main()

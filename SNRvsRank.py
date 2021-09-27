'''
Comparison of rQRd^n with SVD for the rank spanning the interval [0,order].
#########

Copyright (c) 2013 IGBMC and CNRS. All rights reserved.

Marc-Andr\'e Delsuc <madelsuc@unistra.fr>
Lionel Chiron <lionel.chiron@gmail.com>

This software is governed by the CeCILL  license under French law and
abiding by the rules of distribution of free software.  You can  use, 
modify and/ or redistribute the software under the terms of the CeCILL
license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info". 

As a counterpart to the access to the source code and  rights to copy,
modify and redistribute granted by the license, users are provided only
with a limited warranty  and the software's author,  the holder of the
economic rights,  and the successive licensors  have only  limited
liability. 

In this respect, the user's attention is drawn to the risks associated
with loading,  using,  modifying and/or developing or reproducing the
software by the user in light of its specific status of free software,
that may mean  that it is complicated to manipulate,  and  that  also
therefore means  that it is reserved for developers  and  experienced
professionals having in-depth computer knowledge. Users are therefore
encouraged to load and test the software's suitability as regards their
requirements in conditions enabling the security of their systems and/or 
data to be ensured and,  more generally, to use and operate it in the 
same conditions as regards security. 

The fact that you are presently reading this means that you have had
knowledge of the CeCILL license and that you accept its terms.

'''

import os, sys
import time
import numpy as np
import scipy.linalg as lin
import numpy as np, numpy.linalg as linalg
from scipy.interpolate import interp1d
from scipy.linalg import norm
from math import *
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from frQRd import rQRd
from Cadzow import dt2svd, svdclean, svd2dt     # load internal pieces of Cadzow to speed-up computation
from sane import sane


def SNR(noisy,target):
       "computes and return SNR value, in dB"
       return 10*np.log10(sum(abs(target)**2)/sum(abs(noisy - target)**2))
   ########--------
   
def makesignoise(shift,nbpeaks,sigmas,lengthfid, noise):
    '''
    building the signal with noise
    '''
    LB = 1  # linewidth
    x = np.arange(lengthfid*1.0)/lengthfid        
    fid0 = 1j*np.zeros_like(x)      # complex fid
    omeg = 43.1*1j
    for i in range(1, nbpeaks+1):
        fid0 +=  i*sigmas*np.exp(omeg*(i+shift)*x)*np.exp(-LB*x)   # 
    fid = fid0 + noise*(np.random.randn(x.size)+1j*np.random.randn(x.size))# additive noise 
    return fid,fid0

def testrQR(fid, rank, orda, U, S, Vh):
     "compute rQRd^n and Cadzow at rank=rank   SVD pre-computed in U S Vh"
     ########
     iSNR = SNR(fid,fid0)
     '''
     Cadzow with classical svd
     '''     
     #fidcadz = cadzow(fid, n_of_line=rank, n_of_iter=1, orda=orda)# Cadzow with classical SVD.
     S1 = svdclean(S, keep=rank, remove=1)
     fidcadz = svd2dt(U, S1, Vh)
     snrlogrqrdcadz = SNR(fidcadz,fid0)-iSNR # norm(fidcadz-fid0)/norm(fid0)
     ###########----
     '''
     rQR
     '''
     fidrQRd = rQRd(fid, k = rank, orda= orda)
     snrlogrqrd = SNR(fidrQRd,fid0)-iSNR

     fidrQRd2 = rQRd(fid, k = rank, orda= orda, iterations=2)
     snrlogrqrd2 = SNR(fidrQRd2,fid0)-iSNR

     fidrQRd3 = rQRd(fid, k = rank, orda= orda, iterations=3)
     snrlogrqrd3 = SNR(fidrQRd3,fid0)-iSNR 

     fidrQRd4 = rQRd(fid, k = rank, orda= orda, iterations=4)
     snrlogrqrd4 = SNR(fidrQRd4,fid0)-iSNR

     return (snrlogrqrd, snrlogrqrd2, snrlogrqrd3, snrlogrqrd4), snrlogrqrdcadz

def testSANE(fid, rank, orda, U, S, Vh):
     "compute SANE^n and Cadzow at rank=rank   SVD pre-computed in U S Vh"
     ########
     iSNR = SNR(fid,fid0)
     '''
     Cadzow with classical svd
     '''     
     #fidcadz = cadzow(fid, n_of_line=rank, n_of_iter=1, orda=orda)# Cadzow with classical SVD.
     S1 = svdclean(S, keep=rank, remove=1)
     fidcadz = svd2dt(U, S1, Vh)
     snrlogrqrdcadz = SNR(fidcadz,fid0)-iSNR # norm(fidcadz-fid0)/norm(fid0)
     ###########----
     '''
     SANE
     '''
     fidrQRd = sane(fid, k = rank, orda= orda)
     snrlogrqrd = SNR(fidrQRd,fid0)-iSNR

     fidrQRd2 = sane(fid, k = rank, orda= orda, iterations=2)
     snrlogrqrd2 = SNR(fidrQRd2,fid0)-iSNR

     fidrQRd3 = sane(fid, k = rank, orda= orda, iterations=3)
     snrlogrqrd3 = SNR(fidrQRd3,fid0)-iSNR 

     fidrQRd4 = sane(fid, k = rank, orda= orda, iterations=4)
     snrlogrqrd4 = SNR(fidrQRd4,fid0)-iSNR

     return (snrlogrqrd, snrlogrqrd2, snrlogrqrd3, snrlogrqrd4), snrlogrqrdcadz

def setuplot(xmax):
    '''
    Setup for picture
    '''
    sizeax = 15
    ax[0].legend()
#    ax[0].xaxis.set_major_locator(MaxNLocator(3))
    ax[0].set_xlabel('rank', size = sizeax)
    ax[0].set_ylabel('$SNR\ Gain\ (dB)$', size = sizeax)
    pts = [0,5,10]
    ax[0].set_ylim(-3,12)
    ax[0].set_xlim(0,500)
#    ax[0].set_ylim(pts[0],pts[2]+5)
    #ax[0].set_ylim(2,pts[2])
    #ax[0].set_xlim(10, orda)
    ax[0].set_yticks([0,5,10])
    ax[0].plot([0,xmax],[0,0],color='k')
#    ax[0].set_yticklabels([-5,0,5,10,15])
    #ax[0].set_xticks([0,10,100])
    #ax[0].set_xticklabels(['0','10','100'])
    plt.setp(ax[0].get_yticklabels(), visible = True)
 
def figure(
    run = True, # activate each phase independently
    draw = True,
    svd_store = "svd.npy", # file to store results
    rqrd_store = "rqrd.npy",
    steprank = 20,    # resolution of the plot
    nbpeaks = 50,
    sigmas = 30.0,
    lengthfid = 1000,
    noise = 200.0 ,
    orda = 499 ):
    ###
    global fid0
    global ax
    shift = 333
    if run:
        np.random.seed(12345)
        #####
        rqrd_res = []
        svd_res = []
        listcadz = []
        listrqrd = [None]*4
        for i in range(4):
            listrqrd[i] = []
        fid, fid0 = makesignoise(shift, nbpeaks, sigmas, lengthfid, noise)
        (U, S, Vh) = dt2svd(fid, orda=orda)    # """first steps of Cadzow""" - SVD done once for all
        for rank in range(steprank,orda-35,steprank):
            if rank%20 == 0 : print( "rank ",rank)
            snrlogrqrds, snrlogrqrdcadz = testSANE(fid, rank, orda, U, S, Vh)
            listcadz.append(snrlogrqrdcadz)# list for Cadzow
            for i in range(4):
                listrqrd[i].append(snrlogrqrds[i]) # list for rQRd
        svd_res.append(listcadz)
        rqrd_res.append(listrqrd)
        np.save(svd_store, np.array(svd_res))
        np.save(rqrd_store, np.array(rqrd_res))
    if draw:
        fig = plt.figure(figsize=(20/2.54,12/2.54))
        ax=[]
        ax.append(fig.add_subplot(1,1,1))
        svd_res = np.load(svd_store)
        rqrd_res = np.load(rqrd_store)
        listcadz = svd_res[0]
        listrqrd = rqrd_res[0]
        ###
        r = np.arange(steprank,orda-35,steprank)
        plt.plot(r,listcadz,'-k', label='SVD',linewidth=2)
        ##
        colors=['--k',':k','-.k','-k']
        for i in range(4):
            plt.plot(r,listrqrd[i], colors[i], label='SANE$^%d$'%(i+1),linewidth=1)

        setuplot(rqrd_res.shape[1])
        plt.savefig("SNRvsRank.pdf")
if __name__ == '__main__':
    t0 = time.time()
    figure(steprank=4)
    print (time.time()-t0,"sec")
    plt.show()
    
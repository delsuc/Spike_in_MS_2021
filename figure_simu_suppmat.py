#!/usr/bin/env python
# encoding: utf-8
"""

Figure 1 c-d-e-f)
& et b are elsewhere

Created by Marc-André on 2012-09-24.
Copyright (c) 2012 IGBMC. All rights reserved.
"""

import numpy as np
import matplotlib.pyplot as plt
import os.path as op

symr = 'ko-'
symu = 'kD-'
syms = 'k+--'
def load(fich):
    res = []
    F = open(fich)
    for l in F:
        if not l.startswith('#'):
            res.append(l.split())
    res = np.array(res,dtype=float)
    return res

def figure1(fich):
    "temps"
    res = load(fich)
    N = res[:,0]
    tsvd = res[:,1]
    trQR = res[:,2]
    trandalgo = res[:,3]
    f1(N,tsvd,trQR,trandalgo)
def f1(N,tsvd,trQR,trandalgo):
    "do the plot"
    print "SVD:", np.polyfit(np.log(N[:11]), np.log(tsvd[:11]), 1)[0]
    print "rQRd", np.polyfit(np.log(N[:11]), np.log(trQR[:11]), 1)[0]
    print "urQrd", np.polyfit(np.log(N[15:]), np.log(trandalgo[15:]), 1)[0]
    plt.loglog(N,trQR,symr,label='rQRd')
    plt.loglog(N,trandalgo,symu,label='urQRd')
    plt.loglog(N,tsvd,syms,label='SVD')
    temps(1E7)
    plt.ylabel('Processing time (sec)')
    plt.legend(loc=2)
    plt.title("a) compared speed of different methods")

def figure2(fich):
    "SNR"
    res = load(fich)
    N = res[:,0]
    SSVD = res[:,7]
    SrQRd = res[:,8]
    SurQRd2 = res[:,9]
    f2(N,SSVD,SrQRd,SurQRd2)

def f2(N,SSVD,SrQRd,SurQRd):
    "do the plot"
    plt.semilogx(N,SrQRd,symr,label='rQRd')
    plt.semilogx(N,SurQRd,symu,label='urQRd')
    plt.semilogx(N,SSVD,syms,label='SVD')
    plt.xlabel('Data length')
    plt.ylabel('SNR gain (dB)')
    plt.axis(ymin=0, ymax=40)
    snr(bord=1E7)
#    plt.legend(loc=2)
    plt.title("b) compared SNR of different methods")


def figure3bis(loc):
    "comparaison des SNR en f() de trunc"
    symbs = ['kd-','kx-','kD-','k+-','ko-']
    symbs.reverse()
    ta = 200
    for (tb,tc) in ((250,250),(1000,250),(1000,1000),(4000,1000),(4000,4000)):
    # for tb in (1000,4000,8000):
    #     for tc in (1000,4000,8000):
                f = op.join(loc,"partiel-%d-%d-%d.txt"%(ta,tb,tc))
                try:
                    res = load(f)
                except:
                    print "file not found ",f
                    continue
                symb = symbs.pop()
                N = res[:,0]
                turQRd2 = res[:,3]
                print "For N' = (%d,%d,%d)  time fit is in N^%.2f"%(ta, tb, tc, np.polyfit(np.log(N[14:]), np.log(turQRd2[14:]), 1)[0])
                SurQRd2 = res[:,9]
                plt.subplot(222)
                plt.loglog(N,turQRd2,symb,label='%d/%d'%(tb,tc))
                plt.subplot(224)
                plt.semilogx(N,SurQRd2,symb,label='%d/%d'%(tb,tc))
    plt.subplot(222)
    plt.title("c) compared speed for different urQRd approx.")
    plt.legend(loc=2)
    plt.axis(ymax=1E5)
    temps(bord=1E6)
    plt.subplot(224)
    plt.xlabel('Data length')
    plt.title("d) compared SNR for different urQRD approx.")
    snr(1E6)
    plt.axis(ymin=0, ymax=40)
    

def temps(bord = 1E6):
    "put decorations"
    plt.plot([1E3,bord],[1,1],'k--')      # 1 sec 
    plt.plot([1E3,bord],[60,60],'k--')     # 1 min
    plt.plot([1E3,bord],[3600,3600],'k--')   # 1 hour
#    plt.plot([1E3,1E6],[24*3600,24*3600],'k--')   # 1 day
    sh = 3.0/np.log(bord/1E3)
    plt.text(sh*bord,1.5,'1 sec')
    plt.text(sh*bord,90,'1 min')
    plt.text(sh*bord,4800,'1 hour')
def snr(bord = 1E6):
    plt.plot([1E3,bord],[10,10],'k--')
    plt.plot([1E3,bord],[20,20],'k--')
    plt.plot([1E3,bord],[30,30],'k--')
#    plt.plot([1E3,bord],[40,40],'k--')

def figure(loc):
    f = op.join(loc,"result.txt")
    plt.figure(figsize=(350/25.4,250/25.4))
    plt.subplots_adjust(left=0.06, bottom=0.07, right=0.97, top=0.95, wspace=0.12,hspace=0.20)
    plt.subplot(221)
    figure1(f)
    
    plt.subplot(223)
    figure2(f)

    figure3bis(loc)

    plt.show()

if __name__ == '__main__':
    figure("addditional_results")
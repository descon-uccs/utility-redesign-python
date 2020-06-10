# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 15:13:01 2020

@author: Philip Brown
"""

import numpy as np
import matplotlib.pyplot as plt

# uncomment the following lines to change your plotting settings:
#from matplotlib import rc
#rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
#rc('text',usetex=True)

boundFcn = lambda e : 1/(1+np.floor(e/2)/2)
boundFcn = np.vectorize(boundFcn)

ee = np.arange(0,13)
eeFine = np.arange(0,13,.01)
ones = [1]*len(eeFine)
Qbound = boundFcn(eeFine)

plt.clf()
plt.step(eeFine,Qbound,where='post')
plt.plot(eeFine,ones)
plt.fill_between(eeFine,ones,Qbound,color='red',alpha=0.2)
plt.ylim([0,1.1])
plt.scatter(ee,boundFcn(ee))

sz = 18
plt.xticks(fontsize=sz)
plt.yticks(fontsize=sz)

plt.xlabel(r'$e({\cal D})$',fontsize=sz)
plt.ylabel(r'$Q\left(f_{\rm max};{\cal G}^{\rm sm},{\cal N}\right)$',fontsize=sz)
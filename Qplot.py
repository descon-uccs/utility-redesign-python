# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 15:13:01 2020

@author: Philip Brown
"""

import numpy as np
import matplotlib.pyplot as plt


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
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
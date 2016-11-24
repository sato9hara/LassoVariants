# -*- coding: utf-8 -*-
"""
@author: satohara
"""

import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
from EnumerateLinearModel import EnumLasso

# setting - data
dim = 10
L = 5
num = 100
eps = 0.1
alpha = np.logspace(-3, 0, 11)

# setting - EnumLasso
rho = 0.01
ratio = 3
maxitr = 10000
tol = 1e-10
delta = 0

# test
ss = []
tt = []
seed = 0
count = 100
for c in range(count):
    print('seed = %2d' % (seed+c,))
    
    # test
    s = []
    t = []
    for a in alpha:
        
        # data
        np.random.seed(seed+c)
        V = np.random.randn(dim, L)
        A = V.dot(V.T)
        A /= np.linalg.norm(A) / dim
        B = (1 - a) * A + a * np.identity(dim)
        x = np.random.randn(num, dim).dot(B)
        y = x[:, 0] + x[:, 1] + eps * np.random.randn(num)
        
        # EnumLasso
        mdl = EnumLasso(rho=rho, r=ratio, maxitr=maxitr, tol=tol, delta=delta)
        mdl.fit(x, y)
        K = len(mdl.obj_)
        obj = []
        sord = np.nan
        stype = np.nan
        for i in range(K):
            nonzeros = np.where(np.abs(mdl.a_[i]) > 0)[0]
            obj.append(mdl.obj_[i])
            if np.isnan(sord):
                if set(nonzeros) <= set([0, 1]):
                    sord = i+1
                    if set(s) == set([0, 1]):
                        stype = 1
                    else:
                        stype = 0
        s.append(sord)
        t.append(stype)
    ss.append(s)
    tt.append(t)

# print
ss = np.array(ss)
mm = np.median(ss, axis=0)
pp1 = np.percentile(ss, 25, axis=0)
pp2 = np.percentile(ss, 75, axis=0)
ax = plt.subplot(111)
ax.set_xscale('log', nonposy='clip')
plt.errorbar(alpha, mm, yerr=[mm-pp1, pp2-mm])
plt.xlabel('log10(alpha)')
plt.ylabel('K')
plt.show()
plt.savefig('./rho%03d_seed%03d-%03d.pdf' % (int(100 * rho), seed, seed+count), format="pdf", bbox_inches="tight")
plt.close()
np.savetxt('./rho%03d_seed%03d-%03d.txt' % (int(100 * rho), seed, seed+count), np.c_[alpha, mm, pp1, pp2], fmt='%f', delimiter=',')

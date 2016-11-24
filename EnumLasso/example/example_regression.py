# -*- coding: utf-8 -*-
"""
@author: Satoshi Hara
"""

import sys
sys.path.append('../')

import numpy as np
from EnumerateLinearModel import EnumLasso

# setting
seed = 0
num = 1000
dim = 10
dim_extra = 3

# data
np.random.seed(seed)
X = np.random.randn(num, dim + dim_extra)
for i in range(dim_extra):
    X[:, dim + i] = 0.8 * X[:, 0] + 0.2 * np.random.randn(num)
y = X[:, 0] + X[:, 1] + 0.5 * np.random.randn(num)

# Alternate Lasso
mdl = EnumLasso(rho=0.1, verbose=True, modeltype='regression', enumtype='r', r=1.2)
mdl.fit(X, y)
print()
print(mdl)

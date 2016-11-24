# -*- coding: utf-8 -*-
"""
@author: Satoshi Hara
"""

import sys
sys.path.append('../')

import numpy as np
from AlternateLinearModel import AlternateLogisticLasso

# setting
seed = 0
num = 1000
dim = 2
dim_extra = 2

# data
np.random.seed(seed)
X = np.random.randn(num, dim + dim_extra)
for i in range(dim_extra):
    X[:, dim + i] = X[:, 0] + 0.5 * np.random.randn(num)
y = X[:, 0] + 0.3 * X[:, 1] + 0.5 * np.random.randn(num) > 0

# Alternate Lasso
mdl = AlternateLogisticLasso(rho=0.1, verbose=True)
mdl.fit(X, y)
print()
print(mdl)

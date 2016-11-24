# -*- coding: utf-8 -*-
"""
@author: satohara
"""

import sys
sys.path.append('../')

import codecs
import numpy as np
import pandas as pd
from EnumerateLinearModel import EnumLasso

# data - x
fn = './data/call_method_32.b'
df = pd.read_csv(fn, sep=',', header=None)
data_id_x = np.array([int(v) for v in df.ix[1, 2:]])
gene_id = df.ix[2:, :1].values
gene_id = np.array([[int(v[0]), int(v[1])] for v in gene_id])
data = df.ix[2:, 2:].values
data[data=='-'] = 0
data[data=='A'] = 1
data[data=='T'] = 2
data[data=='G'] = 3
data[data=='C'] = 4
count = np.c_[np.sum(data == 1, axis=1), np.sum(data == 2, axis=1), np.sum(data == 3, axis=1), np.sum(data == 4, axis=1)]
c = np.argmax(count, axis=1) + 1
x = data.copy()
for i in range(data.shape[1]):
    x[:, i] = 1 - (data[:, i] - c == 0)

# data - y
fn = './data/phenotype_published_raw.tsv'
with codecs.open(fn, 'r', 'Shift-JIS', 'ignore') as file:
    df = pd.read_table(file, delimiter='\t')
y = df.ix[:, 41].values

# data - reordering, remove nan
idx = np.argsort(data_id_x)
x = x[:, idx]
idx = ~np.isnan(y)
x = x[:, idx].T
y = y[idx]

# data - training & test split
seed = 0
r = 0.8
np.random.seed(seed)
idx = np.random.permutation(x.shape[0])
m = int(np.round(x.shape[0] * r))
xte = x[idx[m:], :]
yte = y[idx[m:]]
x = x[idx[:m], :]
y = y[idx[:m]]

# EnumLasso
rho = 0.1
delta = 0.05
mdl = EnumLasso(rho=rho, warm_start=True, enumtype='k', k=50, delta=delta, save='paper_thaliana.npy', modeltype='regression', verbose=True)
mdl.fit(x, y)
print()
print('--- Enumerated Solutions ---')
print(mdl)

# evaluate
print('--- Mean Square Error / # of Non-zeros ---')
for i in range(len(mdl.obj_)):
    a = mdl.a_[i]
    b = mdl.b_[i]
    z = xte.dot(a) + b
    mse = np.mean((z - yte)**2)
    print('Solution %3d: MSE = %f / NNZ = %d' % (i+1, mse, a.nonzero()[0].size))
    
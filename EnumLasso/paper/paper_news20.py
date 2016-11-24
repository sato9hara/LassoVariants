# -*- coding: utf-8 -*-
"""
@author: satohara
"""

import sys
sys.path.append('../')

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from EnumerateLinearModel import EnumLasso

# data
categories = ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware']
newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), categories=categories)
vectorizer = TfidfVectorizer(stop_words='english')
vectors = vectorizer.fit_transform(newsgroups_train.data)
vectors_test = vectorizer.transform(newsgroups_test.data)
x = vectors
y = newsgroups_train.target
xte = vectors_test
yte = newsgroups_test.target
vals = np.array(list(vectorizer.vocabulary_.values()))
keys = np.array(list(vectorizer.vocabulary_.keys()))
i = []
i.append(np.where(keys == 'want')[0][0])
print(i, [vals[j] for j in i])
i = [vals[j] for j in i]
x[:, i] = 0
xte[:, i] = 0

# EnumLasso
rho = 0.001
delta = 4.0
mdl = EnumLasso(rho=rho, warm_start=True, enumtype='k', k=50, delta=delta, save='paper_news20.npy', modeltype='classification', verbose=True)
mdl.fit(x, y, featurename=keys)
print()
print('--- Enumerated Solutions ---')
print(mdl)

# evaluate
print('--- Misclassification Rates ---')
for i in range(len(mdl.obj_)):
    a = mdl.a_[i]
    b = mdl.b_[i]
    t = (xte.dot(a) + b >= 0)
    acc = np.mean(t == yte)
    print('Solution %3d: Rate. = %f' % (i+1, 1 - acc))
    
# -*- coding: utf-8 -*-
"""
@author: Satoshi Hara
"""

import sys
sys.path.append('../')

import numpy as np
from AlternateLinearModel import AlternateLogisticLasso

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

# get news20 data (science)
categories = ['sci.med', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), categories=categories)
vectorizer = TfidfVectorizer(stop_words='english')
vectors = vectorizer.fit_transform(newsgroups_train.data)
vectors_test = vectorizer.transform(newsgroups_test.data)
x = vectors
y = newsgroups_train.target
vals = np.array(list(vectorizer.vocabulary_.values()))
keys = np.array(list(vectorizer.vocabulary_.keys()))
keys = keys[np.argsort(vals)]

# fit
rho=0.001
fn = 'news20_task2.npy'
mdl = AlternateLogisticLasso(rho=rho, verbose=True, save=fn)
mdl.fit(x, y, featurename=keys)

# print
print()
print(mdl)

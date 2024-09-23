import matplotlib.style as ms
import sklearn.cluster
from scipy.stats import pearsonr
from Utils import make_plots, LDA
ms.use('seaborn-muted')
import os
import pickle
import numpy as np

    """
      LDA fitting and projection of the features vectors
    """


""" single notes """
data_dir = '../../../Analysis/Note_collector'
data = open(os.path.normpath('/'.join([data_dir, 'AllExamples_win.pickle'])), 'rb')

features_pickle = pickle.load(data)
features = np.array(features_pickle['f'])

labels = np.array(features_pickle['labels'])
pianos = np.array(features_pickle['pianos'])
types = np.array(features_pickle['type'])
notes = np.array(features_pickle['note'])
d = np.array(features_pickle['d'])
c = features_pickle['c'][0]
del features_pickle
c = [c[:182], c[182:182+174], c[182+174:]]

f = np.concatenate((features, d), axis=1)
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(f)
f = scaler.transform(f)


projected_features = LDA(f, labels, 'all_w', c)
S_top = sklearn.metrics.silhouette_score(projected_features, labels)

######################################################
cov0, cov1 = [], []
for i in range(f.shape[1]-2):
    X = f[:, i].reshape(-1)
    Y0 = projected_features[:, 0].reshape(-1)
    Y1 = projected_features[:, 1].reshape(-1)
    covariance0, _ = pearsonr(X, Y0)
    covariance1, _ = pearsonr(X, Y1)
    cov0.append(covariance0)
    cov1.append(covariance1)
    print(i, '___', covariance0, covariance1)

################################################



""" Chords """

data = open(os.path.normpath('/'.join([data_dir, 'AllExamples_diff_aligned.pickle'])), 'rb')

features_pickle = pickle.load(data)
features = np.array(features_pickle['f'])

labels = np.array(features_pickle['labels'])
pianos = np.array(features_pickle['pianos'])
types = np.array(features_pickle['type'])
notes = np.array(features_pickle['note'])
d = np.array(features_pickle['d'])
c = features_pickle['c'][0]
del features_pickle
c = [c[:18], c[18:36], c[36:54:]]
f = np.concatenate((features, d), axis=1)
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(f)
f = scaler.transform(f)


projected_features = LDA(f, labels, 'all_w', c)
S_top = sklearn.metrics.silhouette_score(projected_features, labels)

######################################################
cov0, cov1 = [], []
for i in range(f.shape[1]-2):
    X = f[:, i].reshape(-1)
    Y0 = projected_features[:, 0].reshape(-1)
    Y1 = projected_features[:, 1].reshape(-1)
    covariance0, _ = pearsonr(X, Y0)
    covariance1, _ = pearsonr(X, Y1)
    cov0.append(covariance0)
    cov1.append(covariance1)
    print(i, '___', covariance0, covariance1)

################################################


""" repeated notes """

data = open(os.path.normpath('/'.join([data_dir, 'AllExamples_diff_aligned.pickle'])), 'rb')

features_pickle = pickle.load(data)
features = np.array(features_pickle['f'])

labels = np.array(features_pickle['labels'])
pianos = np.array(features_pickle['pianos'])
types = np.array(features_pickle['type'])
notes = np.array(features_pickle['note'])
d = np.array(features_pickle['d'])
c = features_pickle['c'][0]
del features_pickle
c = [c[54:60], c[60:78], c[78:]]
f = np.concatenate((features, d), axis=1)
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(f)
f = scaler.transform(f)


projected_features = LDA(f, labels, 'all_w', c)
S_top = sklearn.metrics.silhouette_score(projected_features, labels)

######################################################
cov0, cov1 = [], []
for i in range(f.shape[1]-2):
    X = f[:, i].reshape(-1)
    Y0 = projected_features[:, 0].reshape(-1)
    Y1 = projected_features[:, 1].reshape(-1)
    covariance0, _ = pearsonr(X, Y0)
    covariance1, _ = pearsonr(X, Y1)
    cov0.append(covariance0)
    cov1.append(covariance1)
    print(i, '___', covariance0, covariance1)

################################################

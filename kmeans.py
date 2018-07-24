import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

N = 20
BINS = 100
pi = np.pi

x = pd.read_csv('../../data_fixed/x.txt', sep = " ", header = None)

del x[N + 1]
del x[N]

x[x < 0] += (2 * pi)
x[x > (2 * pi)] -= (2 * pi)

x = np.asarray(x)
for i in range(0, len(x[:,0])):
    x[i,:] = np.sort(x[i,:])
x = pd.DataFrame(x)

xdif = x.copy()
for i in range(1, N):
    xdif[i] = x[i] - x[i - 1]

#Normalize the data
normalize = False
f = open('normalize.txt', 'w')
f.write('%s' % normalize)
f.close()
if normalize == True:
    xdif = xdif.apply(lambda x: (x - np.mean(x)) / np.std(x))
    print "DATA IS NORMALIZED!"
else:
    print "DATA IS UNNORMALIZED!"

#Sample from it
pop = 100000
samples = np.random.choice(range(0, len(xdif[0])), pop, replace = False)
xdif = xdif.iloc[samples].reset_index()
del xdif['index']
f = open('pop.txt', 'w')
f.write('%f' % pop)
f.close()
np.savetxt('samples.txt', samples)

#Want to do it with all 3 adjacent spacings, not just one
#Something about a transition matrix between clusters
#First make the matrix.  Need to do something like:
#d0 d1 d2 .. dND
#d1 d2 d3 .. dND+1
#d2 d3 d4 .. dND+2
#etc, so it'll be len(x[0]) * (N - 2) length now
ND = 2
f = open('ND.txt', 'w')
f.write('%f' % ND)
f.close()
xdifadj = pd.DataFrame(xdif[range(0, ND)], index = range(0, len(xdif[0])), columns = range(0, ND))
for i in range(1, N - (ND - 1)):
    z = xdif.drop([j for j in range(0, i)], axis = 1)
    z.columns = range(0, N - i)
    xdifadj = xdifadj.append(z[range(0, ND)], ignore_index = True)

n_clusters = 9
f = open('n_clusters.txt', 'w')
f.write('%f' % n_clusters)
f.close()

model = KMeans(n_clusters = n_clusters, n_jobs = -1).fit(xdifadj)

#np.savetxt('clustercenters.txt', model.cluster_centers_)
#np.savetxt('labels.txt', model.labels_)
#np.savetxt('inertia.txt', model.inertia_)
from sklearn.externals import joblib
joblib.dump(model, 'kmeans.pkl')

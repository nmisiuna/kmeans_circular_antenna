import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

colors = ['pink', 'red', 'orange', 'green', 'cyan', 'blue', 'purple', 'brown', 'black', 'grey']

fontsize = 15

N = 20
BINS = 1024
pi = np.pi

x = pd.read_csv('../../data_fixed/x.txt', sep = " ", header = None)
cost = -pd.read_csv('../../data_fixed/fitness.txt', sep = " ", header = None)

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
f = open('normalize.txt', 'r')
normalize = f.read()
f.close()
if normalize == 'True':
    xdif = xdif.apply(lambda x: (x - np.mean(x)) / np.std(x))
    print "DATA IS NORMALIZED!"
else:
    print "DATA IS UNNORMALIZED!"

#Sample from it
pop = int(np.loadtxt('pop.txt'))
samples = np.loadtxt('samples.txt')
xdif = xdif.iloc[samples].reset_index()
del xdif['index']

#Want to do it with all adjacent spacings, not just one
#Something about a transition matrix between clusters
#First make the matrix.  Need to do something like:
#d0 d1
#d1 d2
#d2 d3
#etc, so it'll be len(x[0]) * (N - 1) length now
ND = int(np.loadtxt('ND.txt'))
xdifadj = pd.DataFrame(xdif[range(0, ND)], index = range(0, len(xdif[0])), columns = range(0, ND))
for i in range(1, N - (ND - 1)):
    z = xdif.drop([j for j in range(0, i)], axis = 1)
    z.columns = range(0, N - i)
    xdifadj = xdifadj.append(z[range(0, ND)], ignore_index = True)

n_clusters = int(np.loadtxt('n_clusters.txt'))
from sklearn.externals import joblib
model = joblib.load('kmeans.pkl')

print model.cluster_centers_
np.savetxt('clustercenters.txt', np.around(model.cluster_centers_, decimals = 4))
print model.labels_
print model.inertia_

#I also need to save the seqpair, which I'm not doing, thus invalidating all of
#my previous results
seqpair = np.zeros((n_clusters, n_clusters))
labels = model.labels_.reshape((N - ND + 1, pop)).T
for row in range(0, pop):
    for n in range(0, N - ND):
        seqpair[labels[row][n]][labels[row][n + 1]] += 1
#Now I need to normalize each row
for row in range(0, n_clusters):
    seqpair[row][:] /= np.sum(seqpair[row][:])
np.savetxt('seqpair.txt', seqpair, fmt = '%1.2e')


fig = plt.figure()
plt.imshow(seqpair, origin = 'upper', cmap = 'BuPu', interpolation = 'none')
for (j,i),label in np.ndenumerate(seqpair):
    if label < 0.75:
        plt.annotate(np.around(label, decimals = 2), (i - 0.25, j + 0.1), size = 10, color = 'black')
    else:
        plt.annotate(np.around(label, decimals = 2), (i - 0.25, j + 0.1), size = 10, color = 'white')

plt.xticks(range(0, 9))
plt.yticks(range(0, 9))
plt.savefig('transmatrix.pdf', bbox_inches = 'tight')
plt.savefig('transmatrix.eps', bbox_inches = 'tight')
plt.show(block = False)

#Also need the figure of the pdf of adjacent positions overlaid with labels
#Which I deleted.  Goddamit.
count, xedges, yedges = np.histogram2d(xdifadj[0], xdifadj[1], bins = 100, range = [[0, 1.25], [0, 1.25]], normed = True)
fig = plt.figure(facecolor = 'white')
plt.imshow(count, extent = [0, 1.25, 0, 1.25], cmap = 'hot', origin = 'low', interpolation = 'none', rasterized = True)
for i in range(0, n_clusters):
    if i != 5:
        plt.annotate('%s' % i, (model.cluster_centers_[i][0], model.cluster_centers_[i][1]), bbox=dict(boxstyle='circle', fc = 'none', ec = 'white', alpha = 0.8), size = 15, color = 'white')
    else:
        plt.annotate('%s' % i, (model.cluster_centers_[i][0], model.cluster_centers_[i][1]), bbox=dict(boxstyle='circle', fc = 'none', ec = 'black', alpha = 0.8), size = 15, color = 'black')
plt.xlabel('$d_i$', fontsize = fontsize, fontweight = 'bold')
plt.ylabel('$d_{i+1}$', fontsize = fontsize, fontweight = 'bold')
clb = plt.colorbar()
clb.set_label('Prob. Dens.', labelpad = -40, y = 1.05, rotation = 0)
plt.savefig('pdfdadj.pdf', dpi = 400, facecolor = fig.get_facecolor(), transparent = True, bbox_inches = 'tight')
plt.savefig('pdfdadj.eps', dpi = 400, facecolor = fig.get_facecolor(), transparent = True, bbox_inches = 'tight')
plt.show(block = False)


#Now let's look at the contiguous time spent in:
#Chandra: below a boundary of x-int: 0.75, y-int: 0.75
#Me: states 0 or 8 or 4 (three closest to origin)

#Chandra's: let's just do x + y < 0.75
#So, I need to look at the actual data, not the labels

cutoff = 0.7
contiguous = np.zeros(N - ND + 1)
contiguous_cost = np.zeros(N - ND + 1)
ind = np.zeros(N - ND + 1)
total_count = 0
scatter_cost = pd.DataFrame(index = range(0, pop * (N - ND + 1)), columns = range(0, 2))
total = 0
for row in range(0, pop):
    count = 0
    for n in range(0, N - ND + 1):
        if (xdif.iat[row, n] + xdif.iat[row, n + 1]) < cutoff:
            total_count += 1
            count += 1
        elif count > 0:
            contiguous[count - 1] += 1
            contiguous_cost[count - 1] += cost.iloc[row]
            ind[count - 1] += 1
#            scatter_cost = scatter_cost.append([count - 1, cost.iloc[row]], ignore_index = True)
            scatter_cost.iat[total, 0] = count
            scatter_cost.iat[total, 1] = cost.iat[row, 0]
            count = 0
            total += 1
        else:
            count = 0

#Contiguous is the count of each length divided by the total number of occurences
#Interpretation: probability of X length given condition already met
#Starts at 1
#contiguous /= total_count
contiguous /= np.sum(contiguous)
#np.savetxt('contiguous.txt', contiguous)

#Mine: let's just track states 0, 8, 4 using OR
contiguous_label = np.zeros(N - ND + 1)
contiguous_label_total = np.zeros((n_clusters, N - ND + 1))
total_count = 0
for row in range(0, pop):
    count = 1
    for n in range(1, N - ND + 1):
#        if labels[row][n] in [1]:
#            count += 1
#            total_count += 1
#        elif count > 0:
#            contiguous_label[count - 1] += 1
#            count = 0
#        else:
#            count = 0
        #I'd like to track every contiguous label
        if labels[row][n] == labels[row][n - 1]:
            count += 1
            total_count += 1
        else:
            contiguous_label_total[labels[row][n - 1]][count - 1] += 1
            count = 1

#contiguous_label /= np.sum(contiguous_label)
for i in range(0, n_clusters):
    contiguous_label_total[i][:] /= np.sum(contiguous_label_total[i][:])

#np.savetxt('contiguouslabel.txt', contiguous_label)
np.savetxt('contiguouslabelall.txt', contiguous_label_total)

#Plot only her way
fig = plt.figure()
#plt.plot(range(1, N - ND + 1 + 1), contiguous, label = 'bounded')
#plt.plot(range(1, N - ND + 1 + 1), contiguous_label, label = 'labels = 8')
for n in range(0, n_clusters):
#   plt.plot(range(1, N - ND + 1 + 1), contiguous_label_total[n][:], color = colors[n], label = 'labels = %s' % n)
   plt.plot(range(1, N - ND + 1 + 1), contiguous_label_total[n][:], color = colors[n % len(colors)], label = 'labels = %s' % n)
plt.legend()
#plt.xscale('log')
plt.yscale('log')
plt.xscale('log')
plt.grid(True)
plt.xticks(range(1, N - ND + 1 + 1))
plt.savefig('contiguous.pdf', bbox_inches = 'tight')
plt.show(block = False)


#Plot run length vs cost
for i in range(0, N - ND + 1):
    contiguous_cost[i] /= ind[i]
fig = plt.figure()
plt.plot(range(1, N - ND + 1 + 1), contiguous_cost, label = 'bounded')
plt.legend()
plt.grid(True)
plt.xticks(range(1, N - ND + 1 + 1))
plt.savefig('contiguouscost.pdf', bbox_inches = 'tight')
plt.show(block = False)

#Scatter of it
fig = plt.figure()
plt.scatter(scatter_cost[0], scatter_cost[1], rasterized = True)
plt.legend()
plt.grid(True)
plt.xticks(range(1, N - ND + 1 + 1))
plt.savefig('contiguousscattercost.pdf', bbox_inches = 'tight')
plt.show(block = False)

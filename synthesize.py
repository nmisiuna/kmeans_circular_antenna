import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cmath
import scipy

N = 20
BINS = 1024
pi = np.pi
pio2 = pi / 2.0
ENSEMBLES = 1000
KHAT = 5 * pi
umin = 0
umax = 5*pi
NU = 512
du = (umax-umin)/NU
u = np.linspace(umin,umax,NU)
phi_sweep = np.zeros(NU)
total_gen_x = np.zeros(N)
R = np.zeros((ENSEMBLES, NU), dtype = np.complex_)
cost = np.zeros(ENSEMBLES)

des = pd.read_csv('../../data_fixed/mean.txt', sep = " ", header = None)
theta_T = pio2
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

mu = np.mean(xdif, axis = 0)
std = np.std(xdif, axis = 0)

#Sample from it
pop = np.loadtxt('pop.txt')
samples = np.loadtxt('samples.txt')
xdif = xdif.iloc[samples].reset_index()
del xdif['index']

ND = int(np.loadtxt('ND.txt'))

xdifadj = pd.DataFrame(xdif[range(0, ND)], index = range(0, len(xdif[0])), columns = range(0, ND))
for i in range(1, N - (ND - 1)):
    z = xdif.drop([j for j in range(0, i)], axis = 1)
    z.columns = range(0, N - i)
    xdifadj = xdifadj.append(z[range(0, ND)], ignore_index = True)

#Load if normalize or not
f = open('normalize.txt', 'r')
normalize = f.read()
f.close()

#Load the model
n_clusters = np.loadtxt('n_clusters.txt')
from sklearn.externals import joblib
model = joblib.load('kmeans.pkl')


#Grab the run length of label 8 (VERIFY with pdfadj)
#Make into a cdf
chosen = [4]
length = np.loadtxt('contiguouslabelall.txt')
for i in range(0, n_clusters):
    length[i][:] = np.cumsum(length[i][:])
#length = np.cumsum(np.loadtxt('contiguouslabel.txt'))


#Generate via pdf or fixed values?
#pdf, fixed, segmented, centergravity, length, 2dgaussian, samplemean
method = 'fixed'
#Adjust generated values to be cloesr to empirical?
adjusted = False
adj = 0.25
scale = 0.1

#Load the transition matrix
seqpair = np.loadtxt('seqpair.txt')

#Set CHOSEN label prob of staying in state to 0
#As that is handled via run_length
if method == 'length':
    for i in chosen:
        seqpair[i][i] = 0

#Turn the transition matrix into a set of cumulative probabilities
for row in range(0, n_clusters):
    seqpair[row][:] = np.cumsum(seqpair[row][:])
    seqpair[row][:] /= seqpair[row][n_clusters - 1]



#To generate the first label I need to get the probability of all labels
#and draw from that
labels_prob, edges = np.histogram(model.labels_, bins = n_clusters, range = [0, n_clusters], normed = True)
np.savetxt('labels_prob.txt', labels_prob)
labels_prob = np.cumsum(labels_prob)

#Also make a 2D pdf of each label and use it for sampling from that label
count = np.zeros((n_clusters, BINS, BINS))
xedges = np.zeros((n_clusters, BINS + 1))
yedges = np.zeros((n_clusters, BINS + 1))
tables = [[] for i in range(0, n_clusters)]
for i in range(0, n_clusters):
    z = xdifadj.iloc[model.labels_ == i]
    count[i], xedges[i], yedges[i] = np.histogram2d(z[0], z[1], bins = ((BINS, BINS)), range = [[0, 2.0], [0, 2.0]], normed = True)
    index0, index1 = np.meshgrid(np.arange(0, 2.0, 2.0 / BINS), np.arange(0, 2.0, 2.0 / BINS))
    table = np.vstack((count[i].ravel(), index0.ravel(), index1.ravel())).T
    table = table[np.nonzero(table[:, 0]), :][0]
    table[:, 0] = np.cumsum(table[:, 0] * (xedges[i][1] - xedges[i][0]) * (yedges[i][1] - yedges[i][0]))
    tables[i] = table

gen_x = np.zeros((ENSEMBLES, N))
labels_saved = np.zeros((ENSEMBLES, N - ND + 1))

#Calculate the centroid of a pandas dataframe with 2 columns
def calc_centroid(arr):
    return np.sum(arr[0]) / arr.shape[0], np.sum(arr[1]) / arr.shape[0]

centroids = [[] for i in range(0, n_clusters)]
for i in range(0, n_clusters):
    centroids[i] = calc_centroid(xdifadj[model.labels_ == i])

#!!!!
#THE CLUSTER CENTERS ARE THE CLUSTER CENTROIDS
#CONFUSING TERMINOLOGY, CENTER OF THE POLYGON != CENTROID OF POINTS
#FUCK
#!!!!

for ensemble in range(0, ENSEMBLES):
    #Generate the first label
    temp = np.random.uniform(0, 1)
    j = 0
    while labels_prob[j] < temp:
        j += 1
    prev_label = j
    labels_saved[ensemble][0] = prev_label
    #This gives me the first two values
    #Sample from the 2d pdf
    temp = np.random.uniform(0, 1)
    j = 0
    while tables[prev_label][j, 0] < temp:
        j += 1
    
    if method == 'pdf':
        #Generate labels sequentially with values drawn from 2d pdf of cluster
        for i in range(0, ND):
            gen_x[ensemble][i] = tables[prev_label][j, 1 + i]
    elif method == 'fixed':
        if adjusted == False:
            #Generate labels sequentially with values at center of cluster
            for i in range(0, ND):
                gen_x[ensemble][i] = model.cluster_centers_[prev_label, i]
        else:
            for i in range(0, ND):
                gen_x[ensemble][i] = model.cluster_centers_[prev_label, i] + (adj * (mu[i] - model.cluster_centers_[prev_label, i]))
    elif method == 'segmented':
        #Generate all labels in a segmented manner (no overlap)
        for i in range(0, N / ND):
            temp = np.random.uniform(0, 1)
            j = 0
            while labels_prob[j] < temp:
                j += 1
            prev_label = j
            for k in range(0, ND):
                gen_x[ensemble][i * ND + k] = model.cluster_centers_[prev_label, k]
    elif method == 'centergravity':
        #Generate labels sequentially with values at center of gravity
#        centroid = calc_centroid(xdifadj[model.labels_ == prev_label])
        gen_x[ensemble][0] = centroids[prev_label][0]
        gen_x[ensemble][1] = centroids[prev_label][1]
    elif method == 'length':
        #First two elements are always going in the first label
        gen_x[ensemble][0] = model.cluster_centers_[prev_label, 0]
        gen_x[ensemble][1] = model.cluster_centers_[prev_label, 1]
        #If the label is the one near the origin (CHECK pdfdadj TO FIND OUT)
        #sample the run length
        if prev_label in chosen:
            temp = np.random.uniform(0, 1)
            j = 0
            while length[prev_label][j] < temp:
                j += 1
            run_length = j
            prev_chosen = prev_label
        else:
            run_length = 0
    elif method == '2dgaussian':
        gen_x[ensemble][0] = np.random.normal(loc = model.cluster_centers_[prev_label, 0], scale = scale)
        gen_x[ensemble][1] = np.random.normal(loc = model.cluster_centers_[prev_label, 1], scale = scale)
    elif method == 'samplemean':
        gen_x[ensemble][0] = mu[0]
        gen_x[ensemble][1] = mu[1]
    elif method == 'samplemeanvar':
        gen_x[ensemble][0] = np.random.normal(loc = mu[0], scale = std[0])
        gen_x[ensemble][1] = np.random.normal(loc = mu[1], scale = std[1])
    else:
        print "Missing method!"
        quit()

    #Generate remaining labels
    for n in range(0, N - ND):
        #Now move to another label based on the transition matrix
        temp = np.random.uniform(0, 1)
        j = 0
        while seqpair[prev_label][j] < temp:
            j += 1
        prev_label = j
        labels_saved[ensemble][1 + n] = prev_label
        
        temp = np.random.uniform(0, 1)
        j = 0
        while tables[prev_label][j, 0] < temp:
            j += 1

        if method == 'pdf':
            gen_x[ensemble][ND + n] = tables[prev_label][j, ND - 1]
        elif method == 'fixed':
            if adjusted == False:
                gen_x[ensemble][ND + n] = model.cluster_centers_[prev_label, ND - 1]
            else:
                gen_x[ensemble][ND + n] = model.cluster_centers_[prev_label, ND - 1] + (adj * (mu[ND + n] - model.cluster_centers_[prev_label, ND - 1]))
        elif method == 'segmented':
            #If segmented, we already placed all elements
            break
        elif method == 'centergravity':
#            centroid = calc_centroid(xdifadj[model.labels_ == prev_label])
#            gen_x[ensemble][ND + n] = centroid[1]
            gen_x[ensemble][ND + n] = centroids[prev_label][1]
        elif method == 'length':
            #If run_length is 0, sample normally
            if run_length == 0:
                gen_x[ensemble][ND + n] = model.cluster_centers_[prev_label, ND - 1]
                #If the new label is CHOSEN, sample run_length
                if prev_label in chosen:
                    temp = np.random.uniform(0, 1)
                    j = 0
                    while length[prev_label][j] < temp:
                        j += 1
                    run_length = j
                    prev_chosen = prev_label
            #Otherwise, place in CHOSEN label and decrease run_length
            else:
                run_length -= 1
                gen_x[ensemble][ND + n] = model.cluster_centers_[prev_chosen, ND - 1]
                prev_label = prev_chosen
            #Error catching in case run_length goes negative
            if run_length < 0:
                print "MAJOR PROBLEM"
                quit()
             
        elif method == '2dgaussian':
            gen_x[ensemble][ND + n] = np.random.normal(loc = model.cluster_centers_[prev_label, ND - 1], scale = scale)
        elif method == 'samplemean':
            gen_x[ensemble][ND + n] = mu[ND + n]
        elif method == 'samplemeanvar':
            gen_x[ensemble][ND + n] = np.random.normal(loc = mu[ND + n], scale = std[ND + n])
        else:
            print "Missing method!"
            quit()

    #Convert back from zero mean data
    for i in range(0, N):
        if normalize == 'True':
            gen_x[ensemble][i] *= std[i]
            gen_x[ensemble][i] += mu[i]
        if i > 0:
            gen_x[ensemble][i] += gen_x[ensemble][i - 1]


    #Compute the beam pattern
    for i in range(0, NU):
        for j in range(0, N):
            R[ensemble][i] += cmath.exp(complex(0, -1) * (KHAT * cmath.cos((i * (2 * pi) / NU) - gen_x[ensemble][j]) - KHAT * cmath.cos(theta_T - gen_x[ensemble][j])))
                
        R[ensemble][i] = R[ensemble][i] / N

    #Compute cost
    for j in range(0, NU):
        #Only compute the cost if it's above the desired
        if((20 * np.log10(abs(R[ensemble][j]))) > des[2][j]):
            cost[ensemble] += ((20 * np.log10(abs(R[ensemble][j]))) - des[2][j]) * ((20 * np.log10(abs(R[ensemble][j]))) - des[2][j])
    #Normalize with frequency resolution so higher resol != higher cost
    cost[ensemble] = cost[ensemble] / NU


#Now save it all to file
np.savetxt('./data/mean_%s.txt' % method, abs(np.mean(R, axis = 0)), fmt = '%f')
np.savetxt('./data/var_%s.txt' % method, np.var(R, axis = 0), fmt = '%f')
np.savetxt('./data/x_%s.txt' % method, gen_x, fmt = '%f')
np.savetxt('./data/cost_%s.txt' % method, cost, fmt = '%f')
np.savetxt('./data/labels_%s.txt' % method, labels_saved, fmt = '%f')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

colors = ['pink', 'red', 'orange', 'green', 'cyan', 'blue', 'purple', 'brown', 'black', 'grey']

method = 'fixed'

x = pd.read_csv('./data/x_%s.txt' % method, sep = " ", header = None)
xff = pd.read_csv('../../data_fixed/x.txt', sep = " ", header = None)
Rmean = pd.read_csv('./data/mean_%s.txt' % method, sep = " ", header = None)
Rvar = pd.read_csv('./data/var_%s.txt' % method, sep = " ", header = None)
cost = pd.read_csv('./data/cost_%s.txt' % method, sep = " ", header = None)
ffcost = pd.read_csv('../../data_fixed/fitness.txt', sep = " ", header = None)
des = pd.read_csv('../../data_fixed/mean.txt', sep = " ", header = None)

BINS = 100
N = 20
pi = np.pi

fontsize = 14

ND = int(np.loadtxt('ND.txt'))

#Fix normal position vector
del xff[N + 1]
del xff[N]
xff[xff < 0] += (2 * pi)
xff[xff > (2 * pi)] -= (2 * pi)

xff = np.asarray(xff)
for i in range(0, len(xff[:,0])):
    xff[i,:] = np.sort(xff[i,:])
xff = pd.DataFrame(xff)

#Overall positions
#SAMPLE?
#pop = 1000
#samples = np.random.choice(range(0, len(xff[0])), pop, replace = False)
#xff = xff.iloc[samples].reset_index()
#del xff['index']

count, edges = np.histogram(x, bins = BINS, range = [0, 4 * pi], normed = True)
fig = plt.figure()
centers = edges[1:] - ((edges[1] - edges[0]) / 2.0)
plt.plot(centers, count, color = 'blue', label = 'Model C$_{\mathtt{I}}$')
count, edges = np.histogram(xff, bins = BINS, range = [0, 2 * pi], normed = True)
centers = edges[1:] - ((edges[1] - edges[0]) / 2.0)
plt.plot(centers, count, color = 'red', label = 'Firefly')
plt.xlim((0, 8))
plt.legend()
plt.xlabel('x (rad)')
plt.ylabel('Probability Density')
plt.savefig('pdfx.eps', bbox_inches = 'tight')
plt.savefig('pdfx.pdf', bbox_inches = 'tight')
plt.show(block = False)

#Spacings
xdif = x.copy()
for i in range(1, N):
    xdif[i] = x[i] - x[i - 1]

xffdif = xff.copy()
for i in range(1, N):
    xffdif[i] = xff[i] - xff[i - 1]
count, edges = np.histogram(xdif.drop(0, axis = 1), bins = BINS, range = [0, 2], normed = True)
fig = plt.figure()
centers = edges[1:] - ((edges[1] - edges[0]) / 2.0)
plt.plot(centers, count, color = 'blue', label = 'k-means')
count, edges = np.histogram(xffdif.drop(0, axis = 1), bins = BINS, range = [0, 2], normed = True)
plt.plot(centers, count, color = 'red', label = 'firefly')
plt.legend()
plt.xlabel('$d$ (rad)')
plt.ylabel('Probability Density')
plt.savefig('pdfd.pdf', bbox_inches = 'tight')
plt.savefig('pdfd.eps', bbox_inches = 'tight')
plt.show(block = False)

#Beam pattern
fig = plt.figure()
plt.plot(des[0], des[2], color = 'black')
plt.plot(des[0], 20 * np.log10(Rmean), color = 'blue')
plt.plot(des[0], 20 * np.log10(Rmean + np.sqrt(Rvar)), '--', color = 'blue')
plt.plot(des[0], des[1], color = 'red')
plt.xlabel('$\phi$', fontsize = fontsize)
plt.ylabel('$R(\phi)$', fontsize = fontsize)

plt.text(2.3, -2.5, 'Pedestal', fontweight = 'normal', fontsize = (fontsize + 5))

plt.text(4.5, -7.5, '$\mu_{\mathtt{C_I}}+\sigma_{\mathtt{C_I}}$', fontweight = 'normal', fontsize = (fontsize + 5))
plt.arrow(4.9, -8.5, 0, -2, head_width = 0.1, head_length = 1, fc = 'k', ec = 'k')

plt.text(3.25, -60, '$\mu_{f\!f}$', fontweight = 'normal', fontsize = (fontsize + 5))
plt.arrow(3.35, -57, -.15, 3, head_width = 0.1, head_length = 1, fc = 'k', ec = 'k')

plt.text(4.5, -45, '$\mu_{\mathtt{C_I}}$', fontweight = 'normal', fontsize = (fontsize + 5))
plt.arrow(4.6, -42, -.25, 6.5, head_width = 0.1, head_length = 1, fc = 'k', ec = 'k')

plt.xlim((0, 2 * pi))
plt.savefig('meanplusstd.eps', bbox_inches = 'tight')
plt.savefig('meanplusstd.pdf', bbox_inches = 'tight')
plt.show(block = False)

#Cost

fontsize = 12

fitnessind = pd.read_csv('../../sample_xind/data/cost_sample.txt', sep = " ", header = None)
fitnessall = pd.read_csv('../../sample_xall/data/cost_sample.txt', sep = " ", header = None)
fitnessxjoint = pd.read_csv('../../sample_xjoint/data/cost_sample.txt', sep = " ", header = None)
fitnessxdifjoint = pd.read_csv('../../sample_xdifjoint/data/cost_sample.txt', sep = " ", header = None)
fitnessq1 = pd.read_csv('../../sample_q1/data/cost_sample.txt', sep = " ", header = None)
fitnessfullq1 = pd.read_csv('../../sample_fullq1/data/cost_sample.txt', sep = " ", header = None)
fitnesshmm = pd.read_csv('../../hmm/gen_cost.txt', sep = " ", header = None)


cost_all = [-ffcost, fitnessall, fitnessind, fitnessxjoint, fitnessxdifjoint, fitnessq1, fitnessfullq1, cost, fitnesshmm]
mean_all = []
var_all = []
for i in range(0, 9):
    mean_all.append(np.mean(cost_all[i])[0])
    var_all.append(np.var(cost_all[i])[0])


fig = plt.figure(facecolor = 'white', figsize = (18.0, 12.0))
#plt.scatter(mean_all, var_all)
plt.xlim((0, 10))
plt.ylim((0, 12))
labels = ["ff", "xall", "xind", "xjoint", "xdifjoint", "q1", "full q1", "kmeansND%s" % ND, "ghmm"]
for i, txt in enumerate(labels):
    plt.annotate(txt, (mean_all[i], var_all[i]), bbox=dict(boxstyle='circle', fc = 'none', ec = 'black', alpha = 0.8), size = 15)
plt.xlabel('Mean Cost', fontweight = 'bold', fontsize = fontsize)
plt.ylabel('Var. of Cost', fontweight = 'bold', fontsize = fontsize)
fig.savefig('meanvarauto.png', facecolor = fig.get_facecolor(), transparent = True)
fig.savefig('meanvarauto.eps', facecolor = fig.get_facecolor(), transparent = True, bbox_inches = 'tight')
fig.savefig('meanvarauto.pdf', facecolor = fig.get_facecolor(), transparent = True)
plt.show(block = False)


cost_all = [-ffcost, cost]
mean_all = []
var_all = []
for i in range(0, 2):
    mean_all.append(np.mean(cost_all[i])[0])
    var_all.append(np.var(cost_all[i])[0])

fig = plt.figure(facecolor = 'white', figsize = (18.0, 12.0))
#plt.scatter(mean_all, var_all)
plt.xlim((0, 4))
plt.ylim((0, 6))
labels = ["ff", "kmeansND%s" % ND]
for i, txt in enumerate(labels):
    plt.annotate(txt, (mean_all[i], var_all[i]), bbox=dict(boxstyle='circle', fc = 'none', ec = 'black', alpha = 0.8), size = 15)
plt.xlabel('Mean Cost', fontweight = 'bold', fontsize = fontsize)
plt.ylabel('Var. of Cost', fontweight = 'bold', fontsize = fontsize)
fig.savefig('meanvarauto.png', facecolor = fig.get_facecolor(), transparent = True)
fig.savefig('meanvarauto2.eps', facecolor = fig.get_facecolor(), transparent = True, bbox_inches = 'tight')
fig.savefig('meanvarauto2.pdf', facecolor = fig.get_facecolor(), transparent = True)
plt.show(block = False)


fig = plt.figure()
count, edges = np.histogram(cost, bins = BINS, range = [0, 10], normed = True)
centers = edges[1:] - ((edges[1] - edges[0]) / 2.0)
plt.plot(centers, count, color = 'blue', label = 'Model C$_{\mathtt{I}}$')
count, edges = np.histogram(-ffcost, bins = BINS, range = [0, 10], normed = True)
centers = edges[1:] - ((edges[1] - edges[0]) / 2.0)
plt.plot(centers, count, color = 'red', label = 'Firefly')
plt.legend()
plt.xlabel('Cost')
plt.ylabel('Probability Density')
plt.savefig('pdfcost.pdf', bbox_inches = 'tight')
plt.savefig('pdfcost.eps', bbox_inches = 'tight')
plt.show(block = False)



#Now break it down via cost
cutoff = 1.2
cost = np.asarray(cost).ravel()

#Let's look at those above and below cost
count, edges = np.histogram(xdif[cost < cutoff], bins = BINS, range = [0, 2], normed = True)
fig = plt.figure()
centers = edges[1:] - ((edges[1] - edges[0]) / 2.0)
plt.plot(centers, count, color = 'red')
count, edges = np.histogram(xdif[cost >= cutoff], bins = BINS, range = [0, 2], normed = True)
plt.plot(centers, count, color = 'blue')
plt.savefig('pdfdcost.pdf', bbox_inches = 'tight')
plt.show(block = False)
#They're the same distributions, so this path is fruitless
#Let's look at the labels, maybe there's something there

labels = np.loadtxt('./data/labels.txt')
n_clusters = np.loadtxt('n_clusters.txt')
count, edges = np.histogram(labels[cost < cutoff], bins = n_clusters, range = [0, n_clusters], normed = True)
fig = plt.figure()
centers = range(0, n_clusters)
plt.plot(centers, count, color = 'red')
count, edges = np.histogram(labels[cost > cutoff], bins = n_clusters, range = [0, n_clusters], normed = True)
plt.plot(centers, count, color = 'blue')
plt.savefig('pdflabelcost.pdf', bbox_inches = 'tight')
plt.show(block = False)


#Compare all the different methods
costpdf = pd.read_csv('./data/cost_pdf.txt', sep = " ", header = None)
costfixed = pd.read_csv('./data/cost_fixed.txt', sep = " ", header = None)
costsegmented = pd.read_csv('./data/cost_segmented.txt', sep = " ", header = None)
costcentergravity = pd.read_csv('./data/cost_centergravity.txt', sep = " ", header = None)
costlength = pd.read_csv('./data/cost_length.txt', sep = " ", header = None)


fig = plt.figure()
count, edges = np.histogram(costpdf, bins = BINS, range = [0, 20], normed = True)
centers = edges[1:] - ((edges[1] - edges[0]) / 2.0)
plt.plot(centers, count, label = 'pdf')
count, edges = np.histogram(costfixed, bins = BINS, range = [0, 20], normed = True)
plt.plot(centers, count, label = 'fixed')
count, edges = np.histogram(costsegmented, bins = BINS, range = [0, 20], normed = True)
plt.plot(centers, count, label = 'seg')
count, edges = np.histogram(costcentergravity, bins = BINS, range = [0, 20], normed = True)
plt.plot(centers, count, label = 'cgrav')
count, edges = np.histogram(costlength, bins = BINS, range = [0, 20], normed = True)
plt.plot(centers, count, label = 'length')
plt.legend()
plt.savefig('pdfcostall.pdf', bbox_inches = 'tight')
plt.show(block = False)



#Individual spacing means/var
fig = plt.figure()
#plt.scatter(range(0, N), np.mean(xdif, axis = 0), color = 'blue', label = 'Synth.')
plt.plot(np.mean(xdif, axis = 0), color = 'blue', label = 'Synth.')
#plt.scatter(range(0, N), np.mean(xffdif, axis = 0), color = 'red', label = 'ff')
plt.plot(np.mean(xffdif, axis = 0), color = 'red', label = 'ff')
plt.legend()
plt.ylabel('Mean Spacing')
plt.xticks(range(0, N))
plt.grid()
plt.savefig('xdifmean.pdf', bbox_inches = 'tight')
plt.show(block = False)

fig = plt.figure()
plt.plot(np.var(xdif, axis = 0), color = 'blue', label = 'Synth.')
plt.plot(np.var(xffdif, axis = 0), color = 'red', label = 'ff')
#plt.scatter(range(0, N), np.var(xdif, axis = 0), color = 'blue', label = 'Synth.')
#plt.scatter(range(0, N), np.var(xffdif, axis = 0), color = 'red', label = 'ff')
plt.legend()
plt.ylabel('Var. Spacing')
plt.xticks(range(0, N))
plt.grid()
plt.savefig('xdifvar.pdf', bbox_inches = 'tight')
plt.show(block = False)

labels_prob = np.loadtxt('labels_prob.txt')

labels_prob = np.expand_dims(labels_prob, axis = 0)
fig = plt.figure()
plt.imshow(labels_prob, origin = 'upper', cmap = 'BuPu', interpolation = 'none')
for (j,i),label in np.ndenumerate(labels_prob):
    if label < 0.25:
        plt.annotate('%1.2f' % label, (i - 0.25, j + 0.1), size = 10, color = 'black')
    else:
        plt.annotate('%1.2f' % label, (i - 0.25, j + 0.1), size = 10, color = 'white')

plt.xticks(range(0, 9))
plt.yticks([])
plt.savefig('labels_prob.pdf', bbox_inches = 'tight')
plt.savefig('labels_prob.eps', bbox_inches = 'tight')
plt.show(block = False)

import numpy as np
import pylab as plt


nsim = ['753', '800', '900', '1000', '2000', '3000', '4000', '5000']


xnsim = []
a = []
astd = []
b = []
bstd = []

for item in nsim:
    op1 = open('nsim_' + item + '/num_res_nsim_' + item + '.dat', 'r')
    lin1 = op1.readlines()
    op1.close()

    data1 = [elem.split() for elem in lin1]

    a.append(float(data1[0][1]))
    astd.append(float(data1[1][1]))
    b.append(float(data1[4][1]))
    bstd.append(float(data1[5][1]))
    xnsim.append(int(item))

op2 = open('correct_covariance/num_res_corr_cov.dat', 'r')
lin2 = op2.readlines()
op2.close()

data2 = [elem.split() for elem in lin2]

a_corr = float(data2[0][1])
astd_corr = float(data2[1][1])

b_corr = float(data2[4][1])
bstd_corr = float(data2[5][1])

fig = plt.figure()
plt.subplot(1,2,1)
plt.scatter(xnsim, a, color='blue', marker='o', label='a')
plt.scatter(xnsim, b, color='green', marker='^', label='b')
plt.plot([xnsim[0], xnsim[-1]], [a_corr, a_corr], color='blue', ls=':', label='a - corr cov')
plt.plot([xnsim[0], xnsim[-1]], [b_corr, b_corr], color='green', ls='-', label='b - corr cov')
plt.plot([xnsim[0], xnsim[-1]], [1,1], color='red', lw=0.8, label='a - fiducial')
plt.plot([xnsim[0], xnsim[-1]], [0,0], color='red', lw=0.8, ls='-.', label='b - fiducial')
plt.legend(fontsize=10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.xlabel('n_S', fontsize=10)
plt.ylabel('mean of intercept, slope', fontsize=10)

plt.subplot(1,2,2)
plt.scatter(xnsim, astd, color='blue', marker='o', label='a_std')
plt.scatter(xnsim, bstd, color='green', marker='^', label='b_std')
plt.legend(fontsize=10)
plt.xlabel('n_S', fontsize=10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=4)
plt.ylabel('std of intercept, slope', fontsize=10)

fig.subplots_adjust(left=0.075, right=0.99, bottom=0.09, top=0.975, hspace=0.15, wspace=0.25)
plt.savefig('line_mean_std_ABC.pdf')
    

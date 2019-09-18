import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mv


x = np.arange(-100, 100, 1)
y = x

sigma = 5

y1 = mv.rvs(mean=x, cov=np.diag([5] * x.shape[0]))
y2 = np.random.normal(loc=x, scale=sigma)

cov1 = []
for j in range(x.shape[0]):
    line = []
    for i in range(x.shape[0]):
        if i == j:
            line.append(sigma)
        elif j - 1 <= i <= j + 1:
            line.append(sigma)

        else:
            line.append(0)
    cov1.append(line)

cov1 = np.array(cov1)

y3 = mv.rvs(mean=x, cov=cov1)

cov2 = []
for j in range(x.shape[0]):
    line = []
    for i in range(x.shape[0]):
        if i == j:
            line.append(sigma)
        elif j - 10 <= i <= j + 10:
            line.append(sigma)

        else:
            line.append(0)
    cov2.append(line)

cov2 = np.array(cov2)

y4 = mv.rvs(mean=x, cov=cov2)

cov3 = np.array([[sigma for j in range(x.shape[0])] for i in range(x.shape[0])])
y5 = mv.rvs(mean=x, cov=cov3)


plt.figure()
plt.plot(x, y, color='red', lw=1.5, label='fiducial')
#plt.scatter(x, y1-x, color='blue', s=5, label='diag - mvn', marker='o')
plt.scatter(x, y2, color='orange', s=5, label='diag - rand', marker='*')
#plt.scatter(x, y3-x, color='green', s=5, label='3col', marker='s')
#plt.scatter(x, y4-x, color='purple', s=5, label='21col', marker='^')
plt.scatter(x, y5, color='black', s=5, label='all', marker='+')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

%import matplotlib
from matplotlib import pyplot as plt
import numpy as np

a1 = np.load('acc_var1.npy')
a2 = np.load('acc_var2.npy')
a3 = np.load('acc_var3.npy')
a4 = np.load('acc_var4.npy')
a5 = np.load('acc_var5.npy')

r1 = np.load('rand_acc1.npy')
r2 = np.load('rand_acc2.npy')
r3 = np.load('rand_acc3.npy')
r4 = np.load('rand_acc4.npy')
r5 = np.load('rand_acc5.npy')

a = np.mean([a1, a2, a3, a4, a5], axis=0)
r = np.mean([r1, r2, r3, r4, r5], axis=0)

plt.axis([0, 1000, 0.8, 1])
plt.yticks(np.array(range(11))*0.02 + 0.8)
plt.xticks(np.array(range(10))*100)
plt.plot(np.array(range(99))*10, a, label='var-ratio')
plt.plot(np.array(range(99))*10, r, 'g', label='random')
plt.grid()
plt.title('Give more weight to the newly acquired points')
plt.legend(loc=0)
plt.show()
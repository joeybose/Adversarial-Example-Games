import numpy as np
import ipdb
from scipy.stats import wilcoxon, ttest_rel

# MNIST
mi_attack = [90.000000, 87.575768, 81.515160, 90.909088, 84.848480, 88.787872,
             89.090904]
di_attack = [90.606056, 90.000000, 85.454552, 91.818176, 88.484856, 89.696968,
             0.606071]
tid_attack = [90.000000, 83.939400, 84.545456, 86.666664, 83.333336, 83.333336,
              86.060608]
aeg_mnist = [88.095, 91.071, 88.690, 89.881, 85.714, 91.071, 91.667]

w_mi, p_mi = wilcoxon(mi_attack, aeg_mnist, alternative='less', zero_method='zsplit')
print("MNIST-- MI-Attack vs. AEG: W:  %f , P: %f" %(w_mi, p_mi))

w_di, p_di = wilcoxon(di_attack, aeg_mnist, alternative='less', zero_method='zsplit')
print("MNIST-- DI-Attack vs. AEG: W:  %f , P: %f" %(w_di, p_di))

w_tid, p_tid = wilcoxon(tid_attack, aeg_mnist, alternative='less', zero_method='zsplit')
print("MNIST-- TID-Attack vs. AEG: W:  %f , P: %f" %(w_tid, p_tid))

# CIFAR
c_mi_attack = [48.176666, 60.848335, 57.434998, 49.005005, 64.980003,
               60.071667]
c_di_attack = [83.571671, 85.126671, 84.953331, 79.344994, 83.279999, 87.748329]
c_tid_attack = [8.991667, 8.716668, 9.298335, 9.150001, 9.185000, 9.225000]
c_sgm_attack = [55.240002, 63.230000, 58.849995, 49.519997, 66.979996,
                68.919998]
aeg_cifar = [87.51, 87.353, 87.197, 86.761, 86.683, 86.529]

c_w_mi, c_p_mi = wilcoxon(c_mi_attack, aeg_cifar, alternative='less', zero_method='zsplit')
print("CIFAR-- MI-Attack vs. AEG: W:  %f , P: %f" %(c_w_mi, c_p_mi))

c_w_di, c_p_di = wilcoxon(c_di_attack, aeg_cifar, alternative='less', zero_method='zsplit')
print("CIFAR-- DI-Attack vs. AEG: W:  %f , P: %f" %(c_w_di, c_p_di))

c_w_tid, c_p_tid = wilcoxon(c_tid_attack, aeg_cifar, alternative='less', zero_method='zsplit')
print("CIFAR-- TID-Attack vs. AEG: W:  %f , P: %f" %(c_w_tid, c_p_tid))

c_w_sgm, c_p_sgm = wilcoxon(c_sgm_attack, aeg_cifar, alternative='less', zero_method='zsplit')
print("CIFAR-- SGM-Attack vs. AEG: W:  %f , P: %f" %(c_w_sgm, c_p_sgm))

# T Test- MNIST
w_mi, p_mi = ttest_rel(mi_attack, aeg_mnist)
print("T-Test MNIST-- MI-Attack vs. AEG: W:  %f , P: %f" %(w_mi, p_mi))

w_di, p_di = ttest_rel(di_attack, aeg_mnist)
print("T-Test MNIST-- DI-Attack vs. AEG: W:  %f , P: %f" %(w_di, p_di))

w_tid, p_tid = ttest_rel(tid_attack, aeg_mnist)
print("T-Test MNIST-- TID-Attack vs. AEG: W:  %f , P: %f" %(w_tid, p_tid))

# T Test- CIFAR
c_w_mi, c_p_mi = ttest_rel(c_mi_attack, aeg_cifar)
print("T-Test CIFAR-- MI-Attack vs. AEG: W:  %f , P: %f" %(c_w_mi, c_p_mi))

c_w_di, c_p_di = ttest_rel(c_di_attack, aeg_cifar)
print("T-Test CIFAR-- DI-Attack vs. AEG: W:  %f , P: %f" %(c_w_di, c_p_di))

c_w_tid, c_p_tid = ttest_rel(c_tid_attack, aeg_cifar)
print("T-Test CIFAR-- TID-Attack vs. AEG: W:  %f , P: %f" %(c_w_tid, c_p_tid))

c_w_sgm, c_p_sgm = ttest_rel(c_sgm_attack, aeg_cifar)
print("T-Test CIFAR-- SGM-Attack vs. AEG: W:  %f , P: %f" %(c_w_sgm, c_p_sgm))

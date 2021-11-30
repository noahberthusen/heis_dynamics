import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib
import numpy as np
import os

data = []
ii = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,18]
for i in ii:
    if os.path.isfile(f'../results/entanglement_results/8_pvstime/results_time_8_{i}.txt'):
        arr = np.loadtxt(f'../results/entanglement_results/8_pvstime/results_time_8_{i}.txt', delimiter=',')
        data.append(arr)

l_star = []
epsilon = 5e-3
for i in range(0, 24): # loop over all times
    for d in range(len(data)): # loop over all l's
        if data[d][i][0] < epsilon:
            l_star.append(ii[d])
            break

# ---------------------------------------------------------------------------------------------

fig, ax = plt.subplots(2, 1, figsize=(6,7))
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman'] + plt.rcParams['font.serif']

for i in range(2):
    ax[i].tick_params(axis='x', labelsize=12)
    ax[i].tick_params(axis='y', labelsize=12)
    ax[i].xaxis.set_tick_params(width=1.5)
    ax[i].yaxis.set_tick_params(width=1.5)

ax[0].set_ylabel(r'$1-\mathcal{F}(t,  \hat{\vartheta}^{(\ell)})$', fontsize=16)
ax[1].set_ylabel(r'$\ell^*$', fontsize=18)

for axis in ['top','bottom','left','right']:
    ax[0].spines[axis].set_linewidth(1.3)
    ax[1].spines[axis].set_linewidth(1.3)

plt.text(0, 0.87, "(a)", fontsize=16, transform=plt.gcf().transFigure, fontfamily='sans-serif')
plt.text(0, 0.46, "(b)", fontsize=16, transform=plt.gcf().transFigure, fontfamily='sans-serif')

ax[1].set_xlabel('$Jt$', fontsize=16)
ax[0].set_yscale('log')

color = matplotlib.cm.viridis
ax[0].set_prop_cycle('color', [color(i) for i in np.linspace(0, 0.8, 5)])

ells = [2,6,10,14,18] # <--- choose \ell's to show here 

for d in [ii.index(l) for l in ells]: 
    ax[0].plot(data[d][:,1], data[d][:,0], '-', label=f'$\ell={ii[d]}$', markersize=3, linewidth=2)
ax[0].axhline(5e-3, color='k', linestyle='--', linewidth=2)
ax[0].legend(fontsize=12)

def line(x, a, b):
    return a * x + b
popt, pcov = curve_fit(line, [i*0.1 for i in range(0, 24)], l_star)
xline = np.linspace(0,2.4,100)
ax[1].plot(xline, line(xline, popt[0], popt[1]), c='k', linewidth=2, linestyle='--', alpha=0.5)
ax[1].scatter([i*0.1 for i in range(0, 24)], l_star, c='k')
ax[1].set_yscale('linear')

plt.savefig('../figures/final/early_time_results.png', dpi=600, transparent=False, bbox_inches='tight')
# plt.show()
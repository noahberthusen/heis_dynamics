import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib
import numpy as np
import os


ee = np.array([[0, 0], [1/10, 0.0348518], [1/5, 0.110803], [3/10, 0.210377], [2/5, 
  0.323438], [1/2, 0.442919], [3/5, 0.563954], [7/10, 0.683504], [4/5,
   0.800025], [9/10, 0.91309], [1, 1.02295], [11/10, 1.13012], [6/5, 
  1.23492], [13/10, 1.33727], [7/5, 1.43647], [3/2, 1.53121], [8/5, 
  1.61969], [17/10, 1.69982], [9/5, 1.76947], [19/10, 1.82678], [2, 
  1.87037], [21/10, 1.8996], [11/5, 1.91467], [23/10, 1.9167], [12/5, 
  1.90759], [5/2, 1.88989], [13/5, 1.86651], [27/10, 1.8407], [14/5, 
  1.81564], [29/10, 1.79366], [3, 1.77568], [31/10, 1.76137], [16/5, 
  1.74972], [33/10, 1.73947], [17/5, 1.72954], [7/2, 1.71945], [18/5, 
  1.70947], [37/10, 1.70061], [19/5, 1.6944], [39/10, 1.69258], [4, 
  1.69629], [41/10, 1.70477], [21/5, 1.7162], [43/10, 1.72896], [22/5,
   1.74174], [9/2, 1.75323], [23/5, 1.76175], [47/10, 1.7658], [24/5, 
  1.76475], [49/10, 1.7576], [5, 1.74318], [51/10, 1.72164], [26/5, 
  1.69409], [53/10, 1.6618], [27/5, 1.62546], [11/2, 1.58517], [28/5, 
  1.54101], [57/10, 1.49361], [29/5, 1.44434], [59/10, 1.39508], [6, 
  1.34807], [61/10, 1.30589], [31/5, 1.27142], [63/10, 1.24772], [32/
  5, 1.23742], [13/2, 1.24185], [33/5, 1.26059], [67/10, 
  1.29151], [34/5, 1.33194], [69/10, 1.37899], [7, 1.42969], [71/10, 
  1.48116], [36/5, 1.53143], [73/10, 1.57928], [37/5, 1.62375], [15/2,
   1.66406], [38/5, 1.69976], [77/10, 1.73061], [39/5, 1.75627], [79/
  10, 1.77606], [8, 1.78903]])

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

inset_dim2 = [0.71, 0.13, 0.17, 0.15]

ax3 = plt.axes(inset_dim2)
ax3.patch.set_alpha(0)
ax3.plot(ee[:,0], ee[:,1], linestyle='-', c='k')
ax3.set_xlabel('$Jt$', fontsize=11)
ax3.set_ylabel(r'$S(\rho_A)$', fontsize=11)
ax3.xaxis.set_label_position('top') 
ax3.xaxis.set_ticks_position('top') 

for axis in ['top','bottom','left','right']:
    ax[0].spines[axis].set_linewidth(1.3)
    ax[1].spines[axis].set_linewidth(1.3)
    ax3.spines[axis].set_linewidth(1)

plt.savefig('../figures/final/early_time_results.png', dpi=600, transparent=False, bbox_inches='tight')
# plt.show()
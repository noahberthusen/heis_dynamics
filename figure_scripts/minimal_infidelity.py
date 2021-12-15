import matplotlib.ticker as mticker
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

Ls = [[1.7996267587250258e-10, 5.878299180750446e-11, 4.295401812015598e-11],
     [0.0011696504177758317, 0.0001847764634799076, 7.770443912269443e-07, 8.577016519240032e-10],
     [0.4639926762403798, 0.11391314946947262, 0.0052318151198888965, 7.07515616982235e-09, 4.817845151716682e-09],   
     [0.6363176386045497, 0.3392298484219568, 0.17048011852949127, 0.052668122694601784, 0.00967727126882913, 6.0256055398317797e-05, 3.719534741541963e-08],
     [0.6416560976244899, 0.5086738165494462, 0.36342221837249766, 0.21626469635978224, 0.12237679637738229, 0.045167593212329396, 0.019818305324384538, 0.004553513550590642, 0.0002063399825817802, 2.2342880081627215e-07],
     [0.7590718540880695, 0.5719211366047965, 0.44510673104760407, 0.3241760396769466, 0.2550839283040911, 0.20338644928503405, 0.16657267588874713, 0.13303684756303158, 0.09179578796168021, 0.06102443877810375, 0.032180786182541075, 0.021368967806325422, 0.010754411919488305, 0.0043950994366458995, 0.0012846150810468914, 0.00020951238811922623, 1.1106172605279685e-05],
     [0.8487796466464832, 0.7246713564910872, 0.6212977022476487, 0.5601619345396789, 0.4708760688783845, 0.41385549557756507, 0.3608916901805358, 0.30257921928460585, 0.24985423332471882, 0.19539132415233107, 0.1679920211863423, 0.13669538443585097, 0.11212563368131197, 0.08873980822580929, 0.06822397613094205, 0.05092762634877268, 0.03898776643879624, 0.02921738040071047, 0.020098550656726288, 0.014498611773404769, 0.0071355138558909514, 0.0043125361783976635, 0.002548221151697283, 0.0011333996251954304, 0.0004074573587542485],
    [0.8743803794511876, 0.7602827036510854, 0.6283331545866673, 0.5628693380619201, 0.4921650094542621, 0.4256666141075628, 0.38455154797254093, 0.33474153611510793, 0.30555276126916875, 0.2997335321936316, 0.28403853926060596, 0.32331078335113633, 0.2605064704572483, 0.2701202553798495, 0.25988398779544275, 0.23296151808521248, 0.21691028941228307, 0.18272598198029488, 0.18016026093971216, 0.14507519284391246, 0.14432891969153894, 0.13215513174156102, 0.11645455273223815, 0.10780069121860303, 0.08184172387280209, 0.07124292315053182, 0.06695249692045613, 0.06004606181960437, 0.05447963474429951, 0.05066997157910613, 0.035307219169577686, 0.03405049668035065, 0.024741642898890692, 0.021846105971423002, 0.017336086230235193, 0.012528126860723132, 0.01008166471798333, 0.008636011985746816, 0.007922392087377883, 0.005485597676916254]
 ]

LsHz = [[5.588469909056926e-10, 2.0827159996628098e-10],
     [0.001168806471793331, 6.137607163125836e-07, 2.9467021409601557e-08],
     [0.221483226411549, 3.516649276943173e-05, 3.5626192662618907e-09, 2.3491798506469763e-09],
     [0.5235831350848252, 0.1083608801605783, 0.005335858738842259, 1.6297208194071544e-08],
     [0.6040549826328518, 0.23093713565766655, 0.05580251843277639, 0.006179206353788902, 1.9743014617601418e-06, 6.762318172093273e-08],
     [0.6176635173647363, 0.4390706383021266, 0.2408873676967919, 0.14031169478591124, 0.05890306619134159, 0.02307252601390696, 0.005092728379774802, 0.0004371954153208346, 2.697683505470394e-07],
     [0.785334008250023, 0.613777408810309, 0.4576167020008981, 0.3288543148417202, 0.22143088145446158, 0.14860190229834255, 0.08759629838864513, 0.05802319045287724, 0.031662276466621564, 0.01587701353584159, 0.007213445838580079, 0.0024537179100346806, 0.0002624472754672036, 7.774376831970464e-06],
     [0.8610560425939218, 0.7231270541629369, 0.6285820336580716, 0.5370094711445761, 0.4586670741282329, 0.3807929396950779, 0.3174909773888193, 0.2513449067963391, 0.2195720123424823, 0.1836289000938064, 0.13420489033620941, 0.10721758114982793, 0.08139500801780553, 0.06053323515587661, 0.04577449938456679, 0.031191148885733266, 0.02140432456271879, 0.015623953576980876, 0.008632197587004065, 0.006153346788576708, 0.003555021475440623, 0.0012671810611337708, 0.00041213024535813325]
]


fig, ax = plt.subplots(2, 1, figsize=(6.5,6.5))
fig.tight_layout(pad=2)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman'] + plt.rcParams['font.serif']

# ax0 = fig.add_subplot(111, frame_on=False)   # creating a single axes
# ax0.set_xticks([])
# ax0.set_yticks([])
# ax0.set_ylabel(r'$1- \mathcal{F}(t_f,  \vartheta^{(\ell)})$', fontsize=16, labelpad=45)

cmap = matplotlib.cm.viridis

# ax_top = ax[0].twiny()
# ax_top2 = ax[1].twiny()

# for i in range(len(LsHz)):
    # ax_top2.plot(np.array([2*j*(i+3) for j in range(0, len(LsHz[-1]))]), LsHz[-1], alpha=0)
# ax_top2.set_xlabel('Number of parameters', size=14)
# ax_top2.set_xscale('log')
# ax_top2.xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.0f}'))

# for i in range(len(Ls)):
    # ax_top.plot(np.array([j*(i+3) for j in range(0, len(Ls[-1]))]), Ls[-1], alpha=0)
# ax_top.set_xlabel('Number of parameters', size=14)
# ax_top.set_xscale('log')
# ax_top.xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.0f}'))

for i in range(2):
    ax[i].tick_params(axis='x', labelsize=12)
    ax[i].tick_params(axis='y', labelsize=12)
    ax[i].xaxis.set_tick_params(width=1.5)
    ax[i].yaxis.set_tick_params(width=1.5)
    ax[i].set_xscale('log')
    ax[i].set_ylabel(r'$1- \mathcal{F}(t_f,  \hat{\vartheta}^{(\ell)})$', fontsize=14)
    ax[i].axhline(5e-3, color='k', linestyle='--')
    ax[i].set_yscale('log')
    ax[i].set_prop_cycle('color', [cmap(i) for i in np.linspace(0, 0.95, len(Ls))])
    # ax[i].tick_params(axis='x', which='minor', bottom=False)
    ax[i].grid(True, alpha=0.5)
    ax[i].xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.0f}'))


# fig.supylabel(r'$1- \mathcal{F}(t_f,  \vartheta^{(\ell)})$', fontsize=14)

for i in range(len(Ls)):
    ax[0].plot(np.array([j for j in range(1, len(Ls[i])+1)]), Ls[i], linewidth=2)
    ax[1].plot(np.array([j for j in range(1, len(LsHz[i])+1)]), LsHz[i], linewidth=2) 

bounds = [3,4,5,6,7,8,9,10]

norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N-16)
cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax[0])
cbar.ax.set_title('M', fontsize=14)
cbar2 = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax[1])
cbar2.remove()

ax[1].set_xlabel('$\ell$', fontsize=18)

Lstar = np.array([1, 1, 3, 6, 8, 14, 22, 41]) # calculated by hand
L = np.array([3, 4, 5, 6, 7, 8, 9, 10])
Lpred = [11, 12]
Lstarpred = [76, 120]
inset_dim1 = [0.6, 0.58, 0.15, 0.15]

ax3 = plt.axes(inset_dim1)
ax3.patch.set_alpha(0)
ax3.plot([(0,0), (1,1)], linestyle='--', c='#d1d1d1', linewidth=0.75)
ax3.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax3.tick_params(axis='y', which='both', left=False, top=False, labelleft=False)

ax2 = plt.axes(inset_dim1)
ax2.patch.set_alpha(0)
ax2.set_xlabel('M', fontsize=11)
ax2.set_ylabel('$\ell^*$', fontsize=14, rotation=0)
ax2.xaxis.set_label_position('top') 
ax2.xaxis.set_ticks_position('top')
ax2.tick_params(axis='x', labelsize=10)
ax2.tick_params(axis='y', labelsize=10)
ax2.scatter(L, Lstar, s=7, c='k')
ax2.plot(Lpred, Lstarpred, markersize=7, marker='x', markeredgewidth=1, c='k', linestyle='None', zorder=0)
ax2.set_yscale('log')

LHz = np.array([3, 4, 5, 6, 7, 8, 9, 10])
LstarHz = np.array([1, 1, 2, 4, 5, 8, 12, 21])
LHzpred = [11, 12]
LstarHzpred = [36, 61]
inset_dim2 = [0.6, 0.10, 0.15, 0.15]

ax3 = plt.axes(inset_dim2)
ax3.patch.set_alpha(0)
ax3.plot([(0,0), (1,1)], linestyle='--', c='#d1d1d1', linewidth=0.75)
ax3.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax3.tick_params(axis='y', which='both', left=False, top=False, labelleft=False)

ax4 = plt.axes(inset_dim2)
ax4.patch.set_alpha(0)
ax4.set_xlabel('M', fontsize=11)
ax4.set_ylabel('$\ell^*$', fontsize=14, rotation=0)
ax4.tick_params(axis='x', labelsize=10)
ax4.tick_params(axis='y', labelsize=10)
ax4.xaxis.set_label_position('top') 
ax4.xaxis.set_ticks_position('top') 
ax4.scatter(LHz, LstarHz, s=7, c='k')
ax4.plot(LHzpred, LstarHzpred, markersize=7, marker='x', markeredgewidth=1, c='k', linestyle='None')
ax4.set_yscale('log')

plt.text(-0.02, 0.94, "(a)", fontsize=16, transform=plt.gcf().transFigure, fontfamily='sans-serif')
plt.text(-0.02, 0.46, "(b)", fontsize=16, transform=plt.gcf().transFigure, fontfamily='sans-serif')

for axis in ['top','bottom','left','right']:
    ax[0].spines[axis].set_linewidth(1.3)
    ax[1].spines[axis].set_linewidth(1.3)
    ax2.spines[axis].set_linewidth(1)
    ax4.spines[axis].set_linewidth(1)

plt.savefig('../figures/final/p_vs_fidelity.png', dpi=600, transparent=False, bbox_inches='tight')
# plt.show()

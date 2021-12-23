import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import expm_multiply
from scipy.linalg import expm
import matplotlib.pyplot as plt
import os
import matplotlib

def FlipFlop(n, i, j):
    v = list(format(n, '0{}b'.format(L)))
    if (v[i] != '0' and v[j] != '1'):
        v[i] = '0'
        v[j] = '1'
        return int(''.join(v), 2)
    else:
        return -1
        
def Raise(n, i):
    v = list(format(n, '0{}b'.format(L)))
    # checking mod here, unsure why since accesses aren't modded
    if (v[i] != '1'):
        v[i] = '1'
        return int(''.join(v), 2)
    else:
        return -1

L = 6

Sz = []
for i in range(L):
    sprs = csc_matrix((2**L, 2**L), dtype=np.int8)
    for j in range(2**L):
        sprs[j, j] = 1-2*int(format(j, '0{}b'.format(L))[i])
    Sz.append(sprs)
SzTot = sum(Sz)

Sp = []
for i in range(L):
    sprs = csc_matrix((2**L, 2**L), dtype=np.int8)
    for j in range(2**L):
        h = Raise(j, i)
        if (h != -1):
            sprs[h, j] = 1
    Sp.append(sprs)

Heis = []
for i in range(L):
    _ = []
    for k in range(L):
        sprs = csc_matrix((2**L, 2**L), dtype=np.int8)
        for j in range(2**L):
            h = FlipFlop(j, i, k)
            v = lambda i: 1-2*int(format(j, '0{}b'.format(L))[i])
            if (h != -1):
                sprs[j, h] = 2
                sprs[h, j] = 2
            sprs[j, j] = v(i) * v(k)
        _.append(sprs)    
    Heis.append(_)

Sm = [Sp[i].T for i in range(L)]
Sx = [Sp[i]+Sm[i] for i in range(L)]
Sy = [-1j*Sz[i] @ Sx[i] for i in range(L)]
SxTot = sum(Sx)
SyTot = sum(Sy)

H = sum([Heis[i][(i+1)%L] for i in range(L-1)]) / 4

tf = 35
dt = tf / 300
Nt = int(tf / dt)
c = [str((1 + (-1)**(i+1)) // 2) for i in range(L)]
UnitVector = lambda c: np.eye(2**L)[c]
init = UnitVector(int(''.join(c), 2))

def ExactTimeEvolution(dt, nt, init):
    revos = [np.zeros(2**L) for i in range(nt+1)]
    revos[0] = init
    for i in range(nt):
        # scipy.sparse.linalg.expm_multiply
        revos[i+1] = expm_multiply(-1j * H * dt, revos[i])
    return revos

revos = ExactTimeEvolution(dt, Nt, init)

def TrotterEvolve(dt, nt, init):
    if (L % 2 == 0):
        UOdd = expm(-1j * dt * sum([Heis[i][(i+1)%L] for i in range(0, L-1, 2)]) / 4) # 0 indexing, this is actually even indices
        UEven = expm(-1j * dt * sum([Heis[i][(i+1)%L] for i in range(1, L-1, 2)]) / 4) # range(1, L, 2) for periodic bdy conditions
        UTrotter = UOdd @ UEven
        # UZ = expm(-1j * dt * sum([diags(Heis[i][(i+2)%L].diagonal()) for i in range(L)]) / 2)
        # UTrotter = UEven @ UOdd @ UZ
    else:
        UOdd = expm(-1j * dt * sum([Heis[i][(i+1)%L] for i in range(0, L-1, 2)]) / 4)
        UEven = expm(-1j * dt * sum([Heis[i][(i+1)%L] for i in range(1, L, 2)]) / 4)
        # UBdy = expm(-1j * dt * Heis[L-1][0] / 4)
        UTrotter = UOdd @ UEven # UBdy @ UOdd @ UEven
    psi_trot = init
    for i in range(nt):
        psi_trot = UTrotter @ psi_trot
    return psi_trot

ntrot = 7
dtrot = 0.4
tf_vtc = 35
tf = 28
op = Sz[0]

Szt = []
for i in range(len(revos)):
    Szt.append(np.conj(revos[i]) @ op @ revos[i] / 2)

TrotterFixStepList = [init]
ts = [i*ntrot*dtrot for i in range(int(np.ceil(tf / (ntrot * dtrot)))+1)]

for i in range(int(np.ceil(tf / (ntrot * dtrot)))):
    TrotterFixStepList.append(TrotterEvolve(dtrot, ntrot, TrotterFixStepList[i]))
TrotterFixStepSz = [np.conj(TrotterFixStepList[i]) @ op @ TrotterFixStepList[i] / 2 for i in range(len(TrotterFixStepList))]

BadTrotterFixStepList = [init]
for i in range(Nt):
    BadTrotterFixStepList.append(TrotterEvolve((i+1)*dt/(ntrot*3), ntrot*3, init))
BadTrotterFixStepSz = [np.conj(BadTrotterFixStepList[i]) @ op @ BadTrotterFixStepList[i] / 2 for i in range(len(BadTrotterFixStepList))]


ExFidelity = [abs(np.conj(revos[i]) @ revos[i])**2 for i in range(len(revos))]
BadTrotterFidelity = [abs(np.conj(revos[i]) @ BadTrotterFixStepList[i])**2 for i in range(len(BadTrotterFixStepList))]

revos_ = [expm(-1j * H * t) @ init for t in ts]

TrotterFidelity = [abs(np.conj(revos_[i]) @ TrotterFixStepList[i])**2 for i in range(len(TrotterFixStepList))]

# -----------------------------------------------------------------------------------------

ts = [i*ntrot*dtrot for i in range(int(np.ceil(tf / (ntrot * dtrot)))+1)]
fig, ax = plt.subplots(2, 1, figsize=(6.5,6.5))
fig.tight_layout(pad=2)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman'] + plt.rcParams['font.serif']

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    if n == -1:
        n = cmap.N
    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
         'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
         cmap(np.linspace(minval, maxval, n)))
    return new_cmap

fidelities = []
VTDStepLists = []
ii = np.array([2**i for i in range(14, 17)])
for i in ii:
    if os.path.isfile(f'../results/VTD_results/shots_variation_loschmidt_6/VTD_results_{tf_vtc}_{L}_{ntrot}_{dtrot}_{i}.csv'):
        VTDStepList = pd.read_csv(f'../results/VTD_results/shots_variation_loschmidt_6/VTD_results_{tf_vtc}_{L}_{ntrot}_{dtrot}_{i}.csv', index_col=0)
        VTDStepList = VTDStepList.applymap(lambda x: complex(x))
        VTDStepLists.append(VTDStepList)
        revos_ = [expm(-1j * H * t) @ init for t in ts]

        VTDFidelity = [abs(np.conj(revos_[i]) @ np.array(VTDStepList.iloc[i]))**2 for i in range(len(VTDStepList))]
        fidelities.append(VTDFidelity)

BestCompression = [init]
for i in range(len(VTDStepLists[-1])):
    BestCompression.append(TrotterEvolve(dtrot, ntrot, VTDStepLists[-1].iloc[i]))
BestCompressionFidelity = [abs(np.conj(revos_[i]) @ np.array(BestCompression[i]))**2 for i in range(len(BestCompression[1:]))]
VTDBestCompressionFidelity = [abs(np.conj(BestCompression[i]) @ np.array(VTDStepLists[-1].iloc[i]))**2 for i in range(len(VTDStepLists[-1]))]

color = matplotlib.cm.viridis
color = truncate_colormap(color, 0.0, 0.8)
norm = matplotlib.colors.Normalize(ii.min(), ii.max())
cmap = matplotlib.cm.ScalarMappable(norm=norm, cmap=color)

bounds = [1,2,3,4,5,6,7,8]
norm = matplotlib.colors.BoundaryNorm(bounds, color.N-20)
cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=color), ax=ax[0])
cbar2 = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=color), ax=ax[1])
cbar2.remove()

cbar.set_ticks(bounds)
cbar.set_ticklabels(['$2^{14}$','$2^{15}$','$2^{16}$'])
cbar.ax.set_title('Samples', fontsize=11)
cbar.set_ticks([1,4,8])
ax[0].set_prop_cycle('color', [color(i) for i in np.linspace(0, 1, len(fidelities))])

ax[0].plot(ts, TrotterFidelity, c='#ff7f0e', label='Ideal VTC', linewidth=2)
ax[0].plot([i*dt for i in range(len(revos))], BadTrotterFidelity, label='Trotter $(3\ell)$', c='#CDCDCD', zorder=0, linewidth=2)
ax[0].axvspan(14.5, 29, facecolor='#9cce9c', alpha=0.2)
ax[0].margins(x=0.05, y=0)
ax[0].set_xlim(0,28.3)

ax[1].plot([i*dtrot*ntrot for i in range(len(fidelities[-1]))], fidelities[-1], linestyle='--', marker='.', c='g', linewidth=2, label="VTC")
ax[1].plot(ts, TrotterFidelity, c='#ff7f0e', linewidth=2)
ax[1].plot([i*dt for i in range(len(revos))], BadTrotterFidelity, c='#CDCDCD', zorder=0, linewidth=2)
ax[1].axvspan(14.5, 29, facecolor='#9cce9c', alpha=0.2)
ax[1].margins(x=0.05, y=0)
ax[1].set_xlim(0,28.3)
ax[1].scatter([i*dtrot*ntrot for i in range(len(BestCompressionFidelity))], BestCompressionFidelity, label="Best Compression", marker='x', c='k')
ax[1].scatter([i*dtrot*ntrot for i in range(len(VTDBestCompressionFidelity))], VTDBestCompressionFidelity, label="VTC Overlap", marker='x', c='r')
ax[1].legend(loc='lower left', fontsize=12)

for i in range(len(fidelities)):
    ax[0].plot([i*dtrot*ntrot for i in range(len(fidelities[i]))], fidelities[i], linestyle='--', marker='.', linewidth=2)

for axis in ['top','bottom','left','right']:
    ax[0].spines[axis].set_linewidth(1.3)
    ax[1].spines[axis].set_linewidth(1.3)

ax[0].set_ylabel(r'$\mathcal{F}(t,  \hat{\vartheta}_t)$', fontsize=16)
ax[1].set_ylabel(r'$\mathcal{F}(t,  \hat{\vartheta}_t)$', fontsize=16)

ax[1].set_xlabel('$Jt$', fontsize=16)
ax[0].tick_params(axis='x', labelsize=12)
ax[0].tick_params(axis='y', labelsize=12)
ax[0].legend(loc='lower left', fontsize=12)

plt.text(-0.02, 0.94, "(a)", fontsize=16, transform=plt.gcf().transFigure, fontfamily='sans-serif')
plt.text(-0.02, 0.46, "(b)", fontsize=16, transform=plt.gcf().transFigure, fontfamily='sans-serif')

plt.savefig('../figures/final/shots_variation_L6_alt', dpi=600, transparent=False, bbox_inches='tight')
# plt.show()
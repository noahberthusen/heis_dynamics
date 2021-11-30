import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import eigs, expm_multiply
from scipy.linalg import expm
import matplotlib.pyplot as plt
import os

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

L = 3

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

tf = 20
dt = tf / 300
Nt = int(tf / dt)
c = ['1','1','0'] 
UnitVector = lambda c: np.eye(2**L)[c]
init = UnitVector(int(''.join(c), 2))

def ExactTimeEvolution(dt, nt, init):
    revos = [np.zeros(2**L) for i in range(nt+1)]
    revos[0] = init
    for i in range(nt):
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


ntrot = 2
dtrot = 1.0
tf = 20
shots = 2**13
op = Sz[0]

ts = [i*ntrot*dtrot for i in range(int(np.ceil(tf / (ntrot * dtrot)))+1)]

fidelities = []
for i in range(50):
    if os.path.isfile(f'../results/VTD_results/error_bars/VTD_results_{tf}_{L}_{ntrot}_{dtrot}_{shots}_{i}.csv'):
        VTDStepList = pd.read_csv(f'../results/VTD_results/error_bars/VTD_results_{tf}_{L}_{ntrot}_{dtrot}_{shots}_{i}.csv', index_col=0)
        VTDStepList = VTDStepList.applymap(lambda x: complex(x))

        revos_ = [expm(-1j * H * t) @ init for t in ts]

        VTDFidelity = [abs(np.conj(revos_[i]) @ np.array(VTDStepList.iloc[i]))**2 for i in range(len(VTDStepList))]
        fidelities.append(VTDFidelity)

err = np.std(np.array(fidelities), axis=0)
mean = np.mean(np.array(fidelities), axis=0)

VTDStepList = pd.read_csv(f'../results/VTD_results/real_device/final_results/VTD_results_{tf}_{L}_{ntrot}_{dtrot}_{shots}.csv', index_col=0)
VTDStepList = VTDStepList.applymap(lambda x: complex(x))
VTDSz = [np.array(np.conj(VTDStepList.iloc[i])) @ op @ np.array(VTDStepList.iloc[i]) / 2 for i in range(len(VTDStepList))]

TrotterFixStepList = [init]
ts = [i*ntrot*dtrot for i in range(int(np.ceil(tf / (ntrot * dtrot)))+1)]

for i in range(int(np.ceil(tf / (ntrot * dtrot)))):
    TrotterFixStepList.append(TrotterEvolve(dtrot, ntrot, TrotterFixStepList[i]))
TrotterFixStepSz = [np.conj(TrotterFixStepList[i]) @ op @ TrotterFixStepList[i] / 2 for i in range(len(TrotterFixStepList))]

BadTrotterFixStepList = [init]
for i in range(Nt):
    BadTrotterFixStepList.append(TrotterEvolve((i+1)*dt/(ntrot*3), ntrot*3, init))
BadTrotterFixStepSz = [np.conj(BadTrotterFixStepList[i]) @ op @ BadTrotterFixStepList[i] / 2 for i in range(len(BadTrotterFixStepList))]

BestCompression = [init]
for i in range(len(VTDStepList)):
    BestCompression.append(TrotterEvolve(dtrot, ntrot, VTDStepList.iloc[i]))

# ------------------------------------------------------------------------------------------------------------


fig, ax = plt.subplots(2, 1, figsize=(6,7), sharey=True)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman'] + plt.rcParams['font.serif']

# ax0 = fig.add_subplot(111, frame_on=False)   # creating a single axes
# ax0.set_xticks([])
# ax0.set_yticks([])
# ax0.set_ylabel(r'$\mathcal{F}(t_f,  \hat{\vartheta}^{(\ell)})$', fontsize=16, labelpad=35)

ExFidelity = [abs(np.conj(revos[i]) @ revos[i])**2 for i in range(len(revos))]
BadTrotterFidelity = [abs(np.conj(revos[i]) @ BadTrotterFixStepList[i])**2 for i in range(len(BadTrotterFixStepList))]

revos_ = [expm(-1j * H * t) @ init for t in ts]

TrotterFidelity = [abs(np.conj(revos_[i]) @ TrotterFixStepList[i])**2 for i in range(len(TrotterFixStepList))]
VTDFidelity = [abs(np.conj(revos_[i]) @ np.array(VTDStepList.iloc[i]))**2 for i in range(len(VTDStepList))]
BestCompressionFidelity = [abs(np.conj(revos_[i]) @ np.array(BestCompression[i]))**2 for i in range(len(BestCompression[1:]))]
VTDBestCompressionFidelity = [abs(np.conj(BestCompression[i]) @ np.array(VTDStepList.iloc[i]))**2 for i in range(len(VTDStepList))]

for i in range(2):
    ax[i].tick_params(axis='x', labelsize=12)
    ax[i].tick_params(axis='y', labelsize=12)
    ax[i].xaxis.set_tick_params(width=1.5)
    ax[i].yaxis.set_tick_params(width=1.5)
    ax[i].margins(x=0.05, y=0)
    ax[i].set_xlim(0,20.3)
    ax[i].set_ylabel(r'$\mathcal{F}(t,  \hat{\vartheta}^{(\ell)})$', fontsize=16)
    ax[i].axvspan(9, 21, facecolor='#9cce9c', alpha=0.2)


# ax[0].plot([i*dt for i in range(len(revos))], ExFidelity, label="Exact", linewidth=2)
# ax[1].plot([i*dt for i in range(len(revos))], ExFidelity, linewidth=2)

ax[0].plot(ts, TrotterFidelity, label="Ideal VTC", linewidth=2, c='#ff7f0e', zorder=1)
ax[1].plot(ts, TrotterFidelity, linewidth=2, c='#ff7f0e')

ax[0].errorbar(VTDStepList.index, mean, yerr=err, linestyle="--", marker='.', label="VTC", markersize=5, c='g', linewidth=2)

ax[1].plot(VTDStepList.index, VTDFidelity, linestyle="--", marker='.', markersize=5, c='g', linewidth=2)
ax[1].scatter([i*dtrot*ntrot for i in range(len(BestCompressionFidelity))], BestCompressionFidelity, label="Best Compression", marker='x', c='k', linewidth=2)
ax[1].scatter([i*dtrot*ntrot for i in range(len(VTDBestCompressionFidelity))], VTDBestCompressionFidelity, label="VTC Overlap", marker='x', c='r', linewidth=2)


ax[0].plot([i*dt for i in range(len(revos))], BadTrotterFidelity, label="Trotter $(3\ell)$", c='#CDCDCD', zorder=0, linewidth=2)
ax[1].plot([i*dt for i in range(len(revos))], BadTrotterFidelity, c='#CDCDCD', zorder=0, linewidth=2)

ax[0].legend(loc='lower left', fontsize=12)
ax[1].legend(loc='lower left', fontsize=12)


plt.text(0, 0.87, "(a)", fontsize=16, transform=plt.gcf().transFigure, fontfamily='sans-serif')
plt.text(0, 0.46, "(b)", fontsize=16, transform=plt.gcf().transFigure, fontfamily='sans-serif')


for axis in ['top','bottom','left','right']:
    ax[0].spines[axis].set_linewidth(1.3)
    ax[1].spines[axis].set_linewidth(1.3)

ax[1].set_xlabel('$Jt$', fontsize=16)

plt.savefig(f'../figures/final/combined_VTC.png', dpi=600, transparent=False, bbox_inches='tight')
# plt.show()
import os
import time
import numpy as np
from optimparallel import minimize_parallel
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import expm_multiply
import argparse

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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Command line arguments
    parser.add_argument("-L", "--L", type=int, default=5, help="System size")
    parser.add_argument("-t", "--t", type=int, default=0, help="Time to start at")
    parser.add_argument("-p", "--p", type=int, default=1, help="Number of ansatz steps")
    args = parser.parse_args()
    L = args.L
    t = args.t
    p = args.p

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
    
    H = sum([Heis[i][(i+1)%L] for i in range(L)]) / 4
    tf = 7
    ts = 70
    dt = tf / ts
    Nt = int(tf / dt)
    c = [str((1 + (-1)**(i+1)) // 2) for i in range(L)]
    UnitVector = lambda c: np.eye(2**L)[c]
    init = UnitVector(int(''.join(c), 2))

    revos = [np.zeros(2**L) for i in range(Nt+1)]
    revos[0] = init
    for i in range(Nt):
        # scipy.sparse.linalg.expm_multiply
        revos[i+1] = expm_multiply(-1j * H * dt, revos[i])
        # revos[i+1] = expm(-1j * H * dt) @ revos[i]

    def Ansatz(params, p):
        # check for correct length of params
        psi_ansz = init
        for i in range(p): # len(params) // L
            for j in range(0, L, 2):
                # odd first, then even. Apply to left
                psi_ansz = expm_multiply(-1j * params[(L*i)+j] * Heis[j][(j+1)%L], psi_ansz)
                # psi_ansz = expm(-1j * params[(L*i)+j] * Heis[j][(j+1)%L]) @ psi_ansz
            for j in range(1, L, 2):
                psi_ansz = expm_multiply(-1j * params[(L*i)+j] * Heis[j][(j+1)%L], psi_ansz)
                # psi_ansz = expm(-1j * params[(L*i)+j] * Heis[j][(j+1)%L]) @ psi_ansz
        return psi_ansz

    def Fidelity(x, target, p):
        psi_ansz = Ansatz(x, p)
        return 1 - abs(np.conj(target) @ psi_ansz)**2

    f = open(f'./results_{L}/results_time_{L}_{p}.txt', 'w')

    def OptimizeFidelity(t, p):
        
        init_params = np.random.uniform(0, 2*np.pi, L*p)

        sol = minimize_parallel(fun=Fidelity, x0=init_params, args=(revos[t], p), parallel={'loginfo': True, 'time':True})
        
        # f.write(f'{sol.fun}, {t*tf/ts}\n')
        return sol.fun, t*tf/ts
   

    i = t
    while (i < len(revos)):
        # print(i, p)
        funs = []
        for j in range(2):
            fun, ti = OptimizeFidelity(i, p)
            funs.append(fun)
        f.write(f'{np.mean(funs)}, {ti}\n')
        # f.write(f'\n Converged: Fidelity = {fun} at time t={i} for p={p} \n')
        i += 1

    f.close()

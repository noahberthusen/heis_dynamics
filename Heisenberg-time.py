import os
import time
import numpy as np
from optimparallel import minimize_parallel
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigs
from numpy import linalg as LA
from scipy.sparse.linalg import expm_multiply, expm
#import matplotlib.pyplot as plt
from scipy.optimize import minimize
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
            sprs[j, j] = 2*int(format(j, '0{}b'.format(L))[i])-1
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
                v = lambda i: 2*int(format(j, '0{}b'.format(L))[i])-1 
                if (h != -1):
                    sprs[j, h] = 2
                    sprs[h, j] = 2
                sprs[j, j] = v(i) * v(k)
            _.append(sprs)    
        Heis.append(_)
    
    H = sum([Heis[i][(i+1)%L] for i in range(L)]) / 4
    tf = 20
    ts = 80
    dt = tf / ts
    Nt = int(tf / dt)
    c = [str((1 + (-1)**(i+1)) // 2) for i in range(L)]
    UnitVector = lambda c: np.eye(2**L)[c-1]
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
                psi_ansz = expm(-1j * params[(L*i)+j] * Heis[j][(j+1)%L]) @ psi_ansz
            for j in range(1, L, 2):
                psi_ansz = expm(-1j * params[(L*i)+j] * Heis[j][(j+1)%L]) @ psi_ansz
        return psi_ansz

    def Fidelity(x, target, p):
        psi_ansz = Ansatz(x, p)
        return 1 - abs(np.conj(target) @ psi_ansz)**2

    if not os.path.exists(f'results_{L}'):
        os.makedirs(f'results_{L}')
    f = open(f'./results_{L}/results_{L}_{p}.txt', 'a')

    def OptimizeFidelity(t, p):
        # load in previous x if exists
        if os.path.exists(f'./results_{L}/temp_x_{L}_{t}_{p}.npy'):
            f.write('Loading in previous x array\n')
            with open(f'./results_{L}/temp_x_{L}_{t}_{p}.npy', 'rb') as arrf:
                init_params = np.load(arrf)
        else:
            init_params = np.random.uniform(0, 2*np.pi, L*p)

        sol = minimize_parallel(fun=Fidelity, x0=init_params, args=(revos[t], p), parallel={'loginfo': True, 'time':True}, options={'maxiter':200})
        if (sol.message == b'STOP: TOTAL NO. of ITERATIONS REACHED LIMIT'):
            # ran out of time
            f.write('Ran out of time when p={p} and time t={t}\n')
            with open(f'./results_{L}/temp_x_{L}_{t}_{p}.npy', 'wb') as arrf:
                np.save(arrf, sol.x)
                os.system(f'sbatch job-heis-did2.sh {L} {t} {p}')
            f.close()
            quit()
        else:
            f.write(f'{sol.fun}, {t*tf/ts}\n')
            return sol.fun

    i = t
    while (i < len(revos)):
        # print(i, p)
        fun = OptimizeFidelity(i, p)
        # f.write(f'\n Converged: Fidelity = {fun} at time t={i} for p={p} \n')
        i += 1

    f.close()

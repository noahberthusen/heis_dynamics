import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import eigs
from numpy import linalg as LA
from scipy.linalg import expm
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
from optimparallel import minimize_parallel
import argparse
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Command line arguments
    parser.add_argument("-L", "--L", type=int, default=5, help="System size")
    parser.add_argument("-p", "--p", type=int, default=1, help="Number of ansatz steps")
    parser.add_argument("-t", "--t", type=int, default=20, help="Final time")
    parser.add_argument("-d", "--d", type=float, default=0.25, help="Trotter step size")
    args = parser.parse_args()
    L = args.L
    p = args.p
    dt = args.d
    tf = args.t

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

    Sm = [Sp[i].T for i in range(L)]
    Sx = [Sp[i]+Sm[i] for i in range(L)]
    Sy = [-1j*Sz[i] @ Sx[i] for i in range(L)]
    SxTot = sum(Sx)
    SyTot = sum(Sy)

    H = sum([Heis[i][(i+1)%L] for i in range(L)]) / 4

    c = [str((1 + (-1)**(i+1)) // 2) for i in range(L)]
    UnitVector = lambda c: np.eye(2**L)[c-1]
    init = UnitVector(int(''.join(c), 2))

    def TrotterEvolve(dt, nt, init):
        UOdd = expm(-1j * dt * sum([Heis[i][(i+1)%L] for i in range(0, L, 2)]) / 4) # since Python indices start at 0, this is actually even
        UEven = expm(-1j * dt * sum([Heis[i][(i+1)%L] for i in range(1, L, 2)]) / 4) # since Python indices start at 0, this is actually the odd indices
        # UZ = expm(-1j * dt * sum([diags(Heis[i][(i+2)%L].diagonal()) for i in range(L)]) / 2)
        UTrotter = UEven @ UOdd 
        psi_trot = init
        for i in range(nt):
            psi_trot = UTrotter @ psi_trot
        return psi_trot

    def Ansatz(params, p):
        psi_ansz = init
        for i in range(p): # len(params) // L
            for j in range(0, L, 2):
                psi_ansz = expm(-1j * params[(L*i)+j] * Heis[j][(j+1)%L]) @ psi_ansz
            for j in range(1, L, 2):
                psi_ansz = expm(-1j * params[(L*i)+j] * Heis[j][(j+1)%L]) @ psi_ansz
            # for j in range(L):
            #     psi_ansz = expm(-1j * params[(2*L*i)+L+j] * diags(Heis[j][(j+2)%L].diagonal()).tocsc()) @ psi_ansz
        return psi_ansz

    def Fidelity(x, target, p):
        psi_ansz = Ansatz(x, p)
        return 1 - abs(np.conj(target) @ psi_ansz)**2

    def VTD(tf, dt, p, init):
        VTDFixStep = [init]
        TimeStep = [0]
        nt = int(np.ceil(tf / (dt * p)))

        if (os.path.exists(f'./results_{L}/VTD_results_{tf}_{L}_{p}_{dt}.csv')):
            VTDStepList = pd.read_csv(f'./results_{L}/VTD_results_{tf}_{L}_{p}_{dt}.csv', index_col=0)
            VTDStepList = VTDStepList.applymap(lambda x: complex(x))
        else:
            VTDStepList = pd.DataFrame(np.array(VTDFixStep), index=np.array(TimeStep))
        
        ts = VTDStepList.index
        temp = VTDStepList.iloc[-1]
        temp = TrotterEvolve(dt, p, temp)
        
        init_params = np.random.uniform(0, np.pi, L*p)
        sol = minimize_parallel(fun=Fidelity, x0=init_params, args=(temp, p))
        temp = Ansatz(sol.x, p)
        VTDStepList.loc[ts[-1]+(dt*p)] = np.array(temp) 
        ts = VTDStepList.index


        VTDStepList.to_csv(f'./results_{L}/VTD_results_{tf}_{L}_{p}_{dt}.csv')

        if (ts[-1] >= tf):
            return
        else:
            os.system(f'sbatch job-heis-did2.sh {L} {p} {tf} {dt}')

    res = VTD(tf, dt, p, init)


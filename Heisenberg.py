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
    parser.add_argument("-p", "--p", type=int, default=3, help="Number of ansatz steps")
    parser.add_argument("-n", "--num_iter", type=int, default=1, help="Number of times to minimize (mean)")
    args = parser.parse_args()
    L = args.L
    p = args.p
    n = args.num_iter

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
    
    H = sum([Heis[i][(i+1)%L] for i in range(L-1)]) / 4 # L for PBC, L-1 for OBC
    tf = 50
    dt = tf / 200
    Nt = int(tf / dt)
    c = [str((1 + (-1)**(i+1)) // 2) for i in range(L)]
    UnitVector = lambda c: np.eye(2**L)[c]
    init = UnitVector(int(''.join(c), 2))

    revos = [np.zeros(2**L) for i in range(Nt+1)]
    revos[0] = init
    for i in range(Nt):
        revos[i+1] = expm_multiply(-1j * H * dt, revos[i])

    def Ansatz(params):
        psi_ansz = init
        for i in range(p): # len(params) // L
            for j in range(1, L-1, 2):
                # odd first, then even. Apply to left
                psi_ansz = expm_multiply(-1j * params[(L*i)+j] * Heis[j][(j+1)%L], psi_ansz)
            for j in range(0, L-1, 2):
                psi_ansz = expm_multiply(-1j * params[(L*i)+j] * Heis[j][(j+1)%L], psi_ansz)
        
        # for i in range(p): # l
        #     for j in range(0, L-1):
        #         psi_ansz = expm_multiply(-1j * params[(L*i)+j] * Heis[j][(j+1)%L], psi_ansz)

        return psi_ansz

    def Fidelity(x, target):
        psi_ansz = Ansatz(x)
        return 1 - abs(np.conj(target) @ psi_ansz)**2

    def Loss(x, target):
        psi_ansz = Ansatz(x)
        Sz_ansz = np.conj(psi_ansz) @ (Sz[0] + Sz[1] + Sz[2] + Sz[3] + Sz[4]) @ psi_ansz / 2
        Sz_ex = np.conj(target) @ (Sz[0] + Sz[1] + Sz[2] + Sz[3] + Sz[4]) @ target / 2
        return abs(Sz_ansz - Sz_ex)

    if not os.path.exists(f'results_{L}'):
        os.makedirs(f'results_{L}')
    f = open(f'./results_{L}/results_{L}_{p}.txt', 'w')

    start = time.time()

    # load in previous x if exists
    #if os.path.exists(f'./results_{L}/temp_x_{L}_{p}.npy'):
    #    f.write('Loading in previous x array\n')
    #    with open(f'./results_{L}/temp_x_{L}_{p}.npy', 'rb') as arrf:
    #        init_params = np.load(arrf)
    #else:
    
    #init_params = np.random.uniform(0, 2*np.pi, L*p)
    funs = []

    for i in range(n):
        init_params = np.random.uniform(0, 2*np.pi, L*p)
        sol = minimize_parallel(fun=Fidelity, x0=init_params, args=(revos[-1]), parallel={'loginfo': True, 'time':True})
        funs.append(sol.fun)

    #if (sol.message == b'STOP: TOTAL NO. of ITERATIONS REACHED LIMIT'):
    #    # ran out of time
    #    f.write(str(sol))
    #    f.write('Ran out of time\n')
    #    with open(f'./results_{L}/temp_x_{L}_{p}.npy', 'wb') as arrf:
    #        np.save(arrf, sol.x)
    #        os.system(f'sbatch job-heis-did2.sh {L} {p}')
    #    f.close()
    #    quit()

    #else:
    #    f.write(str(sol))
    
    end = time.time()
    f.write(f'\n{funs}\n')
    f.write(f'\n{np.mean(funs)}\n')
    f.write(f'Total time taken: {end-start}s. Time per run: {(end-start)/n}\n')

    f.close()

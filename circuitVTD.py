import numpy as np
from qiskit import *
from qiskit import Aer
from scipy.optimize import minimize
import pandas as pd
import argparse

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

    def TrotterEvolveCircuit(dt, nt, init):
        """
        Implements trotter evolution of the Heisenberg hamiltonian using the circuit from https://arxiv.org/pdf/1906.06343.pdf
        :param tf: time to evolve to
        :param nt: number of trotter steps to use
        :param init: initial state for the trotter evolution. Should be another Qiskit circuit
        """

        # def get_angles(a, b, c):
        #     return (np.pi/2 - 2*c, 2*a - np.pi/2, np.pi/2 - 2*b)
        def get_angles(a):
            return (np.pi/2 - 2*a, 2*a - np.pi/2, np.pi/2 - 2*a)

        def N(circ, qb0, qb1):
            circ.rz(-np.pi/2, qb1)
            circ.cnot(qb1, qb0)
            circ.rz(theta, qb0)
            circ.ry(phi, qb1)
            circ.cnot(qb0, qb1)
            circ.ry(lambd, qb1)
            circ.cnot(qb1, qb0)
            circ.rz(np.pi/2, qb0)
            return circ

        theta, phi, lambd = get_angles(-dt/4)
        circ = init

        for i in range(nt):
            # even (odd indices)
            if (L % 2 == 0):
                # UEven
                for i in range(1, L, 2):
                    circ = N(circ, i, (i+1)%L)
                # UOdd
                for i in range(0, L-1, 2):
                    circ = N(circ, i, (i+1)%L)
            else:
                # UEven
                for i in range(1, L, 2):
                    circ = N(circ, i, (i+1)%L)
                # UOdd
                for i in range(0, L-1, 2):
                    circ = N(circ, i, (i+1)%L)
                # UBdy
                circ = N(circ, L-1, 0)
        
        return circ

    def AnsatzCircuit(params, p):
        """

        """
        circ = QuantumCircuit(L)

        def get_angles(a):
            return (np.pi/2 - 2*a, 2*a - np.pi/2, np.pi/2 - 2*a)

        def N(cir, angles, qb0, qb1):
            cir.rz(-np.pi/2, qb1)
            cir.cnot(qb1, qb0)
            cir.rz(angles[0], qb0)
            cir.ry(angles[1], qb1)
            cir.cnot(qb0, qb1)
            cir.ry(angles[2], qb1)
            cir.cnot(qb1, qb0)
            cir.rz(np.pi/2, qb0)
            return cir

        for i in range(p):
            if (L % 2 == 0):
                for j in range(1, L, 2):
                    circ = N(circ, get_angles(-params[(L*i)+j]/4), j, (j+1)%L)
                for j in range(0, L-1, 2):
                    circ = N(circ, get_angles(-params[(L*i)+j]/4), j, (j+1)%L)
            else:
                for j in range(1, L, 2):
                    circ = N(circ, get_angles(-params[(L*i)+j]/4), j, (j+1)%L)
                for j in range(0, L-1, 2):
                    circ = N(circ, get_angles(-params[(L*i)+j]/4), j, (j+1)%L)
                circ = N(circ, get_angles(-params[(L*i)+L-1]/4), L-1, 0)
        
        return circ

    def ReorderBasis(circ):
        """
        Reorders basis so that 0th qubit is on the left side of the tensor product
        :param circ: circuit to reorder, can also be a vector
        """
        if (isinstance(circ, qiskit.circuit.quantumcircuit.QuantumCircuit)):
            for i in range(L//2):
                circ.swap(i, L-i-1)
            return circ
        else:
            perm = np.eye(2**L)
            for i in range(1, 2**L//2-1):
                perm[:, [i, 2**L-i-1]] = perm[:, [2**L-i-1, i]]
            return perm @ circ

    def SimulateAndReorder(circ):
        """
        Executes a circuit using the statevector simulator and reorders basis to match with standard
        """
        circ = ReorderBasis(circ)
        backend = Aer.get_backend('statevector_simulator')
        return execute(circ, backend).result().get_statevector()

    def Simulate(circ):
        """
        Executes a circuit using the statevector simulator. Doesn't reorder -- which is needed for intermediate steps in the VTC
        """
        backend = Aer.get_backend('statevector_simulator')
        return execute(circ, backend).result().get_statevector()

    def LoschmidtEcho(params, U_v, U_trot, init, p):
        """
        Cost function using the Loschmidt Echo. Just using statevectors currently -- can rewrite using shots
        :param params: parameters new variational circuit that represents U_trot U_v | init >. Need dagger for cost function
        :param U_v: variational circuit that stores the state before the trotter step
        :param U_trot: trotter step
        :param init: initial state
        :param p: number of ansatz steps
        """
        U_v_prime = AnsatzCircuit(params, p)
        circ = init + U_v + U_trot + U_v_prime.inverse()
        
        loschmidt = Simulate(circ)
        init = Simulate(init)
        return 1 - abs(np.conj(init) @ loschmidt)**2

    def SwapTest(params, U_v, U_trot, init, p, shots):
        """
        Cost function using the swap test. 
        :param params: parameters new variational circuit that represents U_trot U_v | init >. Need dagger for cost function
        :param U_v: variational circuit that stores the state before the trotter step
        :param U_trot: trotter step
        :param init: initial state
        :param p: number of ansatz steps
        :param shots: number of measurements to take
        """
        U_v_prime = init + AnsatzCircuit(params, p)
        U_v_prime = U_v_prime.qasm()
        U_v_prime = U_v_prime.replace(f'q[{L}]', f'q[{2*L+1}]')

        for i in range(L, -1, -1):
            U_v_prime = U_v_prime.replace(f'q[{i}]', f'q[{i+1}]')
        U_v_prime = circuit.QuantumCircuit.from_qasm_str(U_v_prime)

        comp = init + U_v + U_trot
        comp = comp.qasm()
        comp = comp.replace(f'q[{L}]', f'q[{2*L+1}]')
        for i in range(L, -1, -1):
            comp = comp.replace(f'q[{i}]', f'q[{L+i+1}]')
        comp = circuit.QuantumCircuit.from_qasm_str(comp)

        circ = QuantumCircuit(2*L+1, 1)
        circ.h(0)
        circ += U_v_prime 
        circ += comp

        # controlled swaps
        for i in range(L):
            circ.cswap(0, i+1, L+i+1)
        circ.h(0)
        circ.measure(0,0)
        
        backend_sim = Aer.get_backend('qasm_simulator')
        job_sim = execute(circ, backend_sim, shots=shots)
        res = job_sim.result().get_counts()

        # if (res.get('1') is None):
            # return 0
        # else:
        return 1 - (res.get('0')/shots)

    def VTC(tf, dt, p, init):
        """
        :param init: initial state as a circuit
        """
        nt = int(np.ceil(tf / (dt * p)))

        VTCParamList = []
        VTCStepList = [Simulate(init)]
        TimeStep = [0]

        for i in range(nt):
            print(i)
            if (i == 0):
                U_v = QuantumCircuit(L)
            else:
                U_v = AnsatzCircuit(VTCParamList[-1], p)
            U_trot = TrotterEvolveCircuit(dt, p, QuantumCircuit(L))
            
            init_params = np.random.uniform(0, np.pi, L*p)
            # res = minimize(fun=LoschmidtEcho, x0=init_params, args=(U_v, U_trot, init, p))
            res = minimize(fun=SwapTest, x0=init_params, args=(U_v, U_trot, init, p, 1024), method='Powell')

            VTCParamList.append(res.x)
            VTCStepList.append(Simulate(init + AnsatzCircuit(res.x, p)))
            TimeStep.append(TimeStep[-1]+(dt*p))

        VTCStepList = [ReorderBasis(VTCStepList[i]) for i in range(len(VTCStepList))]

        VTCParamList = pd.DataFrame(np.array(VTCParamList), index=np.array(TimeStep[1:]))
        VTCStepList = pd.DataFrame(np.array(VTCStepList), index=np.array(TimeStep))

        VTCStepList.to_csv(f'./results_{L}/VTD_results_{tf}_{L}_{p}_{dt}.csv')

    init = QuantumCircuit(L)
    c = [str((1 + (-1)**(i+1)) // 2) for i in range(L)]
    for q in range(len(c)):
        if (c[q] == '1'):
            init.x(q)

    VTC(tf, dt, p, init)

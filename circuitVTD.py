import numpy as np
from qiskit import *
from qiskit import Aer
from scipy.optimize import minimize
import pandas as pd
from qiskit.test.mock import *
from qiskit.providers.aer import AerSimulator
from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter
from functools import partial
import itertools
import mitiq
import argparse
from cmaes import CMA
import cma

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
                for i in range(1, L-1, 2): # L for periodic bdy conditions
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
                # circ = N(circ, L-1, 0)
        
        return circ

    def AnsatzCircuit(params, p):
        """
        Implements HVA ansatz using circuits from https://arxiv.org/pdf/1906.06343.pdf
        :param params: parameters to parameterize circuit
        :param p: depth of the ansatz
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
                for j in range(1, L-1, 2): # L for periodic bdy conditions
                    circ = N(circ, get_angles(-params[((L-1)*i)+j]/4), j, (j+1)%L)
                for j in range(0, L-1, 2):
                    circ = N(circ, get_angles(-params[((L-1)*i)+j]/4), j, (j+1)%L)
            else:
                for j in range(1, L, 2):
                    circ = N(circ, get_angles(-params[((L-1)*i)+j]/4), j, (j+1)%L)
                for j in range(0, L-1, 2):
                    circ = N(circ, get_angles(-params[((L-1)*i)+j]/4), j, (j+1)%L)
                # circ = N(circ, get_angles(-params[(L*i)+L-1]/4), L-1, 0) # boundary
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
            for i in range(1, 2**L//2):
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

    def SwapTestCircuit(params, U_V, U_trot, init, p):
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

        return circ

    def SwapTestExecutor(circuit, backend, shots, filter):
        job = qiskit.execute(
            experiments=circuit,
            backend=backend,
            optimization_level=0,
            shots=shots
        )

        res = job.result()
        if (filter is not None):
            res = filter.apply(res)
        counts = res.get_counts()

        if counts.get('0') is None:
            expectation_value = 0.
        else:
            expectation_value = counts.get('0') / shots
        return expectation_value
        
    def SwapTest(params, U_v, U_trot, init, p, backend, shots, filter):
        """

        """
        circ = SwapTestCircuit(params, U_v, U_trot, init, p)
        return 1 - mitiq.zne.execute_with_zne(circ, partial(SwapTestExecutor, backend=backend, shots=shots, filter=filter))

    def LoschmidtEchoExecutor(circuits, backend, shots, filter):
        """
        Returns the expectation value to be mitigated.
        :param circuit: Circuit to run.
        :param backend: backend to run the circuit  on
        :param shots: Number of times to execute the circuit to compute the expectation value.
        :param fitter: measurement error mitigator
        """
        scale_factors = [1.0, 2.0, 3.0]
        folded_circuits = []
        for circuit in circuits:
            folded_circuits.append([mitiq.zne.scaling.fold_gates_at_random(circuit, scale) for scale in scale_factors])
        folded_circuits = list(itertools.chain(*folded_circuits))

        job = qiskit.execute(
            experiments=folded_circuits,
            backend=backend,
            optimization_level=0,
            shots=shots
        )

        c = c = ['1', '1', '0'] #[str((1 + (-1)**(i+1)) // 2) for i in range(L)]
        c = ''.join(c)[::-1]
        res = job.result()
        if (filter is not None):
            res = filter.apply(res)

        all_counts = [job.result().get_counts(i) for i in range(len(folded_circuits))]
        expectation_values = []
        for counts in all_counts:
            if counts.get(c) is None:
                expectation_values.append(0)
            else:
                expectation_values.append(counts.get(c)/shots)
        # expectation_values = [counts.get(c) / shots for counts in all_counts]
        
        zero_noise_values = []
        if isinstance(backend, qiskit.providers.aer.backends.qasm_simulator.QasmSimulator): # exact_sim
            for i in range(len(circuits)):
                zero_noise_values.append(np.mean(expectation_values[i*len(scale_factors):(i+1)*len(scale_factors)]))
        else: #device_sim
            fac = mitiq.zne.inference.RichardsonFactory(scale_factors)
            for i in range(len(circuits)):
                zero_noise_values.append(fac.extrapolate(scale_factors, 
                expectation_values[i*len(scale_factors):(i+1)*len(scale_factors)]))
                
        return zero_noise_values

    def LoschmidtEchoCircuit(params, U_v, U_trot, init, p):
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
        circ.measure_all()
        return circ

    def LoschmidtEcho(params, U_v, U_trot, init, p, backend, shots, filter):
        """

        """
        circs = []
        for param in params:
            circs.append(LoschmidtEchoCircuit(param, U_v, U_trot, init, p))
        res = LoschmidtEchoExecutor(circs, backend, shots, filter)
        return 1 - np.array(res)

    def LoschmidtEchoExact(params, U_v, U_trot, init, p):
        U_v_prime = AnsatzCircuit(params, p)
        circ = init + U_v + U_trot + U_v_prime.inverse()

        circ_vec = Simulate(circ)
        init_vec = Simulate(init)
        return 1 - abs(np.conj(circ_vec) @ init_vec)**2

    def CMAES(U_v, U_trot, init, p, backend, shots, filter):
        """
        num_generations = 60
        optimizer = CMA(mean=np.zeros(p*(L-1)), sigma=np.pi/2)
        pop_size = optimizer.population_size

        for generation in range(num_generations):
            print('.', end='')
            solutions = []
            params = []
            for _ in range(pop_size):
                x = optimizer.ask()
                params.append(x)
            values = LoschmidtEcho(params, U_v, U_trot, init, p, backend, shots, filter)
            for i in range(optimizer.population_size):
                solutions.append((params[i], values[i]))
            optimizer.tell(solutions)
        return solutions[0]
        """
        init_params = np.random.uniform(0, 2*np.pi, (L-1)*p)
        es = cma.CMAEvolutionStrategy(init_params, np.pi/2)
        es.opts.set({'ftarget':5e-4, 'maxiter':100})
        while not es.stop():
            solutions = es.ask()
            es.tell(solutions, LoschmidtEcho(solutions, U_v, U_trot, init, p, backend, shots, filter))
            es.disp()
        return es.result_pretty()

    def VTC(tf, dt, p, init, backend, shots, filter):
        """
        :param init: initial state as a circuit
        """
        nt = int(np.ceil(tf / (dt * p)))

        VTCParamList = []
        VTCStepList = [SimulateAndReorder(init.copy())]
        TrotterFixStepList = [init]
        TimeStep = [0]

        for i in range(nt):
            print(i, nt)
            if (i == 0):
                U_v = QuantumCircuit(L)
            else:
                U_v = AnsatzCircuit(VTCParamList[-1], p)
            U_trot = TrotterEvolveCircuit(dt, p, QuantumCircuit(L))
            
            TrotterFixStepList.append(TrotterFixStepList[-1] + U_trot)
            
            init_params = np.random.uniform(0, np.pi, (L-1)*p)
            # res = minimize(fun=SwapTest, x0=init_params, args=(U_v, U_trot, init, p, 8192), method='COBYLA')
            # res = minimize(fun=LoschmidtEcho, x0=init_params, args=(U_v, U_trot, init, p, exact_sim, 8192, None), method='COBYLA')
            # res = minimize(fun=LoschmidtEchoExact, x0=init_params, args=(U_v, U_trot, init, p))
            
            
            res = CMAES(U_v, U_trot, init, p, backend, shots, filter)

            print(res)
            
            VTCParamList.append(res.xbest) #res.x
            VTCStepList.append(SimulateAndReorder(init + AnsatzCircuit(res.xbest, p))) #res.x
            TimeStep.append(TimeStep[-1]+(dt*p))
        
        TrotterFixStepList = pd.DataFrame(np.array([SimulateAndReorder(c.copy()) for c in TrotterFixStepList]), index=np.array(TimeStep))
        VTCParamList = pd.DataFrame(np.array(VTCParamList), index=np.array(TimeStep[1:]))
        VTCStepList = pd.DataFrame(np.array(VTCStepList), index=np.array(TimeStep))

        VTCStepList.to_csv(f'./results_{L}/VTD_results_{tf}_{L}_{p}_{dt}_{shots}.csv')


    qr = QuantumRegister(L)
    meas_calibs, state_labels = complete_meas_cal(qr=qr, circlabel='mcal')

    device_backend = FakeYorktown()
    device_sim = AerSimulator.from_backend(device_backend)
    exact_sim = Aer.get_backend('qasm_simulator')
    sim = device_sim

    # Execute the calibration circuits without noise
    t_qc = transpile(meas_calibs, sim)
    qobj = assemble(t_qc, shots=10000)
    cal_results = sim.run(qobj, shots=10000).result()
    meas_fitter = CompleteMeasFitter(cal_results, state_labels, circlabel='mcal')
    np.around(meas_fitter.cal_matrix, decimals=2)

    init = QuantumCircuit(L)
    c = ['1', '1', '0'] # [str((1 + (-1)**(i+1)) // 2) for i in range(L)]
    for q in range(len(c)):
        if (c[q] == '1'):
            init.x(q)

    VTC(tf, dt, p, init, exact_sim, 2**13, None)

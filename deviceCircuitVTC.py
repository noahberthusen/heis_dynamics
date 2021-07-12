import numpy as np
from qiskit import *
from qiskit import Aer
import pandas as pd
from qiskit.test.mock import *
from qiskit.providers.aer import AerSimulator
from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter
import itertools
import mitiq
import argparse
import cma
import os
import sys
from qiskit import IBMQ
from qiskit.tools.monitor import job_monitor
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Command line arguments
    parser.add_argument("-L", "--L", type=int, default=5, help="System size")
    parser.add_argument("-p", "--p", type=int, default=1, help="Number of ansatz steps")
    parser.add_argument("-t", "--t", type=int, default=20, help="Final time")
    parser.add_argument("-d", "--d", type=float, default=0.25, help="Trotter step size")
    parser.add_argument("-s", "--s", type=int, default=2**13, help="Number of shots")
    args = parser.parse_args()
    L = args.L
    p = args.p
    dt = args.d
    tf = args.t
    shots = args.s

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

    def SwapTestCircuit(params, U_v, U_trot, init, p):
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

    def SwapTestExecutor(circuits, backend, shots, filter):
        scale_factors = [1.0, 1.5, 2.0]
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

        res = job.result()
        if (filter is not None):
            res = filter.apply(res)

        all_counts = [job.result().get_counts(i) for i in range(len(folded_circuits))]
        expectation_values = []
        for counts in all_counts:
            if counts.get('0') is None:
                expectation_values.append(0)
            else:
                expectation_values.append(counts.get('0')/shots)
        
        zero_noise_values = []
        if isinstance(backend, qiskit.providers.aer.backends.qasm_simulator.QasmSimulator): # exact_sim
            for i in range(len(circuits)):
                zero_noise_values.append(np.mean(expectation_values[i*len(scale_factors):(i+1)*len(scale_factors)]))
        else: #device_sim
            fac = mitiq.zne.inference.LinearFactory(scale_factors)
            for i in range(len(circuits)):
                zero_noise_values.append(fac.extrapolate(scale_factors, 
                expectation_values[i*len(scale_factors):(i+1)*len(scale_factors)]))

        return zero_noise_values

        
    def SwapTest(params, U_v, U_trot, init, p, backend, shots, filter):
        """

        """
        circs = []
        for param in params:
            circs.append(SwapTestCircuit(param, U_v, U_trot, init, p))
        res = SwapTestExecutor(circs, backend, shots, filter)
        return abs(1 - np.array(res))

    def LoschmidtEchoExecutor(circuits, backend, shots, filter):
        """
        Returns the expectation value to be mitigated.
        :param circuit: Circuit to run.
        :param backend: backend to run the circuit  on
        :param shots: Number of times to execute the circuit to compute the expectation value.
        :param fitter: measurement error mitigator
        """
        scale_factors = [1.0, 1.5, 2.0]
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
        job_monitor(job)

        c = ['1','1','0']
        # c = [str((1 + (-1)**(i+1)) // 2) for i in range(L)]
        c = ''.join(c)[::-1]
        res = job.result()
        if (filter is not None):
            res = filter.apply(res)

        all_counts = [job.result().get_counts(i) for i in range(len(folded_circuits))]
        expectation_values = []
        for counts in all_counts:
            total_allowed_shots = [counts.get(''.join(p)) for p in set(itertools.permutations(c))]
            total_allowed_shots = sum([0 if x is None else x for x in total_allowed_shots])
            if counts.get(c) is None:
                expectation_values.append(0)
            else:
                expectation_values.append(counts.get(c)/total_allowed_shots)
        # expectation_values = [counts.get(c) / shots for counts in all_counts]
        
        zero_noise_values = []
        if isinstance(backend, qiskit.providers.aer.backends.qasm_simulator.QasmSimulator): # exact_sim
            for i in range(len(circuits)):
                zero_noise_values.append(np.mean(expectation_values[i*len(scale_factors):(i+1)*len(scale_factors)]))
        else: #device_sim
            fac = mitiq.zne.inference.LinearFactory(scale_factors)
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
        return abs(1 - np.array(res))

    def LoschmidtEchoExact(params, U_v, U_trot, init, p):
        U_v_prime = AnsatzCircuit(params, p)
        circ = init + U_v + U_trot + U_v_prime.inverse()

        circ_vec = Simulate(circ)
        init_vec = Simulate(init)
        return 1 - abs(np.conj(circ_vec) @ init_vec)**2

    def CMAES(U_v, U_trot, init, p, backend, shots, filter):
        init_params = np.random.uniform(0, 2*np.pi, (L-1)*p)
        es = cma.CMAEvolutionStrategy(init_params, np.pi/2)
        es.opts.set({'ftarget':5e-3, 'maxiter':1000})
        # es = pickle.load(open(f'./results_{L}/optimizer_dump', 'rb'))
        while not es.stop():
            solutions = es.ask(25)
            es.tell(solutions, LoschmidtEcho(solutions, U_v, U_trot, init, p, backend, shots, filter))
            es.disp()
            open(f'./results_{L}/optimizer_dump', 'wb').write(es.pickle_dumps())
        return es.result_pretty()

    def VTC(tf, dt, p, init, backend, shots, filter):

        VTCParamList = [np.zeros((L-1)*p)]
        VTCStepList = [SimulateAndReorder(init.copy())]
        # TrotterFixStepList = [init]
        TimeStep = [0]

        if (os.path.exists(f'./results_{L}/VTD_params_{tf}_{L}_{p}_{dt}_{shots}.csv')):
            VTCParamList = pd.read_csv(f'./results_{L}/VTD_params_{tf}_{L}_{p}_{dt}_{shots}.csv', index_col=0)
            VTCStepList = pd.read_csv(f'./results_{L}/VTD_results_{tf}_{L}_{p}_{dt}_{shots}.csv', index_col=0)
            temp = VTCParamList.iloc[-1]
            U_v = AnsatzCircuit(temp, p)
        else:
            VTCParamList = pd.DataFrame(np.array(VTCParamList), index=np.array(TimeStep))
            VTCStepList = pd.DataFrame(np.array(VTCStepList), index=np.array(TimeStep))
            U_v = QuantumCircuit(L)

        ts = VTCParamList.index
        U_trot = TrotterEvolveCircuit(dt, p, QuantumCircuit(L))

        res = CMAES(U_v, U_trot, init, p, backend, shots, filter)
        print(res)

        res = res.xbest
        VTCParamList.loc[ts[-1]+(dt*p)] = np.array(res) 
        VTCStepList.loc[ts[-1]+(dt*p)] = np.array(SimulateAndReorder(init + AnsatzCircuit(res, p)))
        ts = VTCParamList.index

        # VTCParamList = pd.DataFrame(np.array(VTCParamList), index=np.array(TimeStep))
        # VTCStepList = pd.DataFrame(np.array(VTCStepList), index=np.array(TimeStep))

        VTCParamList.to_csv(f'./results_{L}/VTD_params_{tf}_{L}_{p}_{dt}_{shots}.csv')
        VTCStepList.to_csv(f'./results_{L}/VTD_results_{tf}_{L}_{p}_{dt}_{shots}.csv')

        if (ts[-1] >= tf):
            return
        else:
            os.system(f'python deviceCircuitVTC.py -L {L} -p {p} -t {tf} -d {dt} -s {shots}')

    provider = IBMQ.load_account()
    qr = QuantumRegister(L)
    meas_calibs, state_labels = complete_meas_cal(qr=qr, circlabel='mcal')

    device_backend = FakeSantiago()
    device_sim = AerSimulator.from_backend(device_backend)
    # noise_model = NoiseModel.from_backend(device_backend)
    real_device = provider.get_backend('ibmq_manila')
    # device_sim = QasmSimulator(method='statevector', noise_model=noise_model)
    exact_sim = Aer.get_backend('qasm_simulator') # QasmSimulator(method='statevector')

    t_qc = transpile(meas_calibs, device_sim)
    qobj = assemble(t_qc, shots=8192)
    cal_results = real_device.run(qobj, shots=8192).result()
    meas_fitter = CompleteMeasFitter(cal_results, state_labels, circlabel='mcal')
    # np.around(meas_fitter.cal_matrix, decimals=2)

    init = QuantumCircuit(L)
    # c = [str((1 + (-1)**(i+1)) // 2) for i in range(L)]
    c = ['1','1','0']
    for q in range(len(c)):
        if (c[q] == '1'):
            init.x(q)

    nt = int(np.ceil(tf / (dt * p)))
    f = open(f'./results_{L}/logging.txt', 'a')
    sys.stdout = f
    VTC(tf, dt, p, init, real_device, shots, meas_fitter.filter)
    f.close()
import time

from qiskit_aer import AerSimulator
from qamomile.qiskit import QiskitExecutor, QiskitTranspiler
from qiskit import transpile
from qiskit_ibm_runtime import SamplerV2, QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

class IBMRuntimeExecutor(QiskitExecutor):
    """QiskitExecutor that runs sampling through a pre-built SamplerV2.

    Use this for IBM Quantum hardware or IBM Runtime cloud simulators.
    """

    def __init__(self, sampler, backend, estimator=None, optimization_level=1):
        """
        Args:
            sampler: A configured SamplerV2 instance (e.g. SamplerV2(mode=backend),
                     SamplerV2(mode=Session(backend=...)), or SamplerV2(mode=Batch(...))).
            backend: The Qiskit backend the sampler is targeting. Used to build
                     the ISA pass manager so circuits are transpiled to the
                     backend's native gate set before being sent to the sampler.
            estimator: Optional EstimatorV2 for expectation values.
            optimization_level: Preset pass manager optimization level (0-3).
        """
        super().__init__(backend=backend, estimator=estimator)
        self._sampler = sampler
        self._pm = generate_preset_pass_manager(
            optimization_level=optimization_level, backend=backend
        )

    def execute(self, circuit, shots):
        # Strip any measurements before ISA transpilation; HLS cannot synthesize them.
        circuit = circuit.remove_final_measurements(inplace=False)
        isa_circuit = self._pm.run(circuit)
        isa_circuit = self._ensure_measurements(isa_circuit)

        job = self._sampler.run([isa_circuit], shots=shots)
        pub_result = job.result()[0]

        # SamplerV2 stores counts under the classical register name(s).
        # measure_all() -> "meas"; explicit named registers -> that name.
        data = pub_result.data
        reg_name = next(iter(data))
        return getattr(data, reg_name).get_counts()


_AER_PREPARED = "__aer_prepared__"


class AerLocalExecutor(QiskitExecutor):
    """QiskitExecutor that runs sampling directly on a local AerSimulator.

    Bypasses SamplerV2 so all options configured on the backend instance
    (method, device, max_memory_mb, max_qubits, ...) are honoured without
    any re-wrapping or option-forwarding logic.  Use this for local CPU or
    GPU simulations (including tensor_network on NVIDIA GPUs).

    No external pass manager is used.  AerSimulator handles standard Qiskit
    gates (RZZ, RXX, RX, CX, ...) natively in its C++ layer.  Running an
    external preset pass manager would decompose RZZ -> CX+RZ+CX, multiplying
    the two-qubit gate count ~3x and making tensor-network path-finding
    intractable for circuits with many interactions.

    Transpilation is still cached once (just measurements stripped/re-added),
    so every optimizer iteration only pays for assign_parameters().
    """

    def __init__(self, backend, estimator=None, optimization_level=1, verbose=False):
        """
        Args:
            backend: A configured AerSimulator instance.
            estimator: Optional EstimatorV2 for expectation values.
            optimization_level: Kept for API compatibility; not used for
                AerSimulator — native gate handling is always preferred.
            verbose: Print per-call timing and device info for the first 5
                     calls and every 50th call afterwards.
        """
        super().__init__(backend=backend, estimator=estimator)
        self._run_backend = backend
        # Maps id(template_circuit) -> measured parametric circuit (no decomp).
        self._transpiled_cache: dict[int, object] = {}
        self._verbose = verbose
        self._call_count = 0

    def bind_parameters(self, circuit, bindings, parameter_metadata):
        """Cache the circuit once (measurements only), then bind per iteration."""
        cid = id(circuit)
        if cid not in self._transpiled_cache:
            # Strip any existing measurements, add them back cleanly.
            # Do NOT run a pass manager: external decomposition of gates like
            # RZZ -> CX+RZ+CX multiplies gate count ~5x and causes cuTensorNet
            # path-finding to fail or time out.
            stripped = circuit.remove_final_measurements(inplace=False)
            prepared = self._ensure_measurements(stripped)
            prepared.metadata[_AER_PREPARED] = True
            self._transpiled_cache[cid] = prepared

        qiskit_bindings = {
            p.backend_param: bindings[p.name]
            for p in parameter_metadata.parameters
            if p.name in bindings
        }
        return self._transpiled_cache[cid].assign_parameters(qiskit_bindings)

    def execute(self, circuit, shots):
        if not circuit.metadata.get(_AER_PREPARED):
            # Non-parametric circuit sent directly (no prior bind_parameters).
            circuit = self._ensure_measurements(
                circuit.remove_final_measurements(inplace=False)
            )

        t0 = time.perf_counter()
        result = self._run_backend.run(circuit, shots=shots).result()
        t1 = time.perf_counter()

        self._call_count += 1
        if self._verbose and (self._call_count <= 5 or self._call_count % 50 == 0):
            exp = result.results[0]
            meta = exp.metadata
            sim_ms = (exp.time_taken or 0.0) * 1000
            total_ms = (t1 - t0) * 1000
            print(
                f"[AerLocalExecutor #{self._call_count}] "
                f"total={total_ms:.1f}ms  sim={sim_ms:.2f}ms  "
                f"overhead={total_ms - sim_ms:.1f}ms  "
                f"device={meta.get('device', '?')}  method={meta.get('method', '?')}  "
                f"qubits={circuit.num_qubits}  gates={circuit.size()}  "
                f"success={result.success}  status={exp.status}"
            )

        if not result.success:
            exp = result.results[0]
            raise RuntimeError(
                f"Aer simulation failed — status: {exp.status!r}. "
                f"Circuit: {circuit.num_qubits} qubits, {circuit.size()} gates, "
                f"device={self._run_backend.options.get('device', '?')}, "
                f"method={self._run_backend.options.get('method', '?')}. "
                f"Try device='CPU' if this is a GPU OOM error."
            )

        return result.get_counts(0)

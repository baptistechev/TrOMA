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
    (method, device, max_memory_mb, max_qubits, …) are honoured without
    any re-wrapping or option-forwarding logic.  Use this for local CPU or
    GPU simulations (including tensor_network on NVIDIA GPUs).

    Transpilation is cached: the pass manager runs once on the parametric
    circuit template and the result is reused across all optimizer iterations
    (Qamomile passes the same circuit object each time, only the angle values
    change).

    Parameter binding uses qiskit-aer's parameter_binds API so the C++
    backend receives the same parametric circuit object on every call and
    can cache its compiled representation, rather than seeing a freshly
    bound (structurally identical but object-distinct) circuit each time.
    """

    def __init__(self, backend, estimator=None, optimization_level=1, verbose=False):
        """
        Args:
            backend: A configured AerSimulator instance.
            estimator: Optional EstimatorV2 for expectation values.
            optimization_level: Preset pass manager optimization level (0-3).
            verbose: Print per-call timing and device info for the first few
                     calls and every 50th call afterwards.
        """
        super().__init__(backend=backend, estimator=estimator)
        self._run_backend = backend
        self._pm = generate_preset_pass_manager(
            optimization_level=optimization_level, backend=backend
        )
        # Maps id(template_circuit) -> transpiled+measured parametric circuit.
        self._transpiled_cache: dict[int, object] = {}
        # Pending parameter bindings set by bind_parameters, consumed by execute.
        self._pending_bindings: dict | None = None
        self._verbose = verbose
        self._call_count = 0

    def bind_parameters(self, circuit, bindings, parameter_metadata):
        """Transpile the template circuit once (cached), store bindings for execute."""
        cid = id(circuit)
        if cid not in self._transpiled_cache:
            stripped = circuit.remove_final_measurements(inplace=False)
            transpiled = self._pm.run(stripped)
            prepared = self._ensure_measurements(transpiled)
            prepared.metadata[_AER_PREPARED] = True
            self._transpiled_cache[cid] = prepared

        # Store {Qiskit Parameter → float} for the parameter_binds call in execute.
        self._pending_bindings = {
            p.backend_param: bindings[p.name]
            for p in parameter_metadata.parameters
            if p.name in bindings
        }
        # Return the parametric (unbound) compiled circuit — execute will bind
        # via parameter_binds so qiskit-aer sees the same circuit object every call.
        return self._transpiled_cache[cid]

    def execute(self, circuit, shots):
        t0 = time.perf_counter()

        if circuit.metadata.get(_AER_PREPARED) and self._pending_bindings is not None:
            # Hot path: parametric circuit + pending bindings from bind_parameters.
            # parameter_binds lets qiskit-aer handle substitution internally so it
            # can cache the compiled circuit representation across calls.
            result = self._run_backend.run(
                circuit, shots=shots, parameter_binds=[self._pending_bindings]
            ).result()
            self._pending_bindings = None
        else:
            # Cold path: non-parametric circuit sent directly (no bind_parameters).
            if not circuit.metadata.get(_AER_PREPARED):
                circuit = circuit.remove_final_measurements(inplace=False)
                circuit = self._pm.run(circuit)
                circuit = self._ensure_measurements(circuit)
            result = self._run_backend.run(circuit, shots=shots).result()

        t1 = time.perf_counter()
        self._call_count += 1

        if self._verbose and (self._call_count <= 5 or self._call_count % 50 == 0):
            meta = result.results[0].metadata
            sim_ms = result.results[0].time_taken * 1000
            total_ms = (t1 - t0) * 1000
            device = meta.get("device", "?")
            method = meta.get("method", "?")
            print(
                f"[AerLocalExecutor #{self._call_count}] "
                f"total={total_ms:.1f}ms  sim={sim_ms:.2f}ms  "
                f"overhead={total_ms - sim_ms:.1f}ms  "
                f"device={device}  method={method}  "
                f"qubits={circuit.num_qubits}  gates={circuit.size()}"
            )

        return result.get_counts(0)

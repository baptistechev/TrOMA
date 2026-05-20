from qiskit_aer import AerSimulator
from qamomile.qiskit import QiskitExecutor, QiskitTranspiler
from qiskit import transpile
from qiskit_ibm_runtime import SamplerV2, QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

class IBMRuntimeExecutor(QiskitExecutor):
    """QiskitExecutor that runs sampling through a pre-built SamplerV2."""

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
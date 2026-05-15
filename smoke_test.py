from troma import Restriction
from troma.combinatorial_problem import CombinatorialProblem, RestrictedProblem
from troma.core.sampling import restricted_objective_sampling
from troma.sample import Sample

# 1) from troma import Restriction
print(f"Restriction imported: {Restriction is not None}")

# 2) CombinatorialProblem.restrict
problem = CombinatorialProblem(lambda x: 1.0, 2)
rp = problem.restrict(dit_restrictions=[0], dit_value_restrictions=[[0, 1]], additional_dits_val=0)
print(f"rp is RestrictedProblem: {isinstance(rp, RestrictedProblem)}")
print(f"rp.restriction is Restriction: {isinstance(rp.restriction, Restriction)}")

# 3) RestrictedProblem legacy positional
try:
    rp_legacy = RestrictedProblem(problem, [0], [[0, 1]], 0)
    print(f"rp_legacy is RestrictedProblem: {isinstance(rp_legacy, RestrictedProblem)}")
except Exception as e:
    print(f"rp_legacy failed: {e}")

# 4) restricted_objective_sampling
rest = Restriction([0], [[0, 1]], 0)
res_sample = restricted_objective_sampling(lambda x: 1.0, 1, 1, restriction=rest, dit_dimension=2)
# 
print(f"res_sample is Sample: {isinstance(res_sample, Sample)}")

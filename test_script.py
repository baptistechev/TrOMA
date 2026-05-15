import traceback
from troma.combinatorial_problem import CombinatorialProblem, SketchType
from troma.matching_pursuit import matching_pursuit

try:
    f = lambda bits: sum(bits)
    p = CombinatorialProblem(f, problem_size=4, problem_dimension=2)
    rp = p.restrict([1, 2], [0, 1], additional_dits_val=0)
    rp.sampling(30)
    ps = rp.sketching(SketchType.NEAREST_NEIGHBORS, interaction_size=2)
    
    print(f"Type of ps: {type(ps)}")
    for attr in ['problem_size', 'problem_dimension']:
        if hasattr(ps, attr):
            print(f"{attr}: {getattr(ps, attr)}")
        else:
            print(f"{attr}: Not present")
            
    try:
        result = matching_pursuit(ps, iteration_number=2)
        print(f"Matching Pursuit Success: True")
        print(f"Positions: {result}")
    except Exception as e:
        print(f"Matching Pursuit Failed: {e}")
        traceback.print_exc()

except Exception:
    traceback.print_exc()

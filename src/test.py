from util.data import CNFData
from util.solver_wrapper import Forqes
import numpy as np

data = CNFData("test.dimacs", "/home/moriyama/Thesis", split_literals=False)

print(data.edge_attr)
print(data.edge_index)
print(data.getFeatures())

solver = Forqes()
print(solver.solve(data, np.ones(data.n_clauses)))

data.loadMUS(mode="forqes")
print(data.mus, data.mus_bin)
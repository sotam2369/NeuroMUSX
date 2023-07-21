import numpy as np

import subprocess

class Solver():

    def __init__(self):
        pass

    def toCNF(self, cnf_data, used_clauses):
        cnf = ['p cnf ' + str(cnf_data.n_vars) + ' ' + str(int(np.sum(used_clauses))) + '\n']
        for clause in list(np.argwhere(used_clauses == 1)):
            cnf.append(' '.join(list(map(str, cnf_data.formula.soft[clause[0]])) + ['0']) + '\n')
        return cnf

    def getOutput(self, input_cnf, command):
        p = subprocess.Popen(command, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout_data = p.communicate(input=' '.join(input_cnf).encode())
        return stdout_data[0].decode()

class Forqes(Solver):

    def solve(self, cnf_data, used_clauses, forqes_path='./forqes/forqes-linux-x86-64'):
        cnf = self.toCNF(cnf_data, used_clauses)
        lines = self.getOutput(cnf, [forqes_path, '-vv']).split("\n")
        solution = [line.split() for line in lines if len(line) > 0 and line[0] == 'v']
        solution_int = [int(var) for var in solution[0][1:-1]]
        return solution_int

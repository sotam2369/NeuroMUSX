import subprocess
import numpy as np

class Solver():

    def __init__(self):
        pass

    def getCNF(self, cnf_data, unsat_core):
        cnf_final = ['p cnf ' + str(cnf_data.n_var) + ' ' + str(int(np.sum(unsat_core))) + '\n']
        for i in range(cnf_data.n_clause):
            if unsat_core[i] == 1:
                cnf_final.append(' '.join(cnf_data.clauses[i] + ['0']) + '\n')
        return cnf_final


    def createPredFile(self, cnf_data, unsat_core):
        with open("./temp_output/pred_core.cnf", "w+") as pred_core:
            pred_core.writelines(self.getCNF(cnf_data, unsat_core))

    def outputCoreDRAT(self):
        subprocess.call(["~/Thesis/drat-trim/drat-trim ~/Thesis/GNNUnsat/temp_output/pred_core.cnf ~/Thesis/GNNUnsat/temp_output/temp_proof -c  ~/Thesis/GNNUnsat/temp_output/unsat_core.cnf"], 
                        shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        
    def outputCoreMUSer2(self):
        with open("./temp_output/muser2_output.txt", "w+") as output_file:
            subprocess.call(["~/Thesis/muser/src/tools/muser2/muser2 -wf ~/Thesis/GNNUnsat/temp_output/unsat_core ~/Thesis/GNNUnsat/temp_output/pred_core.cnf"], 
                            shell=True, stdout=output_file, stderr=subprocess.STDOUT)


class Forques(Solver):

    def __init__(self):
        pass
    
    def solve(self, cnf_data, unsat_core):
        self.createPredFile(cnf_data, unsat_core)
        
        with open("./temp_output/forques_output.txt", "w+") as output_file:
            #print("Running Forques")
            subprocess.call(["./forqes-linux-x86-64", "-vv", "./temp_output/pred_core.cnf"], 
                            stdout=output_file, stderr=output_file)
            #print("Finished Running Forques")
        
        with open("./temp_output/forques_output.txt", "r") as output_file:
            lines = output_file.readlines()
            solution = [line.split() for line in lines if line[0] in ['v']]
            solution_int = [int(var) for var in solution[0][1:-1]]
            return solution_int

class KissatSolver(Solver):

    def __init__(self):
        pass

    def solve(self, cnf_data, unsat_core):
        self.createPredFile(cnf_data, unsat_core)

        with open("./temp_output/kissat_output.txt", "w+") as output_file:
            subprocess.call(["~/Thesis/kissat/build/kissat ~/Thesis/GNNUnsat/temp_output/pred_core.cnf ~/Thesis/GNNUnsat/temp_output/temp_proof"], 
                            shell=True, stdout=output_file, stderr=subprocess.STDOUT)

        with open("./temp_output/kissat_output.txt", "r") as output_file:
            lines = output_file.readlines()
            solution = [line.strip().split() for line in lines if line[0] in ['s']]
            if solution[0][1] == "UNSATISFIABLE":
                return 0
            else:
                return 1
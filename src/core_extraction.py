from z3 import *
import pickle
import numpy
from tqdm import tqdm
from CNF_Data import CNF_Data
import signal

def raiseException(signum, frame):
    raise Exception()

def loadCNF(path):
    n_var = 0
    n_clause = 0
    with open(path, 'r') as cnf_file:
        lines = cnf_file.readlines()
        n_var = int(lines[0].split()[-2])
        n_clause = int(lines[0].split()[-1])
        clauses = [line.strip().split()[:-1] for line in lines if line[0] not in ['c', 'p']]
    return [clauses, n_clause, n_var]


if __name__ == "__main__":
    core_type = int(input("1. Train \n2. Test \n3. Single Test\n4. SATComp\nUnsat Cores to load: "))

    cnf_data_list = []
    test = True
    root_folder = "../sr1-40_full/"

    if core_type == 1:
        print("Mode: Train")
        for cnf_file in tqdm(os.listdir(root_folder + "train/sr1-40/")):
            if 'sat=0' in cnf_file:
                cnf_data_list.append(CNF_Data(root_folder + "train/sr1-40/" + cnf_file, cnf_file))

        #for cnf_file in tqdm(os.listdir(root_folder + "train/sr5/grp2")):
        #    if 'sat=0' in cnf_file:
        #        cnf_data_list.append(CNF_Data(root_folder + "train/sr5/grp2/" + cnf_file, cnf_file))
                
    elif core_type == 2:
        print("Mode: Test")
        for cnf_file in tqdm(os.listdir(root_folder + "test/sr1-40/")):
            if 'sat=0' in cnf_file:
                cnf_data_list.append(CNF_Data(root_folder + "test/sr1-40/" + cnf_file, cnf_file))

        #for cnf_file in tqdm(os.listdir(root_folder + "test/sr5/grp2")):
        #    if 'sat=0' in cnf_file:
        #        cnf_data_list.append(CNF_Data(root_folder + "test/sr5/grp2/" + cnf_file, cnf_file))
    elif core_type == 4:
        SAT_COMP_PART = input("SATComp Part: ")
        i = 1
        for cnf_file in tqdm(os.listdir("satcomp2022/part" + SAT_COMP_PART)):
            unsat_cores = []

            cnf_data = CNF_Data("satcomp2022/part" + SAT_COMP_PART +  "/" + cnf_file, cnf_file)
            
            signal.signal(signal.SIGALRM, raiseException)
            signal.alarm(10)

            try:
                cnf_data.loadMinimalMUSForques()
                print("Loaded OMUS of size: " + str(numpy.sum(cnf_data.unsat_cores[-1])) + " for " + cnf_file)
            except:
                print("Failed to load OMUS")
                cnf_data.loadMUS()
                print("Loaded Normal MUS")
            
            unsat_cores.append(cnf_data)
            with open('extracted_cores/unsat_cores_satcomp(' + SAT_COMP_PART + '-' + str(i) + ').pickle', 'wb') as handle:
                pickle.dump(unsat_cores, handle)
            i += 1
        exit()
    elif core_type == 5:
        print("Mode: Train")
        for cnf_file in tqdm(os.listdir(root_folder + "train/sr1-40/")):
            if 'sat=1' in cnf_file:
                cnf_data_list.append(CNF_Data(root_folder + "train/sr1-40/" + cnf_file, cnf_file))
        unsat_cores = []
        for cnf_data in tqdm(cnf_data_list):
            cnf_data.unsat_cores = [numpy.zeros(cnf_data.n_var+cnf_data.n_clause)]
            unsat_cores.append(cnf_data)
        with open('extracted_cores/unsat_cores_train_40_MMUS_sat.pickle', 'wb') as handle:
            pickle.dump(unsat_cores, handle)
        exit()
        #for cnf_file in tqdm(os.listdir(root_folder + "train/sr5/grp2")):
        #    if 'sat=0' in cnf_file:
        #        cnf_data_list.append(CNF_Data(root_folder + "train/sr5/grp2/" + cnf_file, cnf_file))
                
    elif core_type == 6:
        print("Mode: Test")
        for cnf_file in tqdm(os.listdir(root_folder + "test/sr1-40/")):
            if 'sat=1' in cnf_file:
                cnf_data_list.append(CNF_Data(root_folder + "test/sr1-40/" + cnf_file, cnf_file))
        unsat_cores = []
        for cnf_data in tqdm(cnf_data_list):
            cnf_data.unsat_cores = [numpy.zeros(cnf_data.n_var+cnf_data.n_clause)]
            unsat_cores.append(cnf_data)

        with open('extracted_cores/unsat_cores_test_40_MMUS_sat.pickle', 'wb') as handle:
            pickle.dump(unsat_cores, handle)
        exit()
    else:
        print("Mode: Single Test")
        cnf_data_list.append(CNF_Data("test.dimacs", "test.dimacs"))

    unsat_cores = []
    for cnf_data in tqdm(cnf_data_list):

        if core_type == 2:
            cnf_data.loadMUS()
            print("Loaded Normal MUS")
        else:
            signal.signal(signal.SIGALRM, raiseException)
            signal.alarm(60)
            try:
                cnf_data.loadMinimalMUSForques()
                print("Loaded OMUS")
            except:
                print("Failed to load OMUS:", cnf_data.prob_file)
                cnf_data.loadMUS()
                print("Loaded Normal MUS")

        unsat_cores.append(cnf_data)

        if len(unsat_cores) % 1000 == 0:
            if core_type == 1:
                with open('extracted_cores/unsat_cores_train_1-40full_MMUS.pickle', 'wb') as handle:
                    pickle.dump(unsat_cores, handle)
            elif core_type == 2:
                with open('extracted_cores/unsat_cores_test_1-40full_MMUS.pickle', 'wb') as handle:
                    pickle.dump(unsat_cores, handle)
            elif core_type == 4:
                with open('extracted_cores/unsat_cores_satcomp(' + SAT_COMP_PART + ').pickle', 'wb') as handle:
                    pickle.dump(unsat_cores, handle)
            else:
                with open('extracted_cores/unsat_cores_test_single.pickle', 'wb') as handle:
                    pickle.dump(unsat_cores, handle)
            print("Saved Cores")
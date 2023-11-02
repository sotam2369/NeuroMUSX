import pickle
import os

if __name__ == "__main__":
    split = 10

    os.chdir('../src')

    output_list_unsat = []
    output_list_sat = []
    file_name = "test"
    for i in range(split):
        with open("../dataset/processed/" + file_name + ".pkl.part{}".format(i+1), "rb") as f:
            loaded = pickle.load(f)
            print("Loading...{0}/{1}".format(i+1, split))
            output_list_unsat += loaded[0]
            output_list_sat += loaded[1]
    
    with open("../dataset/processed/{}.pkl".format(file_name), "wb") as f:
        pickle.dump([output_list_unsat, output_list_sat], f)
from pysat.examples.musx import MUSX
from pysat.formula import CNF
from tqdm import tqdm
import itertools

def writeHeader(file):
    file.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
    file.write("<graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xsi:schemaLocation=\"http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd\">\n")
    file.write("<graph id=\"CNF\" edgedefault=\"undirected\">\n")
    file.write("<key id=\"n_variable_id\" for=\"node\" attr.name=\"variable_id\" attr.type=\"int\"/>\n")
    file.write("<key id=\"unsat\" for=\"edge\" attr.name=\"color\" attr.type=\"string\"/>\n")

def writeNode(file, id):
    file.write("<node id=\"" + str(id) + "\">\n")
    file.write("<data key=\"n_variable_id\">" + str(id) + "</data>\n")
    file.write("</node>\n")

def writeEdge(file, id1, id2, unsat):
    file.write("<edge source=\"" + str(id1) + "\" target=\"" + str(id2) + "\">\n")
    if unsat:
        file.write("<data key=\"unsat\">red</data>\n")
    else:
        file.write("<data key=\"unsat\">green</data>\n")
    file.write("</edge>\n")

def writeFooter(file):
    file.write("</graph>\n")
    file.write("</graphml>\n")


def writeGraph(file, mus, cnf):
    writeHeader(file)
    for i in range(1,cnf.nv+1):
        writeNode(file, i)
    edges_written = []
    i = 1
    for clause in tqdm(cnf.soft):
        clause_abs = [abs(ele) for ele in clause]
        for edge in itertools.combinations(sorted(clause_abs), 2):
            edge_list = list(edge)
            if edge_list not in edges_written:
                edges_written.append(edge_list)
                writeEdge(file, edge_list[0], edge_list[1], i in mus)
        i += 1
    writeFooter(file)


if __name__ == '__main__':
    file_name = input("File name: ")

    cnf = CNF(from_file=file_name + '.cnf').weighted()
    with MUSX(cnf) as musx:
        mus = musx.compute()

    print(len(mus))
    with open(file_name + "_mus.graphml", 'w+') as file:
        writeGraph(file, mus, cnf)
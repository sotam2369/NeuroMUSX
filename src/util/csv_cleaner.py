import csv

if __name__ == "__main__":
    with open("../../log/output_random.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)
        data = list(reader)
    
    with open("../../log/output_random_cleaned.csv", "w") as f:
        writer = csv.writer(f)
        #../satcomp2011/SAT11/mus/marques-silva/equivalence-checking/c5315-s.cnf
        header = ['File','Problem Set','ID','Normal MUS Size','Normal Oracle Time','GNN MUS Size','GNN Oracle Time']
        writer.writerow(header)
        for row in data:
            row.insert(1, row[0].split('/')[-2])
            row.insert(2, row[0].split('/')[-1].split('.')[0])
            writer.writerow(row)
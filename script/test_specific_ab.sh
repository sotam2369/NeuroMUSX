rm -f ../log/output_ab.csv
rm -f ../log/output_ab_log.txt

i2=0
find ../satcomp2011/SAT11/mus/marques-silva/abstraction-refinement-intel/ -name "*.cnf" -print0 | while read -d $'\0' file
do
    i2=$((i2+1))
    echo -ne "Currently: $i2"\\r
    python ../src/musx_gnn.py $file --model=model_ab.pt --output=output_ab.csv >> ../log/output_ab_log.txt
done
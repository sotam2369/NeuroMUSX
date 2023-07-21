rm -f ../log/output_dd.csv
rm -f ../log/output_dd_log.txt

i2=0
find ../satcomp2011/SAT11/mus/marques-silva/design-debugging/ -name "*.cnf" -print0 | while read -d $'\0' file
do
    i2=$((i2+1))
    echo -ne "Currently: $i2"\\r
    python ../src/musx_gnn.py $file --model=model_dd.pt --output=output_dd.csv >> ../log/output_dd_log.txt
done
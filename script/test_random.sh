rm -f ../log/output_random.csv
rm -f ../log/output_random_log.txt

i2=0
find ../satcomp2011/SAT11 -name "*.cnf" -print0 | while read -d $'\0' file
do
    i2=$((i2+1))
    echo -ne "Currently: $i2"\\r
    python ../src/musx_gnn.py $file --model=model_random.pt --output=output_random.csv >> ../log/output_random_log.txt
done
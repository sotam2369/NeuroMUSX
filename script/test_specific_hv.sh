rm -f ../log/output_hv.csv
rm -f ../log/output_hv_log.txt

i2=0
find ../satcomp2011/SAT11/mus/marques-silva/hardware-verification/ -name "*.cnf" -print0 | while read -d $'\0' file
do
    i2=$((i2+1))
    echo -ne "Currently: $i2"\\r
    python ../src/musx_gnn.py $file --model=model_hv.pt --output=output_hv.csv >> ../log/output_hv_log.txt
done
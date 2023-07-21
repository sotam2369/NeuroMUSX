rm -f ../log/output_app.csv
rm -f ../log/output_app_log.txt

i2=0
find ../satcomp2011/SAT11/application/ -name "*.cnf" -print0 | while read -d $'\0' file
do
    i2=$((i2+1))
    echo -ne "Currently: $i2"\\r
    python ../src/musx_gnn.py $file --model=model_app.pt --output=output_app.csv >> ../log/output_app_log.txt
done
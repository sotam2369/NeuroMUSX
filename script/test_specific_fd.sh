rm -f ../log/output_fd.csv
rm -f ../log/output_fd_log.txt

i2=0
find ../satcomp2011/SAT11/mus/chen/fdmus_v100/ -name "*.cnf" -print0 | while read -d $'\0' file
do
    i2=$((i2+1))
    echo -ne "Currently: $i2"\\r
    python ../src/musx_gnn.py $file --model=model_fd.pt --output=output_fd.csv >> ../log/output_fd_log.txt
done
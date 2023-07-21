rm -rf ../satcomp2011
mkdir ../satcomp2011
wget -P ../satcomp2011 http://www.cril.univ-artois.fr/SAT11/bench/SAT11-Competition-MUS-SelectedBenchmarks.tar
tar -xf ../satcomp2011/SAT11-Competition-MUS-SelectedBenchmarks.tar -C ../satcomp2011
rm -f ../satcomp2011/SAT11-Competition-MUS-SelectedBenchmarks.tar
rm -rf ../satcomp2011/SAT02
rm -rf ../satcomp2011/SAT03
rm -rf ../satcomp2011/SAT04
rm -rf ../satcomp2011/SAT07
rm -rf ../satcomp2011/SAT09


i2=0
find ../satcomp2011/SAT11 -name "*.bz2" -print0 | while read -d $'\0' file
do
    i2=$((i2+1))
    echo -ne "Currently: $i2"\\r
    bzip2 -d $file
done
echo "Finished!"
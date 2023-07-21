for ty in "train" "test" ; do
    rm -rf ../dataset/original/$ty/
    mkdir -p ../dataset/original/$ty/
	python3 ../src/data_generator.py ../dataset/original/$ty/ 10000 --min_n 3 --max_n 40
done;
rm -rf ../dataset/processed/
mkdir -p ../dataset/processed/

for ty in "train" "test" ; do
	python3 ../src/core_extractor.py -dir ../dataset/original/$ty/ -out ../dataset/processed/$ty.pkl \
    -r True
done;
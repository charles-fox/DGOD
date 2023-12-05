mkdir -p Annots
python3 json2csv_bdd10k.py --image_set train
echo BDD10K train annotations are converted
python3 json2csv_bdd10k.py --image_set val
echo BDD10K val annotations are converted
python3 json2csv_cityscapes.py --category all
echo Cityscapes full annotations are converted
python3 json2csv_acdc.py --image_set train
echo ACDC full train annotations are converted
python3 json2csv_acdc.py --image_set val
echo ACDC full val annotations are converted
python3 json2csv_idd.py --image_set train
echo IDD full train annotations are converted
python3 json2csv_idd.py --image_set val
echo IDD full val annotations are converted

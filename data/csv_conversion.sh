python json2csv_bdd100k_full.py
echo BDD100K annotations are converted
python json2csv_cityscapes.py --category all
echo Cityscapes full annotations are converted
python json2csv_acdc.py --image_set train
echo ACDC full train annotations are converted
python json2csv_acdc.py --image_set val
echo ACDC full val annotations are converted
python json2csv_idd.py --image_set train
echo IDD full train annotations are converted
python json2csv_idd.py --image_set val
echo IDD full val annotations are converted

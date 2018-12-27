
import csv
import os
from Constants import TPS_DIR, CATEGORY
from MetaDataReaderProcess import clean_str

y_train_tip = []
f_train = csv.reader(open(os.path.join(TPS_DIR, CATEGORY + '_train.csv'), "r", encoding='utf-8'))
for line in f_train:
    y_train_tip.append(clean_str(line[3]).split(" "))

y_valid_tip = []
f_valid = csv.reader(open(os.path.join(TPS_DIR, CATEGORY + '_valid.csv'), "r", encoding = 'utf-8'))
for line in f_valid:
    y_valid_tip.append(clean_str(line[3]).split(" "))

tip_voc = [xx for x in y_train_tip + y_valid_tip for xx in x]  # tip词集
print(len(tip_voc))
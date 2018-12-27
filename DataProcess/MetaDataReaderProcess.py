import pandas as pd
import numpy as np
import pickle
import os
import gzip
import demjson
import re
from collections import defaultdict
import sys
sys.path.append('./')
from Constants import TPS_DIR, CATEGORY_NUM, TITLE_LEN, META_DIR


def meta_file_reader(TP_file):
    f= gzip.open(TP_file, 'r')
    items_id, categories, description, title =[], [], [], []

    for line in f:
        js = demjson.decode(line)
        if str(js['asin']) == 'unknown':
            continue

        try:
            categories.append(js['categories'])
        except:
            categories.append('')

        try:
            description.append(js['description'])
        except:
            description.append('')

        try:
            title.append(str(js['title']))
        except:
            title.append('')
        items_id.append(str(js['asin']) + ',')

    meta_data = pd.DataFrame({ 'item_id': pd.Series(items_id),
                               'categories': pd.Series(categories),
                               'description': pd.Series(description),
                               'title': pd.Series(title)})[['item_id','categories', 'description', 'title']]
    return meta_data


def numerize_meta(tp, item2id):
    sid = []
    for term in tp['item_id']:
        if term in item2id.keys():
            sid.append(item2id[term])
        else:
            sid.append(-1)
    tp['item_id'] = sid
    return tp

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def pad_sentences(description, description_len, padding_word="<PAD/>"):
    if description_len > len(description):
        num_padding = description_len - len(description)
        new_sentence = description + [padding_word] * num_padding
    else:
        new_sentence = description[: description_len]
    return new_sentence

def title_description_process(sentence_list, len_proportion):
    sentence_len_list = np.array([len(x.split(' ')) for x in sentence_list])
    X = np.sort(sentence_len_list)
    sentence_len = X[int(len_proportion * len(sentence_len_list)) - 1]
    sentence_list = [pad_sentences(x.split(' '), sentence_len) for x in sentence_list]
    sentence_voc = set([xx for x in sentence_list for xx in x])  # description词集
    vocabulary_sentence = dict([(voc, i) for i, voc in enumerate(sentence_voc)])
    sentence_list = [[vocabulary_sentence[word] for word in words] for words in sentence_list]
    return sentence_len, vocabulary_sentence, sentence_list

def MetaDataProcess(item_infor):
    category_num = defaultdict(int)
    category, description, title = [], [], []
    for key, item in item_infor.items():
        for x in item[1]:
            for xx in x:
                category_num[xx] += 1
        category.append(item[1])
        description.append(clean_str(item[2]))
        title.append(clean_str(item[3]))

    category_num = sorted(category_num.items(), key = lambda x : x[1], reverse = True)
    category_select = [cate for cate, num in category_num[0 : CATEGORY_NUM] ]
    category2id = dict([(cate, i) for i, cate in enumerate(category_select)])
    for num, x in enumerate(category):
        vec = [0] * len(category_select)
        for xx in x:
            for xxx in xx:
                if xxx in category_select:
                    vec[category2id[xxx]] = 1
        category[num] = vec

    title_len, title_voc, title = title_description_process(title, TITLE_LEN)
    item_infor_process = {}
    for num, key in enumerate(item_infor.keys()):
        item_infor_process[key] = [category[num], title[num]] # , description[num]

    para = {}
    para['category2id'] = category2id
    para['categorynum'] = len(category2id)
    para['title_voc'] = title_voc
    para['title_len'] = title_len
    para['item_infor'] = item_infor_process
    pickle.dump(para, open(os.path.join(TPS_DIR, 'item_infor.para'), 'wb'))


if __name__ == "__main__":
    TP_file = os.path.join(TPS_DIR, META_DIR)
    item2id = pickle.load(open(os.path.join(TPS_DIR, 'item2id'), 'rb'))
    item_reviews = pickle.load(open(os.path.join(TPS_DIR, 'item_review'), 'rb')).keys()
    meta_data = meta_file_reader(TP_file)
    meta_data = numerize_meta(meta_data, item2id)
    meta_data =meta_data[~meta_data['item_id'].isin([-1])]

    item_infor = {}
    for item in item_reviews:
        tmp_list = np.array(meta_data[meta_data['item_id'] == item]).tolist()
        if tmp_list:
            item_infor[item] = tmp_list[0]

    pickle.dump(item_infor, open(os.path.join(TPS_DIR, 'item_infor'), 'wb'))
    # item_infor = pickle.load(open(os.path.join(TPS_DIR, 'item_infor'), 'rb'))
    MetaDataProcess(item_infor)



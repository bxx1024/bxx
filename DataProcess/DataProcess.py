import itertools
import pickle
import os
import csv
import tensorflow as tf
import numpy as np
from collections import Counter
import sys
sys.path.append('./')
from Constants import TPS_DIR, CATEGORY, REVIEW_NUM, REVIEW_LEN
from MetaDataReaderProcess import clean_str
print("import success...")

tf.flags.DEFINE_string("train_data", os.path.join(TPS_DIR, CATEGORY + '_train.csv'), "Data for training")
tf.flags.DEFINE_string("valid_data", os.path.join(TPS_DIR, CATEGORY + '_valid.csv'), " Data for validation")
tf.flags.DEFINE_string("test_data", os.path.join(TPS_DIR, CATEGORY + '_test.csv'), "Data for testing")
tf.flags.DEFINE_string("user_review", os.path.join(TPS_DIR, 'user_review'), "User's reviews")
tf.flags.DEFINE_string("item_review", os.path.join(TPS_DIR, 'item_review'), "Item's reviews")
tf.flags.DEFINE_string("item_infor", os.path.join(TPS_DIR, 'item_infor.para'), "Item's infor")
tf.flags.DEFINE_string("user_review_id", os.path.join(TPS_DIR, 'user_rid'), "user_review_id")
tf.flags.DEFINE_string("item_review_id", os.path.join(TPS_DIR, 'item_rid'), "item_review_id")


def pad_sentences(u_text, u_len, u2_len, padding_word="<PAD/>"):
    u_text2 = {}
    for i in u_text.keys():
        u_reviews = u_text[i]
        padded_u_train = []
        for ri in range(u_len):
            if ri < len(u_reviews):
                sentence = u_reviews[ri]
                if u2_len > len(sentence):
                    num_padding = u2_len - len(sentence)
                    new_sentence = sentence + [padding_word] * num_padding
                    padded_u_train.append(new_sentence)
                else:
                    new_sentence = sentence[:u2_len]
                    padded_u_train.append(new_sentence)
            else:
                new_sentence = [padding_word] * u2_len
                padded_u_train.append(new_sentence)
        u_text2[i] = padded_u_train
    return u_text2

def pad_tips(sentences, max_len, padding_word="<PAD/>"):
    result = []
    for sentence in sentences:
        if max_len > len(sentence):
            num_padding = max_len - len(sentence)
            new_sentence = sentence + [padding_word] * num_padding
            result.append(new_sentence)
        else:
            new_sentence = sentence[:max_len]
            result.append(new_sentence)
    return result

def pad_reviewid(u_train, u_len, num):
    pad_u_train = []
    for i in range(len(u_train)):
        x = u_train[i]
        while u_len > len(x):
            x.append(num)
        if u_len < len(x):
            x = x[:u_len]
        pad_u_train.append(x)
    return pad_u_train


def build_vocab(sentences):
    word_counts = Counter(itertools.chain(*sentences))
    vocabulary_inv = [x[0] for x in word_counts.most_common()] # 返回一个list, list中包含Counter对象中出现最多前n个元素
    vocabulary_inv = list(sorted(vocabulary_inv))
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(u_text, vocabulary_u):
    u_text2 = {}
    for i in u_text.keys():
        u_reviews = u_text[i]
        u = np.array([[vocabulary_u[word] for word in words] for words in u_reviews])
        u_text2[i] = u
    return u_text2


def load_data(train_data, valid_data, user_review, item_review,item_infor, user_rid, item_rid):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    u_text, i_text, y_train, y_valid, y_train_tip, y_valid_tip, \
    user_num, item_num, u_len, i_len, u2_len, i2_len, uid_train, iid_train, uid_valid, iid_valid, \
    reid_user_train, reid_item_train, reid_user_valid, reid_item_valid \
    = load_data_and_labels(train_data, valid_data, user_review, item_review, item_infor, user_rid, item_rid)
    print("load data done")

    u_text = pad_sentences(u_text, u_len, u2_len)
    reid_user_train = pad_reviewid(reid_user_train, u_len, item_num + 1)
    reid_user_valid = pad_reviewid(reid_user_valid, u_len, item_num + 1)
    print("pad user done")

    i_text = pad_sentences(i_text, i_len, i2_len)
    reid_item_train = pad_reviewid(reid_item_train, i_len, user_num + 1)
    reid_item_valid = pad_reviewid(reid_item_valid, i_len, user_num + 1)
    print("pad item done")

    max_tip_len = sorted([len(x) for x in y_train_tip + y_valid_tip])[-1] # tip_len
    y_train_tip = pad_tips(y_train_tip, max_tip_len)
    y_valid_tip = pad_tips(y_valid_tip, max_tip_len)
    print("pad tip done")

    user_voc = [xx for x in u_text.values() for xx in x]  # 用户词集
    item_voc = [xx for x in i_text.values() for xx in x]  # 商品词集
    tip_voc = [x for x in y_train_tip + y_valid_tip]  # tip词集


    vocabulary_user, vocabulary_inv_user = build_vocab(user_voc)
    vocabulary_item, vocabulary_inv_item = build_vocab(item_voc)
    vocabulary_tip, vocabulary_inv_tip = build_vocab(tip_voc)
    u_text = build_input_data(u_text, vocabulary_user)
    i_text = build_input_data(i_text, vocabulary_item)

    y_train = np.array(y_train)
    y_valid = np.array(y_valid)
    y_train_tip = np.array([[vocabulary_tip[word] for word in words] for words in y_train_tip])
    y_valid_tip = np.array([[vocabulary_tip[word] for word in words] for words in y_valid_tip])
    uid_train = np.array(uid_train)
    uid_valid = np.array(uid_valid)
    iid_train = np.array(iid_train)
    iid_valid = np.array(iid_valid)
    reid_user_train = np.array(reid_user_train)
    reid_user_valid = np.array(reid_user_valid)
    reid_item_train = np.array(reid_item_train)
    reid_item_valid = np.array(reid_item_valid)

    return [u_text, i_text, y_train, y_valid, y_train_tip, y_valid_tip,
            vocabulary_user,vocabulary_item, vocabulary_tip, max_tip_len,
            uid_train, iid_train, uid_valid, iid_valid, user_num, item_num,
            reid_user_train, reid_item_train, reid_user_valid, reid_item_valid,]


def load_data_and_labels(train_data, valid_data, user_review, item_review,item_infor, user_rid, item_rid):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    with open(user_review, "rb") as f1, open(item_review, "rb") as f2, \
          open(user_rid, "rb") as f3, open(item_rid, "rb") as f4:
        user_reviews = pickle.load(f1)
        item_reviews = pickle.load(f2)
        user_rids = pickle.load(f3)
        item_rids = pickle.load(f4)

    print("training...")
    reid_user_train, reid_item_train = [], []
    uid_train, iid_train = [], []  # item id
    y_train, y_train_tip, y_train_infor = [], [], []
    u_text, u_rid = {},{}
    i_text, i_rid = {},{}

    f_train = csv.reader(open(train_data, "r", encoding = 'utf-8'))
    for line in f_train:
        uid_train.append(int(line[0]))
        iid_train.append(int(line[1]))
        if not int(line[0]) in u_text:
            u_text[int(line[0])] = [clean_str(s).split(" ") for s in user_reviews[int(line[0])]]   # preprocess each review
            u_rid[int(line[0])] = [int(s) for s in user_rids[int(line[0])]]
        reid_user_train.append(u_rid[int(line[0])])

        if not int(line[1]) in i_text:
            i_text[int(line[1])] = [clean_str(s).split(" ") for s in item_reviews[int(line[1])]]
            i_rid[int(line[1])] = [int(s) for s in item_rids[int(line[1])]]
        reid_item_train.append(i_rid[int(line[1])])
        y_train.append(float(line[2]))
        y_train_tip.append(clean_str(line[3]).split(" "))

    print("validing...")
    reid_user_valid, reid_item_valid = [], []
    uid_valid, iid_valid = [], []
    y_valid, y_valid_tip, y_valid_infor = [], [], []

    f_valid = csv.reader(open(valid_data, "r", encoding = 'utf-8'))
    for line in f_valid:
        uid_valid.append(int(line[0]))
        iid_valid.append(int(line[1]))
        if int(line[0]) in u_text:
            reid_user_valid.append(u_rid[int(line[0])])
        else: # 训练集中不存在
            u_text[int(line[0])] = [['<PAD/>']]
            u_rid[int(line[0])] = [int(0)]
            reid_user_valid.append(u_rid[int(line[0])])

        if int(line[1]) in i_text:
            reid_item_valid.append(i_rid[int(line[1])])
        else: # 训练集中不存在
            i_text[int(line[1])] = [['<PAD/>']]
            i_rid[int(line[1])] = [int(0)]
            reid_item_valid.append(i_rid[int(line[1])])
        y_valid.append(float(line[2]))
        y_valid_tip.append(clean_str(line[3]).split(" "))


    review_num_u = np.array([len(u) for u in u_text.values()])  # 用户对应的评价个数
    x = np.sort(review_num_u)
    u_len = x[int(REVIEW_NUM * len(review_num_u)) - 1]
    review_len_u = np.array([len(r) for u in u_text.values() for r in u]) # 用户评价的长度
    x2 = np.sort(review_len_u)
    u2_len = x2[int(REVIEW_LEN * len(review_len_u)) - 1]

    review_num_i = np.array([len(i) for i in i_text.values()])
    y = np.sort(review_num_i)
    i_len = y[int(REVIEW_NUM * len(review_num_i)) - 1]
    review_len_i = np.array([len(r) for i in i_text.values() for r in i])
    y2 = np.sort(review_len_i)
    i2_len = y2[int(REVIEW_LEN * len(review_len_i)) - 1] # 最后一个

    user_num = len(u_text)
    item_num = len(i_text)

    print("u_len:", u_len)
    print("i_len:", i_len)
    print("u2_len:", u2_len)
    print("i2_len:", i2_len)
    print("user_num:", user_num)
    print("item_num:", item_num)
    return [u_text, i_text, y_train, y_valid, y_train_tip, y_valid_tip, user_num, item_num,
             u_len, i_len, u2_len, i2_len, uid_train, iid_train, uid_valid, iid_valid,
             reid_user_train, reid_item_train, reid_user_valid, reid_item_valid]


if __name__ == '__main__':
    FLAGS = tf.flags.FLAGS
    FLAGS.flag_values_dict()

    u_text, i_text, y_train, y_valid, y_train_tip, y_valid_tip,\
    vocabulary_user, vocabulary_item, vocabulary_tip, max_tip_len,\
    uid_train, iid_train, uid_valid, iid_valid, user_num, item_num, \
    reid_user_train, reid_item_train, reid_user_valid, reid_item_valid, \
    = load_data(FLAGS.train_data, FLAGS.valid_data, FLAGS.user_review,
                FLAGS.item_review, FLAGS.item_infor,  FLAGS.user_review_id,
                FLAGS.item_review_id)

    np.random.seed(2018)
    shuffle_indices = np.random.permutation(np.arange(len(y_train))) # 生成一个随机序列
    userid_train = uid_train[shuffle_indices]
    itemid_train = iid_train[shuffle_indices]
    y_train = y_train[shuffle_indices]
    y_train_tip = y_train_tip[shuffle_indices]
    reid_user_train = reid_user_train[shuffle_indices]
    reid_item_train = reid_item_train[shuffle_indices]

    y_train = y_train[:, np.newaxis] # 插入新维度
    userid_train = userid_train[:, np.newaxis]
    itemid_train = itemid_train[:, np.newaxis]

    y_valid = y_valid[:, np.newaxis]
    userid_valid = uid_valid[:, np.newaxis]
    itemid_valid = iid_valid[:, np.newaxis]

    batches_train = list(zip(userid_train, itemid_train, reid_user_train, reid_item_train, y_train, y_train_tip))
    batches_test = list(zip(userid_valid, itemid_valid, reid_user_valid, reid_item_valid, y_valid, y_valid_tip))
    print('write begining...')

    pickle.dump(batches_train, open(os.path.join(TPS_DIR, CATEGORY + '.train'), 'wb'))
    pickle.dump(batches_test, open(os.path.join(TPS_DIR, CATEGORY + '.test'), 'wb'))

    para = {}
    para['user_num'] = user_num
    para['item_num'] = item_num
    para['review_num_u'] = u_text[0].shape[0]
    para['review_num_i'] = i_text[0].shape[0]
    para['review_len_u'] = u_text[1].shape[1]
    para['review_len_i'] = i_text[1].shape[1]
    para['max_tip_len'] = max_tip_len
    para['user_vocab'] = vocabulary_user
    para['item_vocab'] = vocabulary_item
    para['tip_vocab'] = vocabulary_tip
    para['train_length'] = len(y_train)
    para['test_length'] = len(y_valid)
    para['u_text'] = u_text
    para['i_text'] = i_text
    pickle.dump(para, open(os.path.join(TPS_DIR, CATEGORY + '.para'), 'wb'))








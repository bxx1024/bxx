import os
import gzip
import demjson
import pickle
import numpy as np
import pandas as pd
import sys
sys.path.append('./')
from Constants import TPS_DIR, REVIEW_DIR, CATEGORY

def review_file_reader(TP_file):
    f= gzip.open(TP_file, 'r')
    users_id, items_id, ratings, summary, reviews=[], [], [], [], []
    for line in f:
        js = demjson.decode(line)
        if str(js['reviewerID']) == 'unknown':
            continue
        if str(js['asin']) == 'unknown':
            continue
        summary.append(js['summary'])
        reviews.append(js['reviewText'])
        users_id.append(str(js['reviewerID'])+',')
        items_id.append(str(js['asin'])+',')
        ratings.append(str(js['overall']))

    review_data = pd.DataFrame({ 'user_id': pd.Series(users_id),
                                 'item_id': pd.Series(items_id),
                                 'ratings': pd.Series(ratings),
                                 'summary': pd.Series(summary),
                                 'reviews': pd.Series(reviews)})[['user_id','item_id','ratings','summary','reviews']]
    return review_data

data = review_file_reader(os.path.join(TPS_DIR, REVIEW_DIR))
def get_count(tp, id):
    playcount_groupbyid = tp[[id, 'ratings']].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count
usercount, itemcount = get_count(data, 'user_id'), get_count(data, 'item_id')
unique_uid = usercount.index
unique_sid = itemcount.index
user2id = dict((uid, i) for (i, uid) in enumerate(unique_uid))
item2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))

def numerize(tp):
    uid = map(lambda x: user2id[x], tp['user_id'])
    sid = map(lambda x: item2id[x], tp['item_id'])
    tp['user_id'] = list(uid)
    tp['item_id'] = list(sid)
    return tp

data = numerize(data)
tp_rating = data[['user_id', 'item_id', 'ratings', 'summary']]

#split train, valid and test (80%, 10%, 10%)
n_ratings = tp_rating.shape[0]
test = np.random.choice(n_ratings, size=int(0.20 * n_ratings), replace = False)
test_idx = np.zeros(n_ratings, dtype=bool)
test_idx[test] = True

tp_1 = tp_rating[test_idx]
tp_train = tp_rating[~test_idx]

data2 = data[test_idx]
data = data[~test_idx]  #train

n_ratings = tp_1.shape[0]
test = np.random.choice(n_ratings, size = int(0.50 * n_ratings), replace = False)
test_idx = np.zeros(n_ratings, dtype = bool)
test_idx[test] = True

tp_test = tp_1[test_idx]
tp_valid = tp_1[~test_idx]

# get user reviews, item reviews and infor for train and test
user_reviews, item_reviews, user_rid, item_rid ={}, {}, {}, {}
for i in data.values:
    if i[0] in user_reviews:
        user_reviews[i[0]].append(i[4])
        user_rid[i[0]].append(i[1])
    else:
        user_reviews[i[0]] = [i[4]]
        user_rid[i[0]] = [i[1]]

    if i[1] in item_reviews:
        item_reviews[i[1]].append(i[4])
        item_rid[i[1]].append(i[0])
    else:
        item_reviews[i[1]] = [i[4]]
        item_rid[i[1]] = [i[0]]

# delete
for i in data2.values:
    if i[0] not in user_reviews:
        user_rid[i[0]]=[0]
        user_reviews[i[0]]=['0']
    if i[1] not in item_reviews:
        item_reviews[i[1]] = [0]
        item_rid[i[1]]=['0']

tp_train.to_csv(os.path.join(TPS_DIR, CATEGORY + '_train.csv'), index = False, header = None)
tp_valid.to_csv(os.path.join(TPS_DIR, CATEGORY + '_valid.csv'), index = False, header = None)
tp_test.to_csv(os.path.join(TPS_DIR, CATEGORY + '_test.csv'), index = False, header = None)
pickle.dump(user_reviews, open(os.path.join(TPS_DIR, 'user_review'), 'wb'))
pickle.dump(item_reviews, open(os.path.join(TPS_DIR, 'item_review'), 'wb'))
pickle.dump(user_rid, open(os.path.join(TPS_DIR, 'user_rid'), 'wb'))
pickle.dump(item_rid, open(os.path.join(TPS_DIR, 'item_rid'), 'wb'))
pickle.dump(user2id, open(os.path.join(TPS_DIR, 'user2id'), 'wb'))
pickle.dump(item2id, open(os.path.join(TPS_DIR, 'item2id'), 'wb'))

print(usercount, itemcount)

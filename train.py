import numpy as np
import pickle
import os
from Model import Model
import tensorflow as tf
from tensorflow.python.client import timeline
from Constants import TPS_DIR, CATEGORY, Word2VecPath

tf.flags.DEFINE_string("Word2Vec", Word2VecPath, "Word2vec file with pre-trained embeddings (default: None)")
tf.flags.DEFINE_string("train_data", os.path.join(TPS_DIR, CATEGORY + '.train'), "Data for training")
tf.flags.DEFINE_string("valid_data", os.path.join(TPS_DIR, CATEGORY + '.test'), " Data for validation")
tf.flags.DEFINE_string("para_data", os.path.join(TPS_DIR, CATEGORY + '.para'), "Data parameters")
tf.flags.DEFINE_string("para_infor", os.path.join(TPS_DIR, 'item_infor.para'), "Data infor")
tf.flags.DEFINE_string("log_path", TPS_DIR + '/log/', "Log Path")
tf.flags.DEFINE_string("save_path", TPS_DIR + '/model/', "Save Path")
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding ")
tf.flags.DEFINE_string("filter_sizes", "3", "Comma-separated filter sizes ")
tf.flags.DEFINE_integer("num_filters", 100, "Number of filters per filter size")
# Training parameters
tf.flags.DEFINE_integer("batch_size", 20, "Batch Size ")
tf.flags.DEFINE_integer("num_epochs", 40, "Number of training epochs ")


def run_model():
    with tf.device('/gpu:1'):
        opt = Model.Model(
            review_num_u = review_num_u,
            review_num_i = review_num_i,
            review_len_u = review_len_u,
            review_len_i = review_len_i,
            category_num = category_num,
            title_len = title_len,
            user_num = user_num,
            item_num = item_num,
            user_vocab_size = len(vocabulary_user),
            item_vocab_size = len(vocabulary_item),
            title_vocab_size = len(title_voc),
            embedding_id = 32,
            attention_size = 32,
            full_connection_size = 32,
            filter_sizes = list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters = FLAGS.num_filters,
            max_tip_len = max_tip_len,
            vocabulary_tip_size = len(vocabulary_tip),
            embedding_size = FLAGS.embedding_dim,
            batch_size = FLAGS.batch_size)
        res_, g_loss_, d_loss_, gen_op, dis_op = Model.textGAN(opt)
        merged = tf.summary.merge_all()

    uidx = 0
    config = tf.ConfigProto(log_device_placement = False, allow_soft_placement = True, graph_options = tf.GraphOptions(build_cost_model = 1))
    np.set_printoptions(precision = 3)
    np.set_printoptions(threshold = np.inf)
    saver = tf.train.Saver()
    run_metadata = tf.RunMetadata() # 定义TensorFlow运行的元信息，这样可以记录训练时运算时间和内存占用等方面的信息。


    with tf.Session(config = config) as sess:
        sess.run(tf.global_variables_initializer())
        # writer = tf.summary.FileWriter("logs/", sess.graph)

        initW = np.random.uniform(-1.0, 1.0, (len(vocabulary_user), FLAGS.embedding_dim))
        for word, idx in vocabulary_user.items():
            if word in vocab_vector_dict:
                try:
                    initW[idx] = vocab_vector_dict[word]
                except:
                    print(idx)
        sess.run(opt.W1.assign(initW))

        initW = np.random.uniform(-1.0, 1.0, (len(vocabulary_item), FLAGS.embedding_dim))
        for word, idx in vocabulary_item.items():
            if word in vocab_vector_dict:
                try:
                    initW[idx] = vocab_vector_dict[word]
                except:
                    print(idx)
        sess.run(opt.W2.assign(initW))

        initW = np.random.uniform(-1.0, 1.0, (len(title_voc), FLAGS.embedding_dim))
        for word, idx in title_voc.items():
            if word in vocab_vector_dict:
                initW[idx] = vocab_vector_dict[word]
        sess.run(opt.W3.assign(initW))

        initW = np.random.uniform(-1.0, 1.0, (len(vocabulary_tip), FLAGS.embedding_dim))
        for word, idx in vocabulary_tip.items():
            if word in vocab_vector_dict:
                initW[idx] = vocab_vector_dict[word]
        sess.run(opt.W4.assign(initW))

        data_size_train, data_size_test = len(train_data), len(test_data)
        batch_size = FLAGS.batch_size
        ll = int(len(train_data) / batch_size)
        print(data_size_train, data_size_test, ll)

        for epoch in range(FLAGS.num_epochs):
            print("starting epoch: %d" % epoch)
            shuffle_indices = np.random.permutation(np.arange(data_size_train))
            shuffled_data = train_data[shuffle_indices]

            d_loss_total, g_loss_total = 0, 0
            for batch_num in range(ll):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size_train)
                data_train = shuffled_data[start_index: end_index]

                uid, iid, reuid, reiid, y_batch, y_tip = zip(*data_train)
                u_batch, i_batch, infor_category, infor_description, infor_title = [], [], [], [], []
                for i in range(len(uid)):
                    u_batch.append(u_text[uid[i][0]])
                    i_batch.append(i_text[iid[i][0]])
                    i_infor = item_infor[iid[i][0]]
                    infor_category.append(i_infor[0])
                    infor_title.append(i_infor[1])
                u_batch = np.array(u_batch)
                i_batch = np.array(i_batch)
                infor_category = np.array(infor_category)
                infor_title = np.array(infor_title)

                feed_dict = {opt.input_u: u_batch,
                             opt.input_i: i_batch,
                             opt.input_category: infor_category,
                             opt.input_title: infor_title,
                             opt.input_uid: uid,
                             opt.input_iid: iid,
                             opt.input_y: y_batch,
                             opt.input_y_tip: y_tip,
                             opt.input_reuid: reuid,
                             opt.input_reiid: reiid}


                uidx += 1
                if uidx % opt.dis_steps == 0:
                    _, d_loss = sess.run([dis_op, d_loss_], feed_dict = feed_dict, options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE), run_metadata = run_metadata)
                    d_loss_total += d_loss

                if uidx % opt.gen_steps == 0:
                    _, g_loss = sess.run([gen_op, g_loss_], feed_dict = feed_dict, options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE), run_metadata = run_metadata)
                    g_loss_total += g_loss

                tl = timeline.Timeline(run_metadata.step_stats)

                print('batch_num: %d, d_loss: %f, g_loss: %f ' % (batch_num, d_loss_total / (batch_num + 1), g_loss_total / (batch_num + 1)))
            #     ctf = tl.generate_chrome_trace_format()
            #     with open('timeline.json', 'w') as f:
            #         f.write(ctf)
            #     if uidx == 1:
            #         break
            # break
            saver.save(sess, TPS_DIR + '/save_path/', global_step = epoch)  # 有问题

if __name__ == '__main__':
    FLAGS = tf.flags.FLAGS
    FLAGS.flag_values_dict()
    for attr, value in sorted(FLAGS.flag_values_dict().items()):
        print("{} : {}".format(attr, value))

    # user-item-rating-tip
    para = pickle.load(open(FLAGS.para_data, 'rb'))
    user_num = para['user_num']
    item_num = para['item_num']
    review_num_u = para['review_num_u']
    review_num_i = para['review_num_i']
    review_len_u = para['review_len_u']
    review_len_i = para['review_len_i']
    max_tip_len = para['max_tip_len']
    vocabulary_user = para['user_vocab']
    vocabulary_item = para['item_vocab']
    vocabulary_tip = para['tip_vocab']
    train_length = para['train_length']
    test_length = para['test_length']
    u_text = para['u_text']
    i_text = para['i_text']
    print(max_tip_len, len(vocabulary_tip), user_num, item_num, review_num_u, review_len_u, review_num_i, review_len_i, len(vocabulary_user),len(vocabulary_item))

    #item-infor
    para_infor = pickle.load(open(FLAGS.para_infor, 'rb'))
    category2id = para_infor['category2id']
    category_num = para_infor['categorynum']
    title_voc = para_infor['title_voc']
    title_len = para_infor['title_len']
    item_infor = para_infor['item_infor']
    print(category_num, title_len)

    # word2vec
    with open(FLAGS.Word2Vec, "rb") as f:
        vocab_vector_dict = pickle.load(f)
    np.random.seed(2018)
    random_seed = 2018



    train_data = pickle.load(open(FLAGS.train_data, 'rb'))
    train_data = np.array(train_data)

    test_data = pickle.load(open(FLAGS.valid_data, 'rb'))
    test_data = np.array(test_data)
    run_model()

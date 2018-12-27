

import numpy as np
import tensorflow as tf
from math import floor
from tensorflow.contrib import layers
from Model import TextGANModel
import sys
sys.path.append('./')


class Model(object):
    def __init__(self, review_num_u, review_num_i, review_len_u, review_len_i, category_num, title_len,
                 user_num, item_num, user_vocab_size, item_vocab_size, title_vocab_size,
                 embedding_id, attention_size, full_connection_size, filter_sizes, num_filters,
                 max_tip_len, vocabulary_tip_size, embedding_size, batch_size):

        # imputs
        self.input_u = tf.placeholder(tf.int32, [None, review_num_u, review_len_u], name="input_u")
        self.input_i = tf.placeholder(tf.int32, [None, review_num_i, review_len_i], name="input_i")
        self.input_reuid = tf.placeholder(tf.int32, [None, review_num_u], name='input_reuid')
        self.input_reiid = tf.placeholder(tf.int32, [None, review_num_i], name='input_reuid')
        self.input_category = tf.placeholder(tf.float32, [None, category_num], name='input_category')
        self.input_title = tf.placeholder(tf.int32, [None, title_len], name='input_title')
        self.input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
        self.input_y_tip = tf.placeholder(tf.int32, [None, max_tip_len], name="input_y_tip")
        self.input_uid = tf.placeholder(tf.int32, [None, 1], name="input_uid")
        self.input_iid = tf.placeholder(tf.int32, [None, 1], name="input_iid")

        # user_item_encoder-parameter
        self.review_num_u = review_num_u
        self.review_num_i = review_num_i
        self.review_len_u = review_len_u
        self.review_len_i = review_len_i
        self.title_len = title_len
        self.user_num = user_num
        self.item_num = item_num
        self.user_vocab_size = user_vocab_size
        self.item_vocab_size = item_vocab_size
        self.title_vocab_size = title_vocab_size
        self.embedding_id = embedding_id
        self.attention_size = attention_size
        self.full_connection_size = full_connection_size
        self.filter_sizes = filter_sizes
        self.num_filters= num_filters

        # voc-vec
        self.W1 = tf.Variable(tf.random_uniform([user_vocab_size, embedding_size], -1.0, 1.0), name = "W1")
        self.W2 = tf.Variable(tf.random_uniform([item_vocab_size, embedding_size], -1.0, 1.0), name = "W2")
        self.W3 = tf.Variable(tf.random_uniform([title_vocab_size, embedding_size], -1.0, 1.0), name = "W3")
        self.W4 = tf.Variable(tf.random_uniform([vocabulary_tip_size, embedding_size], -1.0, 1.0), name="W4")

        # TextGAN-parameter
        self.max_tip_len = max_tip_len
        self.vocabulary_tip_zize = vocabulary_tip_size
        self.n_latent = 32
        self.embedding_size = embedding_size
        self.batch_size = batch_size

        self.additive_noise_lambda = 0.0
        self.sigma_range = [2]
        self.n_hid = 100
        self.n_gan = 128
        self.stride = [2, 2, 2]
        self.L = 1000

        self.discrimination = False
        self.H_dis = 300

        self.batch_norm = False
        self.dropout = False
        self.dropout_ratio = 1
        self.dropout_keep_prob = 1

        self.filter_shape = 5
        self.filter_size = 300
        self.layer = 3

        self.optimizer = 'Adam'
        self.clip_grad = None
        self.lr = 1e-5
        self.clip_grad = None
        self.attentive_emb = False
        self.decay_rate = 0.99

        self.restore = True
        self.dis_steps = 1
        self.gen_steps = 1
        self.valid_freq = 100

        self.sigma_range = [2]

        self.sent_len2 = np.int32(floor((self.max_tip_len - self.filter_shape) / self.stride[0]) + 1)
        self.sent_len3 = np.int32(floor((self.sent_len2 - self.filter_shape) / self.stride[1]) + 1)
        self.sent_len4 = np.int32(floor((self.sent_len3 - self.filter_shape) / self.stride[2]) + 1)

def user_item_encoder(mo):
    with tf.variable_scope("user_embedding"):
        embedded_user = tf.nn.embedding_lookup(mo.W1, mo.input_u)  # 找到要寻找的embedding data中的对应的行下的vector
        embedded_users = tf.expand_dims(embedded_user, -1)  # -1表示增加一维

    with tf.variable_scope("item_embedding"):
        embedded_item = tf.nn.embedding_lookup(mo.W2, mo.input_i)
        embedded_items = tf.expand_dims(embedded_item, -1)

    with tf.variable_scope("item_title_embedding"):
        embedded_item_title = tf.nn.embedding_lookup(mo.W3, mo.input_title)
        embedded_item_titles = tf.expand_dims(embedded_item_title, -1)

    # user-review
    pooled_outputs_u = []
    for i, filter_size in enumerate(mo.filter_sizes):
        with tf.variable_scope("user_conv-maxpool-%s" % filter_size):
            filter_shape = [filter_size, mo.embedding_size, 1, mo.num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[mo.num_filters]), name="b")
            embedded_users = tf.reshape(embedded_users,[-1, mo.review_len_u, mo.embedding_size, 1])  #
            conv = tf.nn.conv2d(embedded_users, W, strides=[1, 1, 1, 1], padding="VALID",name="conv")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            pooled = tf.nn.max_pool(h, ksize=[1, mo.review_len_u - filter_size + 1, 1, 1],strides=[1, 1, 1, 1], padding='VALID', name="pool")
            pooled_outputs_u.append(pooled)
    num_filters_total = mo.num_filters * len(mo.filter_sizes)
    h_pool_u = tf.concat(pooled_outputs_u, 3)
    h_pool_flat_u = tf.reshape(h_pool_u,[-1, mo.review_num_u, num_filters_total])  # 把池化层输出变成一维向量

    # item-review
    pooled_outputs_i = []
    for i, filter_size in enumerate(mo.filter_sizes):
        with tf.variable_scope("item_conv-maxpool-%s" % filter_size):
            filter_shape = [filter_size, mo.embedding_size, 1, mo.num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[mo.num_filters]), name="b")
            embedded_items = tf.reshape(embedded_items, [-1, mo.review_len_i, mo.embedding_size, 1])
            conv = tf.nn.conv2d(embedded_items, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            pooled = tf.nn.max_pool(h, ksize=[1, mo.review_len_i - filter_size + 1, 1, 1],strides=[1, 1, 1, 1], padding='VALID', name="pool")
            pooled_outputs_i.append(pooled)
    num_filters_total = mo.num_filters * len(mo.filter_sizes)
    h_pool_i = tf.concat(pooled_outputs_i, 3)
    h_pool_flat_i = tf.reshape(h_pool_i, [-1, mo.review_num_i, num_filters_total])

    # item-title
    pooled_outputs_title = []
    for i, filter_size in enumerate(mo.filter_sizes):
        with tf.variable_scope("item_title_conv-maxpool-%s" % filter_size):
            filter_shape = [filter_size, mo.embedding_size, 1, mo.num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[mo.num_filters]), name="b")
            embedded_item_titles = tf.reshape(embedded_item_titles, [-1, mo.title_len, mo.embedding_size, 1])
            conv = tf.nn.conv2d(embedded_item_titles, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            pooled = tf.nn.max_pool(h, ksize=[1, mo.title_len - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")
            pooled_outputs_title.append(pooled)
    num_filters_total = mo.num_filters * len(mo.filter_sizes)
    h_pool_flat_title = tf.reshape(tf.concat(pooled_outputs_title, 3), [-1, num_filters_total])
    t_feas = tf.layers.dense(inputs = h_pool_flat_title, units = mo.full_connection_size, activation=tf.nn.relu)

    # item-category
    c_feas = tf.layers.dense(inputs = mo.input_category, units = mo.full_connection_size, activation=tf.nn.relu)

    with tf.variable_scope("attention"):
        iidW = tf.Variable(tf.random_uniform([mo.item_num + 2, mo.embedding_id], -0.1, 0.1), name="iidW")
        uidW = tf.Variable(tf.random_uniform([mo.user_num + 2, mo.embedding_id], -0.1, 0.1), name="uidW")
        Wau = tf.Variable(tf.random_uniform([num_filters_total, mo.attention_size], -0.1, 0.1), name='Wau')
        Wru = tf.Variable(tf.random_uniform([mo.embedding_id, mo.attention_size], -0.1, 0.1), name='Wru')
        Wpu = tf.Variable(tf.random_uniform([mo.attention_size, 1], -0.1, 0.1), name='Wpu')
        bau = tf.Variable(tf.constant(0.1, shape=[mo.attention_size]), name="bau")
        bbu = tf.Variable(tf.constant(0.1, shape=[1]), name="bbu")
        iid_a = tf.nn.relu(tf.nn.embedding_lookup(iidW, mo.input_reuid))
        u_j = tf.einsum('ajk,kl->ajl', tf.nn.relu(tf.einsum('ajk,kl->ajl', h_pool_flat_u, Wau) + tf.einsum('ajk,kl->ajl', iid_a, Wru) + bau), Wpu) + bbu  # None*u_len*1
        u_a = tf.nn.softmax(u_j, 1)  # none*u_len*1

        Wai = tf.Variable(tf.random_uniform([num_filters_total, mo.attention_size], -0.1, 0.1), name='Wai')
        Wri = tf.Variable(tf.random_uniform([mo.embedding_id, mo.attention_size], -0.1, 0.1), name='Wri')
        Wpi = tf.Variable(tf.random_uniform([mo.attention_size, 1], -0.1, 0.1), name='Wpi')
        bai = tf.Variable(tf.constant(0.1, shape=[mo.attention_size]), name="bai")
        bbi = tf.Variable(tf.constant(0.1, shape=[1]), name="bbi")
        uid_a = tf.nn.relu(tf.nn.embedding_lookup(uidW, mo.input_reiid))
        i_j = tf.einsum('ajk,kl->ajl', tf.nn.relu(tf.einsum('ajk,kl->ajl', h_pool_flat_i, Wai) + tf.einsum('ajk,kl->ajl', uid_a, Wri) + bai), Wpi) + bbi
        i_a = tf.nn.softmax(i_j, 1)  # none*len*1

    with tf.variable_scope("add_reviews"):
        u_feas = tf.reduce_sum(tf.multiply(u_a, h_pool_flat_u), 1)
        u_feas = tf.nn.dropout(u_feas, mo.dropout_keep_prob)
        i_feas = tf.reduce_sum(tf.multiply(i_a, h_pool_flat_i), 1)
        i_feas = tf.nn.dropout(i_feas, mo.dropout_keep_prob)

    with tf.variable_scope("fea_fusion"):
        Wu = tf.Variable(tf.random_uniform([num_filters_total, mo.n_latent], -0.1, 0.1), name='Wu')
        u_feas = tf.matmul(u_feas, Wu)

        Wi = tf.Variable(tf.random_uniform([num_filters_total, mo.n_latent], -0.1, 0.1), name='Wi')
        i_feas = tf.matmul(i_feas, Wi)

        Wt = tf.Variable(tf.random_uniform([mo.full_connection_size, mo.n_latent], -0.1, 0.1), name='Wi')
        t_feas = tf.matmul(t_feas, Wt)

        Wc = tf.Variable(tf.random_uniform([mo.full_connection_size, mo.n_latent], -0.1, 0.1), name='Wc')
        c_feas = tf.matmul(c_feas, Wc)

        b = tf.Variable(tf.constant(0.1, shape=[mo.n_latent]), name="b")
        feas = u_feas + i_feas + t_feas + c_feas + b
    return feas


def discriminator(x, mo, prefix = 'd_', is_prob = False, is_reuse = None):
    H = encoder(x, mo.W4, mo, prefix = prefix + 'enc_', is_prob = is_prob, is_reuse = is_reuse)
    logits = TextGANModel.discriminator_2layer(H, mo, is_reuse = is_reuse)
    return logits, H


def encoder(x, W_norm_d, mo, prefix = 'd_', is_prob = False, is_reuse = None):
    if is_prob:
        x_emb = tf.tensordot(x, W_norm_d, [[2], [0]]) # 指定任何轴,指定的轴形状一致
    else:
        x_emb = tf.nn.embedding_lookup(W_norm_d, x)  # batch L emb

    x_emb = tf.expand_dims(x_emb, 3)  # batch L emb 1
    H = TextGANModel.conv_model_3layer(x_emb, mo, prefix = prefix, is_reuse = is_reuse)
    return tf.squeeze(H, [1, 2])


def textGAN(mo):

    # Generator
    with tf.variable_scope("pretrain"):
        z = user_item_encoder(mo)  # 得到初始z
        syn_sent, logits = TextGANModel.lstm_decoder_embedding(z, mo.W4, mo, add_go = True, feed_previous = True, is_reuse = None)
        prob = [tf.nn.softmax(l * mo.L) for l in logits]
        prob = tf.stack(prob, 1)

    # Discriminator
    with tf.variable_scope("d_net"):
        logits_real, H_real = discriminator(mo.input_y_tip, mo)
    with tf.variable_scope("d_net"):
        logits_fake, H_fake = discriminator(prob, mo, is_prob = True, is_reuse = True)

    # Loss
    D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(logits_real), logits = logits_real)) + \
             tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(logits_fake), logits = logits_fake))

    fake_mean = tf.reduce_mean(H_fake, axis = 0)
    real_mean = tf.reduce_mean(H_real, axis = 0)
    mean_dist = tf.sqrt(tf.reduce_mean((fake_mean - real_mean) ** 2))
    G_loss = mean_dist

    GAN_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = logits_fake, labels = tf.ones_like(logits_fake))
    MMD_loss = TextGANModel.compute_MMD_loss(tf.squeeze(H_fake), tf.squeeze(H_real), mo)

    res_ = {}
    res_['syn_sent'] = syn_sent
    res_['real_f'] = tf.squeeze(H_real)
    res_['mean_dist'] = mean_dist
    res_['mmd'] = MMD_loss
    res_['gan'] = tf.reduce_mean(GAN_loss)

    tf.summary.scalar('D_loss', D_loss)
    tf.summary.scalar('G_loss', G_loss)
    # summaries = ["learning_rate", "G_loss", "D_loss"]
    global_step = tf.Variable(0, trainable = False)

    all_vars = tf.trainable_variables()  # 返回的是需要训练的变量列表
    g_vars = [var for var in all_vars if var.name.startswith('pretrain')]
    d_vars = [var for var in all_vars if var.name.startswith('d_')]

    generator_op = layers.optimize_loss(
        G_loss,
        global_step = global_step,
        optimizer = mo.optimizer,
        clip_gradients = (lambda grad: TextGANModel._clip_gradients_seperate_norm(grad, mo.clip_grad)) if mo.clip_grad else None,
        learning_rate_decay_fn = lambda lr, g: tf.train.exponential_decay(learning_rate = lr, global_step = g, decay_rate = mo.decay_rate, decay_steps = 3000),
        learning_rate = mo.lr,
        variables = g_vars)

    discriminator_op = layers.optimize_loss(
        D_loss,
        global_step = global_step,
        optimizer = mo.optimizer,
        clip_gradients = (lambda grad: TextGANModel._clip_gradients_seperate_norm(grad, mo.clip_grad)) if mo.clip_grad else None,
        learning_rate_decay_fn = lambda lr, g: tf.train.exponential_decay(learning_rate = lr, global_step = g, decay_rate = mo.decay_rate, decay_steps = 3000),
        learning_rate = mo.lr,
        variables = d_vars)
    return res_, G_loss, D_loss, generator_op, discriminator_op





import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.python.ops import nn_ops, math_ops, embedding_ops, variable_scope
from tensorflow.python.ops import clip_ops


def lstm_decoder_embedding(H, W_emb, opt, prefix = '', add_go = False, feed_previous = False, is_reuse = None, is_fed_h = False):
    biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
    if not opt.additive_noise_lambda: # true
        H = layers.fully_connected(H, num_outputs = opt.n_hid, biases_initializer = biasInit, activation_fn = None, scope = prefix + 'lstm_decoder', reuse = is_reuse)
    H0 = tf.squeeze(H)
    H1 = (H0, tf.zeros_like(H0))

    with tf.variable_scope(prefix + 'lstm_decoder', reuse = True):
        cell = tf.contrib.rnn.LSTMCell(opt.n_hid)
    with tf.variable_scope(prefix + 'lstm_decoder', reuse = is_reuse):
        weightInit = tf.random_uniform_initializer(-0.001, 0.001)
        W = tf.get_variable('W', [opt.n_hid, opt.embedding_size], initializer = weightInit)
        b = tf.get_variable('b', [opt.vocabulary_tip_zize], initializer = tf.random_uniform_initializer(-0.001, 0.001))
        W_new = tf.matmul(W, W_emb, transpose_b = True)
        out_proj = (W_new, b) if feed_previous else None # feed_previous = True

        with variable_scope.variable_scope("embedding_rnn_decoder"):
            loop_function = _extract_argmax_and_embed(W_emb, H0, out_proj, is_fed_h = is_fed_h)
            decoder_res = rnn_decoder_truncated(opt, H1, cell, loop_function = loop_function)
        outputs = decoder_res[0]

    logits = [nn_ops.xw_plus_b(out, W_new, b) for out in outputs]
    syn_sents = [math_ops.argmax(l, 1) for l in logits[:-1]]
    syn_sents = tf.stack(syn_sents, 1)
    return syn_sents, logits


def _extract_argmax_and_embed(embedding,
                              h,
                              output_projection = None,
                              update_embedding = True,
                              is_fed_h = True):

    def loop_function_with_sample(prev, _):
        if output_projection is not None:
            prev = nn_ops.xw_plus_b(prev, output_projection[0], output_projection[1])

        prev_symbol = math_ops.argmax(prev, 1)
        emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
        emb_prev = tf.concat([emb_prev, h], 1) if is_fed_h else emb_prev
        return emb_prev
    return loop_function_with_sample


def rnn_decoder_truncated(opt,
                          state,
                          cell,
                          loop_function = None,
                          scope = None):
    with variable_scope.variable_scope(scope or "rnn_decoder"):
        outputs = []
        prev = None
        inp = tf.zeros([opt.batch_size, 100]) # 设置初始输入
        for i in range(opt.max_tip_len + 1):
            if loop_function is not None and prev is not None:
                with variable_scope.variable_scope("loop_function", reuse=True):
                    inp = loop_function(prev, i)
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
            output, state = cell(inp, state)
            outputs.append(output)
            if loop_function is not None:
                prev = output
    return outputs, state


def discriminator_2layer(H, opt, prefix = '', is_reuse = None, is_train = True):
    biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
    H = regularization(H, opt, is_train, prefix= prefix + 'reg_H', is_reuse= is_reuse)
    H_dis = layers.fully_connected(H, num_outputs = opt.H_dis, biases_initializer = biasInit, activation_fn = tf.nn.relu, scope = prefix + 'dis_1', reuse = is_reuse)
    H_dis = regularization(H_dis, opt, is_train, prefix= prefix + 'reg_H_dis', is_reuse= is_reuse)
    logits = layers.linear(H_dis, num_outputs = 1, biases_initializer=biasInit, scope = prefix + 'disc', reuse = is_reuse) # 全连接层
    return logits

def regularization(X, opt, is_train, prefix = '', is_reuse = None):
    if '_X' not in prefix and '_H_dec' not in prefix:
        if opt.batch_norm:
            X = layers.batch_norm(X, decay = 0.9, center = True, scale = True, is_training = is_train, scope = prefix+'_bn', reuse = is_reuse)
        X = tf.nn.relu(X)
    X = X if (not opt.dropout or is_train is None) else layers.dropout(X, keep_prob = opt.dropout_ratio, scope=prefix + '_dropout')
    return X

def conv_model_3layer(X, opt, prefix = '', is_reuse= None, num_outputs = None, is_train = True, multiplier = 2):
    biasInit = None if opt.batch_norm else tf.constant_initializer(0.001, dtype = tf.float32) # self.batch_norm = False
    weightInit = tf.constant_initializer(0.001, dtype=tf.float32)

    X = regularization(X, opt,  prefix = prefix + 'reg_X', is_reuse = is_reuse, is_train = is_train) # 不变
    #  num_outputs指定卷积核的个数
    H1 = layers.conv2d(X, num_outputs = opt.filter_size,  kernel_size = [opt.filter_shape, opt.embedding_size], stride = [opt.stride[0], 1],  weights_initializer = weightInit, biases_initializer = biasInit, padding = 'VALID', scope = prefix + 'H1_3', reuse = is_reuse)  # batch L-3 1 Filtersize
    H1 = regularization(H1, opt, prefix = prefix + 'reg_H1', is_reuse = is_reuse, is_train = is_train) # relu

    H2 = layers.conv2d(H1,  num_outputs = opt.filter_size * multiplier,  kernel_size = [opt.filter_shape, 1], stride = [opt.stride[1], 1],  biases_initializer = biasInit, activation_fn=None, padding = 'VALID', scope = prefix + 'H2_3', reuse = is_reuse)
    H2 = regularization(H2, opt,  prefix = prefix + 'reg_H2', is_reuse= is_reuse, is_train = is_train) # relu

    H3 = layers.conv2d(H2, num_outputs = (num_outputs if num_outputs else opt.n_gan),  kernel_size=[opt.sent_len3, 1], activation_fn = tf.nn.tanh, padding = 'VALID', scope = prefix + 'H3_3', reuse = is_reuse) # batch 1 1 2*Filtersize

    return H3


def compute_MMD_loss(H_fake, H_real, opt):
    dividend = 1
    dist_x, dist_y = H_fake / dividend, H_real / dividend
    x_sq = tf.expand_dims(tf.reduce_sum(dist_x ** 2, axis = 1), 1)   #  64 * 1 转置
    y_sq = tf.expand_dims(tf.reduce_sum(dist_y ** 2, axis = 1), 1)   #  64 * 1
    dist_x_T = tf.transpose(dist_x)
    dist_y_T = tf.transpose(dist_y)
    x_sq_T = tf.transpose(x_sq)
    y_sq_T = tf.transpose(y_sq)

    tempxx = -2 * tf.matmul(dist_x, dist_x_T) + x_sq + x_sq_T  # (xi -xj)**2
    tempxy = -2 * tf.matmul(dist_x, dist_y_T) + x_sq + y_sq_T  # (xi -yj)**2
    tempyy = -2 * tf.matmul(dist_y, dist_y_T) + y_sq + y_sq_T  # (yi -yj)**2


    for sigma in opt.sigma_range:
        kxx, kxy, kyy = 0, 0, 0
        kxx += tf.reduce_mean(tf.exp(-tempxx / 2 / (sigma ** 2)))
        kxy += tf.reduce_mean(tf.exp(-tempxy / 2 / (sigma ** 2)))
        kyy += tf.reduce_mean(tf.exp(-tempyy / 2 / (sigma ** 2)))
    gan_cost_g = tf.sqrt(kxx + kyy - 2 * kxy)
    return gan_cost_g

def _clip_gradients_seperate_norm(grads_and_vars, clip_gradients):
    """Clips gradients by global norm."""
    gradients, variables = zip(*grads_and_vars)
    clipped_gradients = [clip_ops.clip_by_norm(grad, clip_gradients) for grad in gradients]
    return list(zip(clipped_gradients, variables))
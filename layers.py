import tensorflow as tf

def create_rnn_cell(cell_type, state_size, default_initializer, reuse=None):
    if cell_type == 'GRU':
        return tf.nn.rnn_cell.GRUCell(state_size, activation=tf.nn.tanh, reuse=reuse)
    elif cell_type == 'LSTM':
        return tf.nn.rnn_cell.LSTMCell(state_size, initializer=default_initializer, activation=tf.nn.tanh, reuse=reuse)
    else:
        return tf.nn.rnn_cell.BasicRNNCell(state_size, activation=tf.nn.tanh, reuse=reuse)


def create_rnn_encoder(x, rnn_units, dropout_rate, seq_length, rnn_cell_type, param_initializer, seed, reuse=None):
    with tf.variable_scope("RNN_Encoder", reuse=reuse):
        rnn_cell = create_rnn_cell(rnn_cell_type, rnn_units, param_initializer)
        rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, input_keep_prob=1 - dropout_rate, seed=seed)
        init_state = rnn_cell.zero_state(tf.shape(x)[0], tf.float32)
        # RNN Encoder: Iteratively compute output of recurrent network
        rnn_outputs, _ = tf.nn.dynamic_rnn(rnn_cell, x, initial_state=init_state, sequence_length=seq_length, dtype=tf.float32)
        return rnn_outputs

def create_basket_encoder(x, dense_units, param_initializer, activation_func=None, name="Basket_Encoder", reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        return tf.layers.dense(x, dense_units, kernel_initializer=param_initializer,
                            bias_initializer=tf.zeros_initializer, activation=activation_func)

def get_last_right_output(full_output, max_length, actual_length, rnn_units):
    batch_size = tf.shape(full_output)[0]
    # Start indices for each sample
    index = tf.range(0, batch_size) * max_length + (actual_length - 1)
    # Indexing
    return tf.gather(tf.reshape(full_output, [-1, rnn_units]), index)


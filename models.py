import tensorflow as tf
import numpy as np

from layers import *
from abc import abstractmethod


class Model:

    def __init__(self, sess, seed, learning_rate, name='model'):
        self.scope = name
        self.session = sess
        self.seed = seed
        self.learning_rate = tf.constant(learning_rate)

    @abstractmethod
    def train_batch(self, s, s_length, y):
        pass

    @abstractmethod
    def validate_batch(self, s, s_length, y):
        pass

    @abstractmethod
    def generate_prediction(self, s, s_length):
        pass


class Beacon(Model):

    def __init__(self, sess, emb_dim, rnn_units, alpha, 
                 max_seq_length, item_probs, adj_matrix, top_k,
                 batch_size, rnn_cell_type, rnn_dropout_rate, seed, learning_rate):

        super().__init__(sess, seed, learning_rate, name="GRN")

        self.emb_dim = emb_dim
        self.rnn_units = rnn_units

        self.max_seq_length = max_seq_length
        self.nb_items = len(item_probs)
        self.item_probs = item_probs
        self.alpha = alpha
        self.batch_size = batch_size
        self.top_k = top_k

        with tf.variable_scope(self.scope):
            # Initialized for n_hop adjacency matrix
            self.A = tf.constant(adj_matrix.todense(), name="Adj_Matrix", dtype=tf.float32)

            uniform_initializer = np.ones(shape=(self.nb_items), dtype=np.float32) / self.nb_items
            self.I_B = tf.get_variable(dtype=tf.float32, initializer=tf.constant(uniform_initializer, dtype=tf.float32), name="I_B")
            self.I_B_Diag = tf.nn.relu(tf.diag(self.I_B, name="I_B_Diag"))

            self.C_Basket = tf.get_variable(dtype=tf.float32, initializer=tf.constant(adj_matrix.mean()), name="C_B")
            self.y = tf.placeholder(dtype=tf.float32, shape=(batch_size, self.nb_items), name='Target_basket')      

            # Basket Sequence encoder
            with tf.name_scope("Basket_Sequence_Encoder"):
                self.bseq = tf.sparse_placeholder(shape=(batch_size, self.max_seq_length, self.nb_items), dtype=tf.float32, name="bseq_input")
                self.bseq_length = tf.placeholder(dtype=tf.int32, shape=(batch_size,), name='bseq_length')

                self.bseq_encoder = tf.sparse_reshape(self.bseq, shape=[-1, self.nb_items], name="bseq_2d")
                self.bseq_encoder = self.encode_basket_graph(self.bseq_encoder, self.C_Basket, True)
                self.bseq_encoder = tf.reshape(self.bseq_encoder, shape=[-1, self.max_seq_length, self.nb_items], name="bsxMxN")
                self.bseq_encoder = create_basket_encoder(self.bseq_encoder, emb_dim, param_initializer=tf.initializers.he_uniform(), activation_func=tf.nn.relu)       

                # batch_size x max_seq_length x H
                rnn_encoder = create_rnn_encoder(self.bseq_encoder, self.rnn_units, rnn_dropout_rate, self.bseq_length, rnn_cell_type, 
                                                    param_initializer=tf.initializers.glorot_uniform(), seed=self.seed)
                
                # Hack to build the indexing and retrieve the right output. # batch_size x H
                h_T = get_last_right_output(rnn_encoder, self.max_seq_length, self.bseq_length, self.rnn_units)

            # Next basket estimation
            with tf.name_scope("Next_Basket"):
                W_H = tf.get_variable(dtype=tf.float32, initializer=tf.initializers.glorot_uniform(), shape=(self.rnn_units, self.nb_items), name="W_H")
                
                next_item_probs = tf.nn.sigmoid(tf.matmul(h_T, W_H))
                logits = (1.0 - self.alpha) * next_item_probs + self.alpha * self.encode_basket_graph(next_item_probs, tf.constant(0.0))

            with tf.name_scope("Loss"):
                self.loss = self.compute_loss(logits, self.y)

                self.predictions = tf.nn.sigmoid(logits)
                self.top_k_values, self.top_k_indices = tf.nn.top_k(self.predictions, 200)
                self.recall_at_k = self.compute_recall_at_topk(top_k)

            # Adam optimizer
            train_op = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
            # Op to calculate every variable gradient
            self.grads = train_op.compute_gradients(self.loss, tf.trainable_variables())
            self.update_grads = train_op.apply_gradients(self.grads)

            # Summarize all variables and their gradients
            total_parameters = 0
            print("-------------------- SUMMARY ----------------------")
            tf.summary.scalar("C_Basket", self.C_Basket)

            for grad, var in self.grads:
                tf.summary.histogram(var.name, var)
                tf.summary.histogram(var.name + '/grad', grad)

                shape = var.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                print("+ {:<64} {:<10,} parameter(s)".format(var.name, variable_parameters))
                total_parameters += variable_parameters

            print("Total number of parameters: {:,}".format(total_parameters))
            print("----------------- END SUMMARY ----------------------\n")

            # Create a summary to monitor cost tensor
            tf.summary.scalar("Train_Batch_Loss", self.loss)
            #tf.summary.scalar("Train_Batch_Recall", self.recall_at_k)

            # Create a summary to monitor cost tensor
            tf.summary.scalar("Val_Batch_Loss", self.loss, collections=['validation'])
            #tf.summary.scalar("Val_Batch_Recall", self.recall_at_k, collections=['validation'])

            # Merge all summaries into a single op
            self.merged_summary_op = tf.summary.merge_all()
            self.val_merged_summary_op = tf.summary.merge_all(key='validation')

    def train_batch(self, s, s_length, y):
        bseq_indices, bseq_values = self.get_sparse_tensor_info(s, True)

        _, loss, recall, summary = self.session.run(
            [self.update_grads, self.loss, self.recall_at_k, self.merged_summary_op],
            feed_dict={self.bseq_length: s_length, self.y: y,
                       self.bseq.indices: bseq_indices, self.bseq.values: bseq_values})

        return loss, recall, summary

    def validate_batch(self, s, s_length, y):
        bseq_indices, bseq_values = self.get_sparse_tensor_info(s, True)

        loss, recall, summary = self.session.run(
            [self.loss, self.recall_at_k, self.val_merged_summary_op],
            feed_dict={ self.bseq_length: s_length, self.y: y,
                        self.bseq.indices: bseq_indices, self.bseq.values: bseq_values})
        return loss, recall, summary

    def generate_prediction(self, s, s_length):
        bseq_indices, bseq_values = self.get_sparse_tensor_info(s, True)
        return self.session.run([self.top_k_values, self.top_k_indices],
                                 feed_dict={self.bseq_length: s_length, self.bseq.indices: bseq_indices, self.bseq.values: bseq_values})

    def encode_basket_graph(self, binput, beta, is_sparse=False):
        with tf.name_scope("Graph_Encoder"):
            if is_sparse:
                encoder = tf.sparse_tensor_dense_matmul(binput, self.I_B_Diag, name="XxI_B") 
                encoder += self.relu_with_threshold(tf.sparse_tensor_dense_matmul(binput, self.A, name="XxA"), beta)  
            else:
                encoder = tf.matmul(binput, self.I_B_Diag, name="XxI_B")
                encoder += self.relu_with_threshold(tf.matmul(binput, self.A, name="XxA"), beta) 
        return encoder

    def get_item_bias(self):
        return self.session.run(self.I_B)

    def get_sparse_tensor_info(self, x, is_bseq=False):
        indices = []
        if is_bseq:
            for sid, bseq in enumerate(x):
                for t, basket in enumerate(bseq):
                    for item_id in basket:
                        indices.append([sid, t, item_id])
        else:
            for bid, basket in enumerate(x):
                for item_id in basket:
                    indices.append([bid, item_id])

        values = np.ones(len(indices), dtype=np.float32)
        indices = np.array(indices, dtype=np.int32)

        return indices, values

    def compute_loss(self, logits, y):
        sigmoid_logits = tf.nn.sigmoid(logits)

        neg_y = (1.0 - y)
        pos_logits = y * logits

        pos_max = tf.reduce_max(pos_logits, axis=1)
        pos_max = tf.expand_dims(pos_max, axis=-1)

        pos_min = tf.reduce_min(pos_logits + neg_y * pos_max, axis=1)
        pos_min = tf.expand_dims(pos_min, axis=-1)

        nb_pos, nb_neg = tf.count_nonzero(y, axis=1), tf.count_nonzero(neg_y, axis=1)
        ratio = tf.cast(nb_neg, dtype=tf.float32) / tf.cast(nb_pos, dtype=tf.float32)

        pos_weight = tf.expand_dims(ratio, axis=-1)
        loss = y * -tf.log(sigmoid_logits) * pos_weight + neg_y * -tf.log(1.0 - tf.nn.sigmoid(logits - pos_min))

        return tf.reduce_mean(loss + 1e-8)
    
    def compute_recall_at_topk(self, k=10):
        top_k_preds = self.get_topk_tensor(self.predictions, k)
        correct_preds = tf.count_nonzero(tf.multiply(self.y, top_k_preds), axis=1)
        actual_bsize = tf.count_nonzero(self.y, axis=1)
        return tf.reduce_mean(tf.cast(correct_preds, dtype=tf.float32) / tf.cast(actual_bsize, dtype=tf.float32))
    
    # x -> shape(batch_size,N)
    def get_topk_tensor(self, x, k=10):
        _, index_cols = tf.nn.top_k(x, k)

        index_rows = tf.ones(shape=(self.batch_size, k), dtype=tf.int32) * tf.expand_dims(tf.range(0, self.batch_size), axis=-1)

        index_rows = tf.cast(tf.reshape(index_rows, shape=[-1]), dtype=tf.int64)
        index_cols = tf.cast(tf.reshape(index_cols, shape=[-1]), dtype=tf.int64)

        top_k_indices = tf.stack([index_rows, index_cols], axis=1)
        top_k_values = tf.ones(shape=[self.batch_size * k], dtype=tf.float32)

        sparse_tensor = tf.SparseTensor(indices=top_k_indices, values=top_k_values, dense_shape=[self.batch_size, self.nb_items])
        return tf.sparse_tensor_to_dense(tf.sparse_reorder(sparse_tensor))
    
    def relu_with_threshold(self, x, threshold):
        return tf.nn.relu(x - tf.abs(threshold))
    


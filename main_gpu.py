import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import os
import utils
import models
import procedure

# Parameters
# ###########################
# GPU & Seed

tf.flags.DEFINE_string("device_id", None, "GPU device is to be used in training (default: None)")
tf.flags.DEFINE_integer("seed", 2, "Seed value for reproducibility (default: 89)")

# Model hyper-parameters
tf.flags.DEFINE_string("data_dir", None, "The input data directory (default: None)")
tf.flags.DEFINE_string("output_dir", None, "The output directory (default: None)")
tf.flags.DEFINE_string("tensorboard_dir", None, "The tensorboard directory (default: None)")

tf.flags.DEFINE_integer("emb_dim", 2, "The dimensionality of embedding (default: 2)")
tf.flags.DEFINE_integer("rnn_unit", 4, "The number of hidden units of RNN (default: 4)")
tf.flags.DEFINE_integer("nb_hop", 1, "The number of neighbor hops  (default: 1)")
tf.flags.DEFINE_float("alpha", 0.5, "The reguralized hyper-parameter (default: 0.5)")

tf.flags.DEFINE_integer("matrix_type", 1, "The type of adjacency matrix (0=zero,1=real,default:1)")

# Training hyper-parameters
tf.flags.DEFINE_integer("nb_epoch", 15, "Number of epochs (default: 15)")
tf.flags.DEFINE_integer("early_stopping_k", 5, "Early stopping patience (default: 5)")
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate (default: 0.001)")
tf.flags.DEFINE_float("epsilon", 1e-8, "The epsilon threshold in training (default: 1e-8)")
tf.flags.DEFINE_float("dropout_rate", 0.3, "Dropout keep probability for RNN (default: 0.3)")
tf.flags.DEFINE_integer("batch_size", 32, "Batch size (default: 32)")
tf.flags.DEFINE_integer("display_step", 10, "Show loss/acc for every display_step batches (default: 10)")
tf.flags.DEFINE_string("rnn_cell_type", "LSTM", " RNN Cell Type like LSTM, GRU, etc. (default: LSTM)")
tf.flags.DEFINE_integer("top_k", 10, "Top K Accuracy (default: 10)")
tf.flags.DEFINE_boolean("train_mode", False, "Turn on/off the training mode (default: False)")
tf.flags.DEFINE_boolean("tune_mode", False, "Turn on/off the tunning mode (default: False)")
tf.flags.DEFINE_boolean("prediction_mode", False, "Turn on/off the testing mode (default: False)")

config = tf.flags.FLAGS
print("---------------------------------------------------")
print("SeedVal = " + str(config.seed))
print("\nParameters: " + str(config.__len__()))
for iterVal in config.__iter__():
    print(" + {}={}".format(iterVal, config.__getattr__(iterVal)))
print("Tensorflow version: ", tf.__version__)
print("---------------------------------------------------")

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = config.device_id

# for reproducibility
np.random.seed(config.seed)
tf.set_random_seed(config.seed)
tf.logging.set_verbosity(tf.logging.ERROR)

gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
gpu_config.log_device_placement = False

# ----------------------- MAIN PROGRAM -----------------------

data_dir = config.data_dir
output_dir = config.output_dir
tensorboard_dir=config.tensorboard_dir

training_file = data_dir + "/train.txt"
validate_file = data_dir + "/validate.txt"
testing_file = data_dir + "/test.txt"

print("***************************************************************************************")
print("Output Dir: " + output_dir)

# Create directories
print("@Create directories")
utils.create_folder(output_dir + "/models")
utils.create_folder(output_dir + "/topN")

if tensorboard_dir is not None:
    utils.create_folder(tensorboard_dir)

# Load train, validate & test
print("@Load train,validate&test data")
training_instances = utils.read_file_as_lines(training_file)
nb_train = len(training_instances)
print(" + Total training sequences: ", nb_train)

validate_instances = utils.read_file_as_lines(validate_file)
nb_validate = len(validate_instances)
print(" + Total validating sequences: ", nb_validate)

testing_instances = utils.read_file_as_lines(testing_file)
nb_test = len(testing_instances)
print(" + Total testing sequences: ", nb_test)

# Create dictionary
print("@Build knowledge")
MAX_SEQ_LENGTH, item_dict, rev_item_dict, item_probs = utils.build_knowledge(training_instances, validate_instances)

print("#Statistic")
NB_ITEMS = len(item_dict)
print(" + Maximum sequence length: ", MAX_SEQ_LENGTH)
print(" + Total items: ", NB_ITEMS)

matrix_type = config.matrix_type
if matrix_type == 0:
    print("@Create an zero adjacency matrix")
    adj_matrix = utils.create_zero_matrix(NB_ITEMS)
else:
    print("@Load the normalized adjacency matrix")
    matrix_fpath = data_dir + "/adj_matrix/r_matrix_" + str(config.nb_hop)+ "w.npz"
    adj_matrix = sp.load_npz(matrix_fpath)
    print(" + Real adj_matrix has been loaded from" + matrix_fpath)


print("@Compute #batches in train/validation/test")
total_train_batches = utils.compute_total_batches(nb_train, config.batch_size)
total_validate_batches = utils.compute_total_batches(nb_validate, config.batch_size)
total_test_batches = utils.compute_total_batches(nb_test, config.batch_size)
print(" + #batches in train ", total_train_batches)
print(" + #batches in validate ", total_validate_batches)
print(" + #batches in test ", total_test_batches)

model_dir = output_dir + "/models"
if config.train_mode:
    with tf.Session(config=gpu_config) as sess:
        # Training
        # ==================================================
        # Create data generator
        train_generator = utils.seq_batch_generator(training_instances, item_dict, config.batch_size)
        validate_generator = utils.seq_batch_generator(validate_instances, item_dict, config.batch_size, False)
        test_generator = utils.seq_batch_generator(testing_instances, item_dict, config.batch_size, False)
        
        # Initialize the network
        print(" + Initialize the network")
        net = models.Beacon(sess, config.emb_dim, config.rnn_unit, config.alpha, MAX_SEQ_LENGTH, item_probs, adj_matrix, config.top_k, 
                             config.batch_size, config.rnn_cell_type, config.dropout_rate, config.seed, config.learning_rate)

        print(" + Initialize parameters")
        sess.run(tf.global_variables_initializer())

        print("================== TRAINING ====================")
        print("@Start training")
        procedure.train_network(sess, net, train_generator, validate_generator, config.nb_epoch,
                                total_train_batches, total_validate_batches, config.display_step,
                                config.early_stopping_k, config.epsilon, tensorboard_dir, model_dir,
                                test_generator, total_test_batches)

        # Reset before re-load
    tf.reset_default_graph()

if config.prediction_mode or config.tune_mode:
    with tf.Session(config=gpu_config) as sess:
        print(" + Initialize the network")

        net = models.Beacon(sess, config.emb_dim, config.rnn_unit, config.alpha, MAX_SEQ_LENGTH, item_probs, adj_matrix, config.top_k, 
                        config.batch_size, config.rnn_cell_type, config.dropout_rate, config.seed, config.learning_rate)

        print(" + Initialize parameters")
        sess.run(tf.global_variables_initializer())

        print("===============================================\n")
        print("@Restore the model from " + model_dir)
        # Reload the best model
        saver = tf.train.Saver()
        recent_dir = utils.recent_model_dir(model_dir)
        saver.restore(sess, model_dir + "/" + recent_dir + "/model.ckpt")
        print("Model restored from file: %s" % recent_dir)

        # Tunning
        # ==================================================
        if config.tune_mode:
            print("@Start tunning")
            validate_generator = utils.seq_batch_generator(validate_instances, item_dict, config.batch_size, False)
            procedure.tune(net, validate_generator, total_validate_batches, config.display_step, output_dir + "/topN/val_recall.txt")

        # Testing
        # ==================================================
        if config.prediction_mode:
            test_generator = utils.seq_batch_generator(testing_instances, item_dict, config.batch_size, False)

            print("@Start generating prediction")
            procedure.generate_prediction(net, test_generator, total_test_batches, config.display_step, 
                        rev_item_dict, output_dir + "/topN/prediction.txt")
        
    tf.reset_default_graph()

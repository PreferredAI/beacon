import tensorflow as tf
import scipy.sparse as sp
import utils

# Model hyper-parameters
tf.flags.DEFINE_string("data_dir", None, "The input data directory (default: None)")
tf.flags.DEFINE_integer("nb_hop", 1, "The order of the real adjacency matrix (default:1)")

config = tf.flags.FLAGS
print("---------------------------------------------------")
print("Data_dir = " + str(config.data_dir))
print("\nParameters: " + str(config.__len__()))
for iterVal in config.__iter__():
    print(" + {}={}".format(iterVal, config.__getattr__(iterVal)))
print("Tensorflow version: ", tf.__version__)
print("---------------------------------------------------")

SEED_VALUES = [2, 9, 15, 44, 50, 55, 58, 79, 85, 92]

# ----------------------- MAIN PROGRAM -----------------------
data_dir = config.data_dir
output_dir = data_dir + "/adj_matrix"

training_file = data_dir + "/train.txt"
validate_file = data_dir + "/validate.txt"
print("***************************************************************************************")
print("Output Dir: " + output_dir)

print("@Create output directory")
utils.create_folder(output_dir)

# Load train, validate & test
print("@Load train,validate&test data")
training_instances = utils.read_file_as_lines(training_file)
nb_train = len(training_instances)
print(" + Total training sequences: ", nb_train)

validate_instances = utils.read_file_as_lines(validate_file)
nb_validate = len(validate_instances)
print(" + Total validating sequences: ", nb_validate)

# Create dictionary
print("@Build knowledge")
MAX_SEQ_LENGTH, item_dict, reversed_item_dict, _ = utils.build_knowledge(training_instances, validate_instances)

print("#Statistic")
NB_ITEMS = len(item_dict)
print(" + Maximum sequence length: ", MAX_SEQ_LENGTH)
print(" + Total items: ", NB_ITEMS)

rmatrix_fpath = output_dir + "/r_matrix_" + str(config.nb_hop) + "w.npz"

print("@Build the real adjacency matrix")
real_adj_matrix = utils.build_sparse_adjacency_matrix_v2(training_instances, validate_instances, item_dict)
real_adj_matrix = utils.normalize_adj(real_adj_matrix)

mul = real_adj_matrix
with tf.device('/cpu:0'):
    w_mul = real_adj_matrix
    coeff = 1.0
    for w in range(1, config.nb_hop):
        coeff *= 0.85
        w_mul *= real_adj_matrix
        w_mul = utils.remove_diag(w_mul)

        w_adj_matrix = utils.normalize_adj(w_mul)
        mul += coeff * w_adj_matrix

    real_adj_matrix = mul

    sp.save_npz(rmatrix_fpath, real_adj_matrix)
    print(" + Save adj_matrix to" + rmatrix_fpath)

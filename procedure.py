import tensorflow as tf
import numpy as np

import sys
import utils
import time


def train_network(sess, net, train_generator, validate_generator, nb_epoch, 
                  total_train_batches, total_validate_batches, display_step,
                  early_stopping_k, epsilon, tensorboard_dir, output_dir,
                  test_generator, total_test_batches):
    summary_writer = None
    if tensorboard_dir is not None:
        summary_writer = tf.summary.FileWriter(tensorboard_dir)
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    val_best_performance = [sys.float_info.max]
    patience_cnt = 0
    for epoch in range(0, nb_epoch):
        print("\n=========================================")
        print("@Epoch#" + str(epoch))

        train_loss = 0.0
        train_recall = 0.0

        for batch_id, data in train_generator:
            start_time = time.time()
            loss, recall, summary = net.train_batch(data['S'], data['L'], data['Y'])

            train_loss += loss
            avg_train_loss = train_loss / (batch_id + 1)

            train_recall += recall
            avg_train_recall = train_recall / (batch_id + 1)
            
            # Write logs at every iteration
            if summary_writer is not None:
                summary_writer.add_summary(summary, epoch * total_train_batches + batch_id)

                loss_sum = tf.Summary()
                loss_sum.value.add(tag="Losses/Train_Loss", simple_value=avg_train_loss)
                summary_writer.add_summary(loss_sum, epoch * total_train_batches + batch_id)

                recall_sum = tf.Summary()
                recall_sum.value.add(tag="Recalls/Train_Recall", simple_value=avg_train_recall)
                summary_writer.add_summary(recall_sum, epoch * total_train_batches + batch_id)


            if batch_id % display_step == 0 or batch_id == total_train_batches - 1:
                running_time = time.time() - start_time
                print("Training | Epoch " + str(epoch) + " | " + str(batch_id + 1) + "/" + str(total_train_batches) 
                    + " | Loss= " + "{:.8f}".format(avg_train_loss)  
                    + " | Recall@"+ str(net.top_k) + " = " + "{:.8f}".format(avg_train_recall) 
                    + " | Time={:.2f}".format(running_time) + "s")

            if batch_id >= total_train_batches - 1:
                break

        print("\n-------------- VALIDATION LOSS--------------------------")
        val_loss = 0.0
        val_recall = 0.0
        for batch_id, data in validate_generator:
            loss, recall, summary = net.validate_batch(data['S'], data['L'], data['Y'])
            
            val_loss += loss
            avg_val_loss = val_loss / (batch_id + 1)

            val_recall += recall
            avg_val_recall = val_recall / (batch_id + 1)

            # Write logs at every iteration
            if summary_writer is not None:
                summary_writer.add_summary(summary, epoch * total_validate_batches + batch_id)

                loss_sum = tf.Summary()
                loss_sum.value.add(tag="Losses/Val_Loss", simple_value=avg_val_loss)
                summary_writer.add_summary(loss_sum, epoch * total_validate_batches + batch_id)

                recall_sum = tf.Summary()
                recall_sum.value.add(tag="Recalls/Val_Recall", simple_value=avg_val_recall)
                summary_writer.add_summary(recall_sum, epoch * total_validate_batches + batch_id)

            if batch_id % display_step == 0 or batch_id == total_validate_batches - 1:
                print("Validating | Epoch " + str(epoch) + " | " + str(batch_id + 1) + "/" + str(total_validate_batches) 
                    + " | Loss = " + "{:.8f}".format(avg_val_loss)
                    + " | Recall@"+ str(net.top_k) + " = " + "{:.8f}".format(avg_val_recall))
            
            if batch_id >= total_validate_batches - 1:
                break

        print("\n-------------- TEST LOSS--------------------------")
        test_loss = 0.0
        test_recall = 0.0
        for batch_id, data in test_generator:
            loss, recall, _ = net.validate_batch(data['S'], data['L'], data['Y'])
            
            test_loss += loss
            avg_test_loss = test_loss / (batch_id + 1)

            test_recall += recall
            avg_test_recall = test_recall / (batch_id + 1)

            # Write logs at every iteration
            if summary_writer is not None:
                #summary_writer.add_summary(summary, epoch * total_test_batches + batch_id)

                loss_sum = tf.Summary()
                loss_sum.value.add(tag="Losses/Test_Loss", simple_value=avg_test_loss)
                summary_writer.add_summary(loss_sum, epoch * total_test_batches + batch_id)

                recall_sum = tf.Summary()
                recall_sum.value.add(tag="Recalls/Test_Recall", simple_value=avg_test_recall)
                summary_writer.add_summary(recall_sum, epoch * total_test_batches + batch_id)

            if batch_id % display_step == 0 or batch_id == total_test_batches - 1:
                print("Testing | Epoch " + str(epoch) + " | " + str(batch_id + 1) + "/" + str(total_test_batches) 
                    + " | Loss = " + "{:.8f}".format(avg_test_loss)
                    + " | Recall@"+ str(net.top_k) + " = " + "{:.8f}".format(avg_test_recall))
            
            if batch_id >= total_test_batches - 1:
                break

        if summary_writer is not None:
            I_B= net.get_item_bias()
            item_probs = net.item_probs

            I_B_corr = np.corrcoef(I_B, item_probs)
            I_B_summ = tf.Summary()
            I_B_summ.value.add(tag="CorrCoef/Item_Bias", simple_value=I_B_corr[1][0])
            summary_writer.add_summary(I_B_summ, epoch)

        avg_val_loss = val_loss / total_validate_batches
        print("\n@ The validation's loss = " + str(avg_val_loss))
        imprv_ratio = (val_best_performance[-1] - avg_val_loss)/val_best_performance[-1]
        if imprv_ratio > epsilon:
            print("# The validation's loss is improved from " + "{:.8f}".format(val_best_performance[-1]) + \
                  " to " + "{:.8f}".format(avg_val_loss))
            val_best_performance.append(avg_val_loss)

            patience_cnt = 0

            save_dir = output_dir + "/epoch_" + str(epoch)
            utils.create_folder(save_dir)

            save_path = saver.save(sess, save_dir + "/model.ckpt")
            print("The model is saved in: %s" % save_path)
        else:
            patience_cnt += 1

        if patience_cnt >= early_stopping_k:
            print("# The training is early stopped at Epoch " + str(epoch))
            break

def tune(net, data_generator, total_batches, display_step, output_file):
    f = open(output_file, "w")
    val_loss = 0.0
    val_recall = 0.0
    for batch_id, data in data_generator:
        loss, recall, _ = net.validate_batch(data['S'], data['L'], data['Y'])
            
        val_loss += loss
        avg_val_loss = val_loss / (batch_id + 1)

        val_recall += recall
        avg_val_recall = val_recall / (batch_id + 1)

        # Write logs at every iteration
        if batch_id % display_step == 0 or batch_id == total_batches - 1:
            print(str(batch_id + 1) + "/" + str(total_batches) + " | Loss = " + "{:.8f}".format(avg_val_loss)
                    + " | Recall@"+ str(net.top_k) + " = " + "{:.8f}".format(avg_val_recall))

        if batch_id >= total_batches - 1:
            break
    avg_val_recall = val_recall / total_batches
    f.write(str(avg_val_recall) + "\n")
    f.close()


def generate_prediction(net, data_generator, total_test_batches, display_step, inv_item_dict, output_file):
    f = open(output_file, "w")

    for batch_id, data in data_generator:
        values, indices = net.generate_prediction(data['S'], data['L'])

        for i, (seq_val, seq_ind) in enumerate(zip(values, indices)):
            f.write("Target:" + data['O'][i])

            for (v, idx) in zip(seq_val, seq_ind):
                f.write("|" + str(inv_item_dict[idx]) + ":" + str(v))

            f.write("\n")

        if batch_id % display_step == 0 or batch_id == total_test_batches - 1:
            print(str(batch_id + 1) + "/" + str(total_test_batches))

        if batch_id >= total_test_batches - 1:
            break
    f.close()
    print(" ==> PREDICTION HAS BEEN DONE!")


def recent_model_dir(dir):
    folder_list = utils.list_directory(dir, True)
    folder_list = sorted(folder_list, key=get_epoch)
    return folder_list[-1]


def get_epoch(x):
    idx = x.index('_') + 1
    return int(x[idx:])

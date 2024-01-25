# from typing_extensions import final
from model import SGKT
import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
import warnings
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
from data_process import DataGenerator
from data_process2 import DataGenerator2


def train(args, train_dkt):
    # Create a TensorFlow session
    run_config = tf.compat.v1.ConfigProto()
    # run_config.gpu_options.allow_growth = True
    warnings.filterwarnings("ignore")

    with tf.compat.v1.Session(config=run_config) as sess:
        print(args.model)

        # Create an instance of the SGKT model
        model = SGKT(args)
        saver = tf.compat.v1.train.Saver
        if train_dkt:
            # Initialize global variables and model save path
            sess.run(tf.global_variables_initializer())
            model_dir = save_model_dir(args)
            best_valid_auc = 0

            train_generator = DataGenerator(args.train_seqs, args.max_step, batch_size=args.batch_size,
                                            feature_size=args.feature_answer_size - 2,
                                            hist_num=args.hist_neighbor_num)
            train_generator2 = DataGenerator2(args.student_emotional_train, args.max_step, batch_size=args.batch_size,
                                              feature_size=args.feature_answer_size - 2, field_size=args.field_size2)


            valid_generator = DataGenerator(args.valid_seqs, args.max_step, batch_size=args.batch_size,
                                            feature_size=args.feature_answer_size - 2,
                                            hist_num=args.hist_neighbor_num)
            valid_generator2 = DataGenerator2(args.student_emotional_valid, args.max_step, batch_size=args.batch_size,
                                              feature_size=args.feature_answer_size - 2, field_size=args.field_size2)

            # Iteratively train the model
            for epoch in tqdm(range(args.num_epochs)):
                # Create data generators for the training and validation sets
                print("epoch:", epoch)
                overall_loss = 0
                train_generator.shuffle()
                preds, binary_preds, targets = list(), list(), list()
                train_step = 0
                # Train on the training set
                while not train_generator.end:
                    train_step += 1
                    [features_answer_index, target_answers, seq_lens, hist_neighbor_index] = train_generator.next_batch()
                    [features_emotional_index, seq_lens2] = train_generator2.next_batch()
                    # Call the model's training method, perform forward and backward propagation, and get binary predictions, probability predictions, and loss
                    binary_pred, pred, loss = model.train(sess, features_answer_index, target_answers, seq_lens, hist_neighbor_index, features_emotional_index)
                    overall_loss += loss
                    # Process each sequence and add the predicted results and true labels to the respective lists
                    for seq_idx, seq_len in enumerate(seq_lens):
                        preds.append(pred[seq_idx, 0:seq_len])
                        binary_preds.append(binary_pred[seq_idx, 0:seq_len])
                        targets.append(target_answers[seq_idx, 0:seq_len])
                # Concatenate the predicted results and true labels for all sequences and calculate metrics such as AUC and accuracy
                train_loss = overall_loss / train_step
                preds = np.concatenate(preds)
                binary_preds = np.concatenate(binary_preds)
                targets = np.concatenate(targets)
                auc_value = roc_auc_score(targets, preds)
                accuracy = accuracy_score(targets, binary_preds)
                precision, recall, f_score, _ = precision_recall_fscore_support(targets, binary_preds)
                print("\ntrain loss = {0}, auc={1}, accuracy={2}, precision={3}, recall={4}, f_score={5}".format(train_loss, auc_value, accuracy, precision, recall, f_score))
                # Record metrics during training to a log file
                write_log(args, model_dir, auc_value, accuracy, epoch, precision, recall, f_score, name='train_')

                # Validation
                # Evaluate on the validation set
                valid_generator.reset()
                preds, binary_preds, targets = list(), list(), list()
                valid_step = 0
                while not valid_generator.end:
                    valid_step += 1
                    [features_answer_index, target_answers, seq_lens, hist_neighbor_index] = valid_generator.next_batch()
                    [features_emotional_index, seq_lens2] = valid_generator2.next_batch()
                    # Call the model's evaluation method, perform forward propagation, and get predicted results
                    binary_pred, pred = model.evaluate(sess, features_answer_index, target_answers, seq_lens, hist_neighbor_index, features_emotional_index, valid_step)
                    # Process each sequence and add the predicted results and true labels to the respective lists
                    for seq_idx, seq_len in enumerate(seq_lens):
                        preds.append(pred[seq_idx, 0:seq_len])
                        binary_preds.append(binary_pred[seq_idx, 0:seq_len])
                        targets.append(target_answers[seq_idx, 0:seq_len])

                # Compute metrics
                # Concatenate the predicted results and true labels for all sequences and calculate metrics such as AUC and accuracy
                preds = np.concatenate(preds)
                binary_preds = np.concatenate(binary_preds)
                targets = np.concatenate(targets)
                auc_value = roc_auc_score(targets, preds)
                accuracy = accuracy_score(targets, binary_preds)
                precision, recall, f_score, _ = precision_recall_fscore_support(targets, binary_preds)
                print("\nvalid auc={0}, accuracy={1}, precision={2}, recall={3}, f_score={4}".format(auc_value, accuracy, precision,
                                                                                        recall, f_score))
                # Record metrics on the validation set to a log file
                write_log(args, model_dir, auc_value, accuracy, epoch, precision, recall, f_score, name='valid_')
                # Select the best model based on the AUC value on the validation set and save the model parameters
                if auc_value > best_valid_auc:
                    print('%3.4f to %3.4f' % (best_valid_auc, auc_value))
                    best_valid_auc = auc_value
                    best_epoch = epoch
                    checkpoint_dir = os.path.join(args.checkpoint_dir, model_dir)
                print(model_dir+"\t"+str(best_valid_auc))



def save(global_step, sess, checkpoint_dir, saver):
    model_name = 'SGKT'

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=global_step)
    print('Save checkpoint at %d' % (global_step))


def save_model_dir(args):
    return '{}_{}_{}lr_{}hop_{}sn_{}qn_{}hn_{}nn_{}_{}bound_{}keep_{}'.format(args.dataset,
                                                args.model,args.lr,args.n_hop,args.skill_neighbor_num,args.question_neighbor_num,args.hist_neighbor_num,\
                                                                     args.next_neighbor_num,args.sim_emb,args.att_bound,args.dropout_keep_probs,args.tag)


def write_log(args,model_dir,auc, accuracy, epoch, precision, recall, f_score, name='train_'):
    log_path = os.path.join(args.log_dir, name+model_dir+'.csv')
    if not os.path.exists(log_path):
        log_file = open(log_path, 'w')
        log_file.write('Epoch\tAuc\tAccuracy\tPrecision\tRecall\tF_score\n')
    else:
        log_file = open(log_path, 'a')

    log_file.write(str(epoch) + '\t' + str(auc) + '\t' + str(accuracy) + '\t' + str(precision) + '\t' +str(recall) + '\t' + str(f_score) + '\n')
    log_file.flush()
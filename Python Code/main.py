import argparse
import time
import os
import tensorflow as tf
import numpy as np
import warnings
from numpy.distutils.fcompiler import str2bool
from data_process import *
from data_process2 import *
from train import train
import json

def main():
    warnings.filterwarnings("ignore")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Set environment variable to suppress TensorFlow logs

    # Default values for training DKT model
    train_dkt = 1
    arg_parser = argparse.ArgumentParser(description="train dkt model")
    arg_parser.add_argument('--data_dir', type=str, default='data')  # The directory path where the data model is stored
    arg_parser.add_argument("--log_dir", type=str, default='logs')  # The directory path where the logs will be stored
    arg_parser.add_argument('--train', type=str2bool, default='t')  # A boolean flag indicating whether to perform training or not
    arg_parser.add_argument('--hidden_neurons', type=int, default=[200, 100]) # A list of integers representing the number of neurons in each hidden layer
    arg_parser.add_argument("--lr", type=float, default=0.00025) # The learning rate for the model optimizer
    arg_parser.add_argument("--lr_decay", type=float, default=0.92) # The learning rate decay factor used during training
    arg_parser.add_argument('--checkpoint_dir', type=str, default='checkpoint') # The directory path where the trained model checkpoints will be saved
    arg_parser.add_argument('--dropout_keep_probs', nargs='?', default=[0.8,0.8,1]) # A list of float values representing the keep probabilities for dropout regularization in each layer
    arg_parser.add_argument('--aggregator', type=str, default='sum') # A string indicating the type of aggregator used in model
    arg_parser.add_argument('--model', type=str, default='ssei') # A string indicating the type of model to use
    arg_parser.add_argument('--l2_weight', type=float, default=1e-8) # The L2 regularization weight applied to the model?
    arg_parser.add_argument('--limit_max_len',type=int,default=200) # An integer representing the maximum length of the input sequence
    arg_parser.add_argument('--limit_min_len',type=int,default=3)   # # An integer representing the minimum length of the input sequence
    arg_parser.add_argument('--dataset', type=str, default='assist12_3')    # A string indicating the dataset used for training
    arg_parser.add_argument("--field_size", type=int, default=3)    # An integer representing the field size for the input data
    arg_parser.add_argument("--field_size2", type=int, default=5)  # An integer representing the field size for the input data
    arg_parser.add_argument("--embedding_size", type=int, default=100)  # An integer representing the size of the embeddings
    arg_parser.add_argument("--max_step", type=int, default=200)    # An integer representing the maximum number of training steps
    arg_parser.add_argument("--input_trans_size", type=int, default=100)    # An integer representing the size of the input transformer
    arg_parser.add_argument("--batch_size", type=int, default=6)    # An integer representing the batch size used during training
    arg_parser.add_argument("--select_index", type=int, default=[0, 1, 2]) # A list of integers representing the indices used for selecting features
    arg_parser.add_argument("--select_index2", type=int, default=[0, 1, 2, 3, 4])
    arg_parser.add_argument('--num_epochs', type=int, default=8)    # An integer representing the number of epochs for training
    arg_parser.add_argument('--n_hop', type=int, default=3) # An integer representing the number of hops，当 n_hop = 1 时，图神经网络只考虑节点的一阶邻居（即直接相连的节点）的信息。
    arg_parser.add_argument('--skill_neighbor_num', type=int, default=4) # An integer representing the number of skill neighbors
    arg_parser.add_argument('--question_neighbor_num', type=int, default=4) # An integer representing the number of question neighbors
    arg_parser.add_argument('--hist_neighbor_num', type=int, default=3)  # An integer representing the number of history neighbors
    arg_parser.add_argument('--next_neighbor_num', type=int, default=4)  # An integer representing the number of next neighbors
    arg_parser.add_argument('--att_bound', type=float, default=0.7)  # A float representing the boundary value used in the top-k selection during filtering irrelevant embeddings
    arg_parser.add_argument('--sim_emb', type=str, default='question_emb')  # A string representing the type of embeddings used for similarity computation in the top-k selection
    args = arg_parser.parse_args()
    #args.dataset = dataset

    print(args.model)   # Print the model name

    # Create a unique tag for logging purposes
    tag_path = os.path.join("%s_tag.txt"%args.dataset)
    tag = time.time()
    args.tag = tag

    # Save the configuration to a JSON file
    config_name = 'logs/%f_config.json' % tag
    config = {}
    for k, v in vars(args).items():
        config[k] = vars(args)[k]
    jsObj = json.dumps(config)
    fileObject = open(config_name, 'w')
    fileObject.write(jsObj)
    fileObject.close()
    print("config: " + str(config))

    # Data processing and preparation
    args = data_process(args)
    args = data_process_withtime(args)
    print("data process over")

    # Start training the DKT model
    train(args, train_dkt)

    # Write the tag to a file for future reference
    log_file = open(tag_path, 'w')
    log_file.write(str(tag))


if __name__ == "__main__":
    main()

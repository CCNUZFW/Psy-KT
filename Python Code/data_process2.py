import numpy as np
import random
import csv
import os
import pandas as pd


def data_process_withtime(args):
    train_data_directory = os.path.join(args.data_dir, args.dataset, args.dataset + '_train_emotional.csv')
    valid_data_directory = os.path.join(args.data_dir, args.dataset, args.dataset + '_test_emotional.csv')
    test_data_directory = os.path.join(args.data_dir, args.dataset, args.dataset + '_test_emotional.csv')

    args.student_data_time_train, args.num_students, max_num_questions = load_data(train_data_directory)
    args.student_data_time_test, _, _ = load_data(test_data_directory)
    args.student_data_time_valid, _, _ = load_data(valid_data_directory)

    args.time_train, args.frustrated_train, args.confused_train, args.concentrating_train, args.bored_train = get_feature_size(train_data_directory)
    args.time_test, args.frustrated_test, args.confused_test, args.concentrating_test, args.bored_test = get_feature_size(test_data_directory)
    args.time_valid, args.frustrated_valid, args.confused_valid, args.concentrating_valid, args.bored_valid = get_feature_size(valid_data_directory)

    args.student_emotional_train = split_list(args.student_data_time_train)
    args.student_emotional_test = split_list(args.student_data_time_test)
    args.student_emotional_valid = split_list(args.student_data_time_valid)
    return args


def load_data(file_path):
    student_data = {}

    with open(file_path, 'r',) as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            user_id = row['user_id']
            ms_first_response = row['ms_first_response']
            frustrated = row['Average_confidence(FRUSTRATED)']
            confused = row['Average_confidence(CONFUSED)']
            concentrating = row['Average_confidence(CONCENTRATING)']
            bored = row['Average_confidence(BORED)']

            data_point = [ms_first_response, frustrated, confused, concentrating, bored]

            if user_id in student_data:
                student_data[user_id].append(data_point)
            else:
                student_data[user_id] = [data_point]

        max_list_count = max(len(data) for data in student_data.values())

        all_values = [value for value in student_data.values()]
        student_data_withtime = []
        for value in all_values:
            student_data_withtime.append(value)

        num_students = len(student_data_withtime)

    return student_data_withtime, num_students, max_list_count


def split_list(original_list, chunk_size=200):
    result_list = []
    split_num = 0

    for sublist in original_list:
        if len(sublist) <= chunk_size:
            result_list.append(sublist)
            continue
        else:
            if len(sublist) % chunk_size == 0:
                split_num = len(sublist) / chunk_size
            else:
                split_num = int(len(sublist) / chunk_size) + 1

            for i in range(0, int(split_num)):
                current = sublist[i*chunk_size: (i+1)*chunk_size]
                result_list.append(current)

    return result_list


def get_feature_size(file_path):
    df = pd.read_csv(file_path)
    time = df["ms_first_response"].max()
    frustrated = df["Average_confidence(FRUSTRATED)"].max()
    confused = df["Average_confidence(CONFUSED)"].max()
    concentrating = df["Average_confidence(CONCENTRATING)"].max()
    bored = df["Average_confidence(BORED)"].max()

    return time, frustrated, confused, concentrating, bored


def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break
    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)    # (nb_samples, maxlen, sample_shape)，是一个三维数组
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre': # maxlen!=none may need to truncating
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen+1]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)
        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))
        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def format_data(seqs, max_step, feature_size):
    seqs = seqs
    seq_lens = np.array(list(map(lambda seq: len(seq), seqs)))  # 用于记录每个序列的长度信息
    #[batch_size,max_len,feature_size]
    features_answer_index = pad_sequences(seqs, maxlen=max_step, padding='post', value=0)
    return features_answer_index, seq_lens


class DataGenerator2(object):

    def __init__(self, seqs, max_step, batch_size, feature_size, field_size):#feature_dkt
        np.random.seed(42)
        self.seqs = seqs
        self.max_step = max_step
        self.batch_size = batch_size
        self.batch_i = 0
        self.end = False
        self.feature_size = feature_size
        self.n_batch = int(np.ceil(len(seqs) / batch_size))
        self.field_size = field_size

    def next_batch(self):
        batch_seqs = self.seqs[self.batch_i*self.batch_size:(self.batch_i+1)*self.batch_size]
        self.batch_i+=1
        if self.batch_i == self.n_batch:
            self.end = True
        format_data_list = format_data(batch_seqs, self.max_step, self.feature_size)  # [feature_emotional_index,sequences_lens]
        return format_data_list

    def shuffle(self):
        self.pos = 0
        self.end = False
        np.random.shuffle(self.seqs)

    def reset(self):
        self.pos = 0
        self.end = False


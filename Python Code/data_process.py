import numpy as np
import random
import os
from numpy.lib.function_base import append

def data_process(args):
    # Load the path to the dataset and store it as a variable
    train_data_directory = os.path.join(args.data_dir, args.dataset, args.dataset + '_train.csv')
    valid_data_directory = os.path.join(args.data_dir, args.dataset, args.dataset + '_test.csv')
    test_data_directory = os.path.join(args.data_dir, args.dataset, args.dataset + '_test.csv')

    args.train_seqs, train_student_num, train_max_skill_id, train_max_question_id, feature_answer_id = load_data(train_data_directory, args.field_size, args.max_step)
    args.test_seqs, test_student_num, test_max_skill_id, test_max_question_id, feature_answer_id_valid = load_data(test_data_directory, args.field_size, args.max_step)
    args.valid_seqs, _, _, _, _ = load_data(valid_data_directory, args.field_size, args.max_step)

    print("original train seqs num: %d" % len(args.train_seqs))
    print("original test seqs num: %d"%len(args.test_seqs))

    # Each student has a corresponding skill-problem-answer list
    # the length of the list corresponding to each student (the number of answers) is stored in len
    lens = []
    for i in range(len(args.test_seqs)):
        lens.append(len(args.test_seqs[i]))

    args.skill_num = max(train_max_skill_id, test_max_skill_id)+1
    args.qs_num = max(train_max_question_id, test_max_question_id)+1
    print("args.qs_num: " + str(args.qs_num))
    args.question_num = args.qs_num-args.skill_num
    args.feature_answer_size = feature_answer_id + 1
    print("args.feature_answer_size: " + str(args.feature_answer_size))
    args.feature_answer_size_valid = feature_answer_id_valid + 1
    print("The skill num is: " + str(args.skill_num))
    print("The question num is: " + str(args.question_num))

    # load skill_matrix
    matrix_directory = os.path.join(args.data_dir, args.dataset, args.dataset + '_skill_matrix.txt')
    args.skill_matrix = np.loadtxt(matrix_directory) # multi-skill
    # Constructing a list of neighbors of a question and extracting relationships between skills and questions
    qs_adj_list, interactions = build_adj_list(args.train_seqs, args.test_seqs, args.skill_matrix, args.qs_num) #[[neighbor skill/question] for all qs]]
    args.question_neighbors, args.skill_neighbors = extract_qs_relations(qs_adj_list, args.skill_num, args.qs_num, args.question_neighbor_num, args.skill_neighbor_num)
    return args

def select_part_seqs(min_len,max_len,seqs):
    temp_seqs = []
    for seq in seqs:
        if len(seq)>=min_len and len(seq)<=max_len:
            temp_seqs.append(seq)

    print("seq num is: %d"%len(temp_seqs))
    return temp_seqs

def build_adj_list(train_seqs, test_seqs, skill_matrix, qs_num):
    # seqs: list - a list containing sequences of student data, each student data sequence is a two-dimensional list
    # skill_matrix: array - matrix of relationships between skills and questions, where 1 is a relationship and 0 is no relationship
    # qs_num: int - number of questions

    interactions = 0  # Used to record the number of interactions between questions and skills
    single_skill = []  # Used to store the indices of questions that only appear in skills
    adj_list = [[] for _ in range(qs_num)]  # Stores the neighbor node lists for each question
    save_question = []  # Used to temporarily store questions already traversed in the current student data sequence
    adj_num = [0 for _ in range(qs_num)]  # Records the number of neighbors for each question

    for seqs in [train_seqs, test_seqs]:  # Process the training and test sets of student data
        for seq in seqs:  # Traverse the student data sequences
            interactions += len(
                seq)  # Update the number of interactions, calculating the total interactions between questions and skills in the student data sequence
            save_question = []  # When processing a new student data sequence, clear the list of questions already traversed
            for step in seq:  # Traverse each time step in the student data sequence
                if step[1] not in save_question:  # If the current question has not been traversed yet
                    save_question.append(step[1])  # Add the current question to the list of questions already traversed
                # Add the neighbor nodes of the current question (including skills and other questions) to the adjacency list
                adj_list[step[1]] = np.reshape(np.argwhere(skill_matrix[step[0]] == 1), [-1]).tolist()
                adj_num[step[1]] += 1  # Update the number of neighbors for the current question
                # Traverse the neighbor nodes of the current question (skills)
                for skill_index in np.reshape(np.argwhere(skill_matrix[step[0]] == 1), [-1]).tolist():
                    adj_num[skill_index] += 1  # Update the number of neighbors for the neighbor node (skill)
                    if skill_index not in single_skill:  # If the neighbor node (skill) is not in the list of questions that only appear in skills
                        single_skill.append(
                            skill_index)  # Add the index of the neighbor node (skill) to the list of questions that only appear in skills
                    if step[1] not in adj_list[
                        skill_index]:  # If the current question is not in the neighbor node's (skill) adjacency list
                        adj_list[skill_index].append(
                            step[1])  # Add the current question to the neighbor node's (skill) adjacency list
            # Combine the questions in the current student data sequence pairwise, adding each other as neighbors to enhance the connections between questions
            for temp1 in save_question:
                for temp2 in save_question:
                    if temp1 != temp2 and temp2 not in adj_list[temp1]:
                        adj_list[temp1].append(temp2)

    return adj_list, interactions  # Return the adjacency list of questions and the number of interactions between questions and skills






def extract_qs_relations(qs_list, s_num, qs_num, q_neighbor_size, s_neighbor_size):
    # qs_list: list - Neighbor list of questions, a list of indices representing neighbors (including skills and other questions) for each question
    # s_num: int - Number of skills
    # qs_num: int - Number of questions
    # q_neighbor_size: int - Number of neighbors for questions, i.e., the number of neighbor nodes for each question
    # s_neighbor_size: int - Number of neighbors for skills, i.e., the number of neighbor nodes for each skill

    question_neighbors = np.zeros([qs_num, q_neighbor_size], dtype=np.int32)  # Array to store indices of neighbors for each question
    skill_neighbors = np.zeros([s_num, s_neighbor_size], dtype=np.int32)  # Array to store indices of neighbors for each skill

    s_num_dic = {}  # Dictionary to count different numbers of skill neighbors
    q_num_dic = {}  # Dictionary to count different numbers of question neighbors

    for index, neighbors in enumerate(qs_list):  # Traverse the neighbor list of questions, where index represents the question index, and neighbors are the indices of its neighbors
        if index < s_num:  # If it is a skill
            if len(neighbors) not in q_num_dic:
                q_num_dic[len(neighbors)] = 1
            else:
                q_num_dic[len(neighbors)] += 1
            if len(neighbors) > 0:  # If the number of neighbor nodes is greater than 0
                # Randomly select s_neighbor_size nodes from the list of neighbor nodes. If there are not enough nodes, perform replacement.
                if len(neighbors) >= s_neighbor_size:
                    skill_neighbors[index] = np.random.choice(neighbors, s_neighbor_size, replace=False)
                else:
                    skill_neighbors[index] = np.random.choice(neighbors, s_neighbor_size, replace=True)
        else:  # If it is a question
            if len(neighbors) not in s_num_dic:
                s_num_dic[len(neighbors)] = 1
            else:
                s_num_dic[len(neighbors)] += 1
            if len(neighbors) > 0:  # If the number of neighbor nodes is greater than 0
                if len(neighbors) >= q_neighbor_size:  # If the number of neighbor nodes is greater than or equal to q_neighbor_size
                    save_skill = []  # Save indices of skill neighbors
                    save_question = []  # Save indices of question neighbors
                    # Group neighbor nodes by skills and questions
                    for i in range(len(neighbors)):
                        if neighbors[i] < s_num:
                            save_skill.append(neighbors[i])
                        else:
                            save_question.append(neighbors[i])
                    if len(save_skill) >= q_neighbor_size:  # If there are q_neighbor_size or more skill neighbors
                        # Randomly select q_neighbor_size skill neighbors as neighbors for the question
                        question_neighbors[index] = np.random.choice(save_skill, q_neighbor_size, replace=False)
                    else:  # If there are fewer than q_neighbor_size skill neighbors
                        # Add all skill neighbors as neighbors for the question and supplement with q_neighbor_size-1 question neighbors
                        for i in range(len(save_skill)):
                            question_neighbors[index][i] = save_skill[i]
                        temp = np.random.choice(save_question, q_neighbor_size - len(save_skill) - 1, replace=False)
                        length = len(save_skill)
                        for i in range(len(temp)):
                            question_neighbors[index][i + length + 1] = temp[i]
                else:  # If the number of neighbor nodes is less than q_neighbor_size
                    # Randomly select q_neighbor_size nodes from the list of neighbor nodes with replacement
                    question_neighbors[index] = np.random.choice(neighbors, q_neighbor_size, replace=True)

    return question_neighbors, skill_neighbors  # Return the 2D arrays of indices for question neighbors and skill neighbors


def load_data(dataset_path, field_size, max_seq_len):
    seqs = []  # List to store processed student data sequences
    student_id = 0  # Student ID counter used to identify different students
    max_skill = -1  # Records the maximum skill number in the dataset
    max_question = -1  # Records the maximum question number in the dataset
    feature_answer_size = -1  # Records the maximum field (feature) number in the dataset

    with open(dataset_path, 'r') as f:
        feature_answer_list = []  # Used to temporarily store feature-answer data for each student
        for lineid, line in enumerate(f):  # Read data line by line
            fields = line.strip().strip(',')  # Remove leading and trailing spaces and commas
            i = lineid % (
                        field_size + 1)  # Calculate the position of the current line in the feature-answer data; i=0 indicates a new student
            if i != 0:  # i!=0 indicates that the current line contains feature-answer data for a student
                feature_answer_list.append(
                    list(map(int, fields.split(","))))  # Parse and add feature-answer data to the temporary list
            if i == 1:  # When i=1, it indicates that the "skill" feature data for a student has been read
                if max(feature_answer_list[-1]) > max_skill:
                    max_skill = max(feature_answer_list[-1])  # Update the maximum skill number
            elif i == 2:  # When i=2, it indicates that both "skill" and "problem" feature data for a student have been read
                if max(feature_answer_list[-1]) > max_question:
                    max_question = max(feature_answer_list[-1])  # Update the maximum question number
            elif i == field_size:  # When i=field_size, it indicates that all feature data for a student has been read
                student_id += 1  # Increment the student ID counter to identify the next student
                if max(feature_answer_list[-1]) > feature_answer_size:
                    feature_answer_size = max(feature_answer_list[-1])  # Update the maximum field number
                if len(feature_answer_list[
                           0]) > max_seq_len:  # If the feature sequence length for the current student exceeds the maximum length
                    n_split = len(feature_answer_list[0]) // max_seq_len
                    if len(feature_answer_list[0]) % max_seq_len:
                        n_split += 1
                else:
                    n_split = 1
                for k in range(n_split):  # Split the feature sequence into multiple sub-sequences of length max_seq_len
                    if k == n_split - 1:
                        end_index = len(feature_answer_list[0])
                    else:
                        end_index = (k + 1) * max_seq_len
                    split_list = []
                    for i in range(len(feature_answer_list)):
                        split_list.append(feature_answer_list[i][k * max_seq_len:end_index])
                    split_list = np.stack(split_list,
                                          1).tolist()  # Stack sub-sequences of multiple feature fields into a 2D list
                    seqs.append(split_list)  # Add the segmented student data sequence to the seqs list
                feature_answer_list = []  # Clear the temporary list to prepare for reading the next student's data

    return seqs, student_id, max_skill, max_question, feature_answer_size


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
    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
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

#select same skill index
def sample_hist_neighbors(seqs_size,max_step,hist_num,skill_index):
    #skill_index:[batch_size,max_step]
    #[batch_size,max_step,M]
    hist_neighbors_index = []
    for i in range(seqs_size):
        seq_hist_index = []
        seq_skill_index = skill_index[i]
        #[max_step,M]
        for j in range(1,max_step):
            same_skill_index = [k for k in range(j) if seq_skill_index[k] == seq_skill_index[j]]
            if hist_num != 0:
                #[0,j] select M
                if len(same_skill_index) >= hist_num:
                    seq_hist_index.append(np.random.choice(same_skill_index,hist_num, replace=False))
                else:
                    if len(same_skill_index)!= 0:
                        seq_hist_index.append(np.random.choice(same_skill_index,hist_num, replace=True))
                    else:
                        seq_hist_index.append(([max_step-1 for _ in range(hist_num)]))
            else:
                seq_hist_index.append([])
        hist_neighbors_index.append(seq_hist_index)
    return hist_neighbors_index


def format_data(seqs, max_step, feature_size, hist_num):
    seqs = seqs
    seq_lens = np.array(list(map(lambda seq: len(seq), seqs)))  # 用于记录每个序列的长度信息
    #[batch_size,max_len,feature_size]
    features_answer_index = pad_sequences(seqs, maxlen=max_step, padding='post', value=0)
    target_answers = pad_sequences(np.array([[j[-1] - feature_size for j in i[1:]] for i in seqs]), maxlen=max_step-1, padding='post', value=0)

    skills_index = features_answer_index[:,:,0]
    hist_neighbor_index = sample_hist_neighbors(len(seqs),max_step,hist_num,skills_index)#[batch_size,max_step,M]
    return features_answer_index, target_answers, seq_lens, hist_neighbor_index


class DataGenerator(object):

    def __init__(self, seqs, max_step, batch_size, feature_size, hist_num):#feature_dkt
        np.random.seed(42)
        self.seqs = seqs
        self.max_step = max_step
        self.batch_size = batch_size
        self.batch_i = 0
        self.end = False
        self.feature_size = feature_size
        self.n_batch = int(np.ceil(len(seqs) / batch_size))
        self.hist_num = hist_num

    def next_batch(self):
        batch_seqs = self.seqs[self.batch_i*self.batch_size:(self.batch_i+1)*self.batch_size]
        self.batch_i+=1
        if self.batch_i == self.n_batch:
            self.end = True
        format_data_list = format_data(batch_seqs, self.max_step, self.feature_size, self.hist_num)  # [feature_index,target_answers,sequences_lens,hist_neighbor_index]
        return format_data_list

    def shuffle(self):
        self.pos = 0
        self.end = False
        np.random.shuffle(self.seqs)

    def reset(self):
        self.pos = 0
        self.end = False

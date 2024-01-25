# encoding:utf-8
import tensorflow as tf
from aggregators import SumAggregator, ConcatAggregator
import math
import warnings
# import tensorflow.contrib as tf_contrib

class SGKT(object):
    def __init__(self, args):
        warnings.filterwarnings("ignore")
        # Initialize parameters and hyperparameters for the SGKT model
        self.args = args
        self.hidden_neurons = args.hidden_neurons  # Number of neurons in the hidden layers
        self.max_step = args.max_step - 1  # Maximum sequence length (subtracting 1 because the next step needs to be predicted)
        self.feature_answer_size = args.feature_answer_size  # Number of feature answers (total number of questions and skills, 53332)
        ## Additional features []---------------------------------------------
        self.select_index2 = args.select_index2
        self.feature_time_size = args.time_train  # Number of unique values for answer time (and similarly for the following features)
        self.feature_frustrated_size = args.frustrated_train
        self.feature_confused_size = args.confused_train
        self.feature_concentrating_size = args.concentrating_train
        self.feature_bored_size = args.bored_train
        self.field_size2 = args.field_size2  # 5, indicating 5 feature domains
        ## ---------------------------------------------
        self.field_size = args.field_size  # Dimensionality of features (number of fields)
        self.embedding_size = args.embedding_size  # Dimensionality of embedding vectors
        self.dropout_keep_probs = eval(str(args.dropout_keep_probs))  # Dropout retention probabilities
        self.select_index = args.select_index  # Selected embedding vector index
        self.hist_neighbor_num = args.hist_neighbor_num  # M - Number of historical neighbors (3)
        self.next_neighbor_num = args.next_neighbor_num  # N - Number of next-step neighbors (4)
        self.lr = args.lr  # Learning rate
        self.n_hop = args.n_hop  # Number of hops in the GNN
        self.question_neighbor_num = args.question_neighbor_num  # Number of question neighbors
        self.skill_neighbor_num = args.skill_neighbor_num  # Number of skill neighbors
        self.question_neighbors = args.question_neighbors  # List of question neighbor indices
        self.skill_neighbors = args.skill_neighbors  # List of skill neighbor indices

        # Add part of the skill neighbors to the question neighbors to share some neighbor nodes between questions and skills
        for i in range(len(self.skill_neighbors)):
            for j in range(len(self.question_neighbors[i])):
                self.question_neighbors[i][j] = self.skill_neighbors[i][j]

        # Create placeholders to receive input data and hyperparameters
        self.keep_prob = tf.compat.v1.placeholder(tf.float32)  # Dropout retention probability placeholder
        self.keep_prob_gnn = tf.compat.v1.placeholder(tf.float32)  # Dropout retention probability placeholder for GNN
        self.is_training = tf.compat.v1.placeholder(tf.bool)  # Placeholder for whether the model is in training mode
        self.features_answer_index = tf.compat.v1.placeholder(tf.int32, [None, self.max_step + 1,
                                                                         self.field_size])  # Placeholder for feature answer indices
        self.target_answers = tf.compat.v1.placeholder(tf.float32, [None,
                                                                    self.max_step])  # Placeholder for target answers (predicting the next step)
        self.sequence_lens = tf.compat.v1.placeholder(tf.int32, [None])  # Placeholder for sequence lengths
        self.hist_neighbor_index = tf.compat.v1.placeholder(tf.int32, [None, self.max_step,
                                                                       self.hist_neighbor_num])  # Placeholder for historical neighbor indices
        self.batch_size = tf.shape(self.features_answer_index)[0]  # Batch size
        self.feature_embedding = tf.compat.v1.get_variable("feature_embedding",
                                                           [self.feature_answer_size, self.embedding_size],
                                                           initializer=tf.contrib.layers.xavier_initializer())  # Variable for feature embedding vectors
        ## Additional features -----------------------------------------------
        self.features_emotional_index = tf.compat.v1.placeholder(tf.int32, [None, self.max_step + 1,
                                                                            self.field_size2])  ## Placeholder for emotional indices (6, 200, 5)
        self.feature_time_embedding = tf.compat.v1.get_variable("feature_time_embedding",
                                                                [self.feature_time_size + 1, self.embedding_size],
                                                                initializer=tf.contrib.layers.xavier_initializer())
        self.feature_frustrated_embedding = tf.compat.v1.get_variable("feature_frustrated_embedding",
                                                                      [self.feature_frustrated_size + 1,
                                                                       self.embedding_size],
                                                                      initializer=tf.contrib.layers.xavier_initializer())
        self.feature_confused_embedding = tf.compat.v1.get_variable("feature_confused_embedding",
                                                                    [self.feature_confused_size + 1,
                                                                     self.embedding_size],
                                                                    initializer=tf.contrib.layers.xavier_initializer())
        self.feature_concentrating_embedding = tf.compat.v1.get_variable("feature_concentrating_embedding",
                                                                         [self.feature_concentrating_size + 1,
                                                                          self.embedding_size],
                                                                         initializer=tf.contrib.layers.xavier_initializer())
        self.feature_bored_embedding = tf.compat.v1.get_variable("feature_bored_embedding",
                                                                 [self.feature_bored_size + 1, self.embedding_size],
                                                                 initializer=tf.contrib.layers.xavier_initializer())
        ## -----------------------------------------------------------------

        ## Placeholder for difficulty calculation
        self.feature_trans_embedding = tf.compat.v1.placeholder(tf.float32,
                                                                shape=[None, self.max_step, self.embedding_size])
        ## Placeholder for emotion
        self.stdv = 1.0 / math.sqrt(self.hidden_neurons[-1])  # Standard deviation for weight initialization

        # Initialize weights and biases for each layer
        self.W_in = tf.get_variable('W_in', shape=[self.embedding_size, self.embedding_size], dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.b_in = tf.get_variable('b_in', [self.embedding_size], dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.W_out = tf.get_variable('W_out', [self.embedding_size, self.embedding_size], dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.b_out = tf.get_variable('b_out', [self.embedding_size], dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.nasr_w1 = tf.get_variable('nasr_w1', [self.embedding_size, self.embedding_size], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.nasr_w2 = tf.get_variable('nasr_w2', [self.embedding_size, self.embedding_size], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.nasr_v = tf.get_variable('nasr_v', [self.embedding_size, self.embedding_size], dtype=tf.float32,
                                      initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.nasr_b = tf.get_variable('nasr_b', [self.embedding_size], dtype=tf.float32,
                                      initializer=tf.zeros_initializer())
        self.nasr_w3 = tf.get_variable('nasr_w3', [self.embedding_size * 2, self.embedding_size], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))

        ## Weights for emotion
        self.W_2 = tf.get_variable('W_2', shape=[self.embedding_size, self.embedding_size], dtype=tf.float32,
                                   initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.b_2 = tf.get_variable('b_2', [self.embedding_size], dtype=tf.float32,
                                   initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.W_3 = tf.get_variable('W_3', shape=[self.embedding_size, self.embedding_size], dtype=tf.float32,
                                   initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.b_3 = tf.get_variable('b_3', [self.embedding_size], dtype=tf.float32,
                                   initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        ## Weights for difficulty analysis
        self.W_d = tf.get_variable('W_d', shape=[self.embedding_size, self.embedding_size], dtype=tf.float32,
                                   initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.b_d = tf.get_variable('b_d', [self.embedding_size], dtype=tf.float32,
                                   initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        ## Beta
        self.W_beta = tf.get_variable('W_beta', shape=[self.embedding_size, self.embedding_size], dtype=tf.float32,
                                      initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.b_beta = tf.get_variable('b_beta', [self.embedding_size], dtype=tf.float32,
                                      initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        ## Theta
        self.W_theta = tf.get_variable('W_theta', shape=[self.embedding_size, self.embedding_size], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.b_theta = tf.get_variable('b_theta', [self.embedding_size], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        ## Alpha
        self.W_alpha = tf.get_variable('W_alpha', shape=[self.embedding_size, self.embedding_size], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.b_alpha = tf.get_variable('b_alpha', [self.embedding_size], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        ## p
        self.W_p = tf.get_variable('W_p', shape=[self.embedding_size, self.embedding_size], dtype=tf.float32,
                                   initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.b_p = tf.get_variable('b_p', [self.embedding_size], dtype=tf.float32,
                                   initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))

        if args.aggregator == 'sum':
            self.aggregator_class = SumAggregator
        elif args.aggregator == 'concat':
            self.aggregator_class = ConcatAggregator
        else:
            raise Exception("Unknown aggregator: " + args.aggregator)

        # bulid model
        self.build_model()


    def build_model(self):
        # The last dimension of self.features_answer_index is 3, indicating each 2D matrix has 3 columns, where the 0th column represents the skill index, the 1st column represents the question index, and the 2nd column represents the answer index.
        hidden_size = self.hidden_neurons[-1]  # Size (dimension) of the last element in the hidden_neurons list.
        select_feature_index = tf.gather(self.features_answer_index, self.select_index,
                                         axis=-1)  # Select specific features from the self.features_answer_index tensor using the indices specified in self.select_index.
        select_size = len(self.select_index)
        questions_index = select_feature_index[:, :-1,
                          1]  # Extract question indices from the select_feature_index tensor, excluding the last element in each row. It represents the indices of questions in the current step. (?,199)
        next_questions_index = select_feature_index[:, 1:,
                               1]  # It represents the indices of questions in the next step. (?,199)
        skill_index = select_feature_index[:, :-1, 0]  # (?,199)
        next_skill_index = select_feature_index[:, 1:, 0]  # (?,199)
        input_answers_index = select_feature_index[:, :-1, -1]  # (?,199)
        self.input_questions_embedding = tf.nn.embedding_lookup(self.feature_embedding,
                                                                questions_index)  # [batch_size, max_step, emb] 6, 199, 100
        self.next_questions_embedding = tf.nn.embedding_lookup(self.feature_embedding,
                                                               next_questions_index)  # [batch_size, max_step, emb]
        self.input_skills_embedding = tf.nn.embedding_lookup(self.feature_embedding,
                                                             skill_index)  # [batch_size, max_step, emb]
        self.next_skills_embedding = tf.nn.embedding_lookup(self.feature_embedding,
                                                            next_skill_index)  # [batch_size, max_step, emb]
        # Vector mapping for student answers
        input_answers_embedding = tf.nn.embedding_lookup(self.feature_embedding,
                                                         input_answers_index)  # [batch_size, max_step, emb]

        ## Regarding the newly added features -------------------------------------
        select_feature_index2 = tf.gather(self.features_emotional_index, self.select_index2, axis=-1)
        time_index = select_feature_index2[:, :-1, 0]
        frustrated_index = select_feature_index2[:, :-1, 1]  # 6,199
        confused_index = select_feature_index2[:, :-1, 2]
        concentrating_index = select_feature_index2[:, :-1, 3]
        bored_index = select_feature_index2[:, :-1, 4]

        # It's time to extract features for the next step. After feature extraction, consider how to concatenate them, taking into account the time step.
        self.input_time_embedding = tf.nn.embedding_lookup(self.feature_time_embedding,
                                                           time_index)  # [batch_size, max_step, emb] 6, 199, 100
        self.input_frustrated_embedding = tf.nn.embedding_lookup(self.feature_frustrated_embedding, frustrated_index)
        self.input_confused_embedding = tf.nn.embedding_lookup(self.feature_confused_embedding, confused_index)
        self.input_concentrating_embedding = tf.nn.embedding_lookup(self.feature_concentrating_embedding,
                                                                    concentrating_index)
        self.input_bored_embedding = tf.nn.embedding_lookup(self.feature_bored_embedding, bored_index)
        ## ------------------------------------------------------------------------

        # HRG Embedding Module
        # gcn
        if self.n_hop > 0:
            # Obtain neighbors' indices for each question index using the get_neighbors function
            input_neighbors = self.get_neighbors(self.n_hop,
                                                 questions_index)  # [[batch_size,seq_len],[batch_size,seq_len,q_neighbor_num],[batch_size,seq_len,q_neighbor_num*s_neighbor_num]
            # Aggregate question vector mappings and neighbor vector mappings. aggregate_embedding contains the aggregated vectors, encoding relationships between each question and its neighbors.
            aggregate_embedding, self.aggregators = self.aggregate(input_neighbors, self.input_questions_embedding)
            # Obtain neighbors' indices for each next question index using the get_neighbors function
            next_input_neighbors = self.get_neighbors(self.n_hop,
                                                      next_questions_index)  # [[batch_size,seq_len],[batch_size,seq_len,q_neighbor_num],[batch_size,seq_len,q_neighbor_num*s_neighbor_num]
            # Aggregate next question vector mappings and neighbor vector mappings. next_aggregate_embedding contains the aggregated vectors, encoding relationships between each question and its neighbors.
            next_aggregate_embedding, self.aggregators = self.aggregate(next_input_neighbors,
                                                                        self.next_questions_embedding)

            # Linear transformation and ReLU activation for question vector mappings, converting them into vectors of size hidden_size. The transformed vectors are named feature_trans_embedding and next_trans_embedding, with shapes [batch_size, max_step, hidden_size].
            feature_emb_size = self.embedding_size
            self.feature_trans_embedding = tf.reshape(
                tf.layers.dense(tf.reshape(aggregate_embedding[0], [-1, feature_emb_size]), hidden_size,
                                activation=tf.nn.relu, name='feature_layer', reuse=False),
                [-1, self.max_step, hidden_size])  # [batch_size,max_step,hidden_size]
            self.next_trans_embedding = tf.reshape(
                tf.layers.dense(tf.reshape(next_aggregate_embedding[0], [-1, feature_emb_size]), hidden_size,
                                activation=tf.nn.relu, name='feature_layer', reuse=True),
                [-1, self.max_step, hidden_size])  # [batch_size,max_step,hidden_size]
        else:
            feature_emb_size = self.embedding_size
            # Linear transformation and ReLU activation for question vector mappings without graph convolution, converting them into vectors of size hidden_size. The transformed vectors are named feature_trans_embedding, with shape [batch_size, max_step, hidden_size].
            self.feature_trans_embedding = tf.reshape(
                tf.layers.dense(tf.reshape(self.input_questions_embedding, [-1, feature_emb_size]), hidden_size,
                                activation=tf.nn.relu, name='feature_layer', reuse=False),
                [-1, self.max_step, hidden_size])  # [batch_size,max_step,hidden_size]
            next_trans_embedding = tf.reshape(
                tf.layers.dense(tf.reshape(self.next_questions_embedding, [-1, feature_emb_size]), hidden_size,
                                activation=tf.nn.relu, name='feature_layer', reuse=True),
                [-1, self.max_step, hidden_size])  # [batch_size,max_step,hidden_size]
            input_neighbors = self.get_neighbors(1,
                                                 questions_index)  # [[batch_size,seq_len],[batch_size,seq_len,q_neighbor_num],[batch_size,seq_len,q_neighbor_num*s_neighbor_num]
            next_input_neighbors = self.get_neighbors(1,
                                                      next_questions_index)  # [[batch_size,seq_len],[batch_size,seq_len,q_neighbor_num],[batch_size,seq_len,q_neighbor_num*s_neighbor_num]
            # Prepare next_aggregate_embedding and aggregate_embedding without graph convolution for later use
            next_aggregate_embedding = [next_trans_embedding, tf.reshape(
                tf.gather(self.feature_embedding, tf.reshape(next_input_neighbors[-1], [-1])),
                [self.batch_size, self.max_step, -1, self.embedding_size])]
            aggregate_embedding = [self.feature_trans_embedding, tf.reshape(
                tf.gather(self.feature_embedding, tf.reshape(input_neighbors[-1], [-1])),
                [self.batch_size, self.max_step, -1, self.embedding_size])]

        # Concatenate question vector mappings and answer vector mappings, and then map the concatenated vector to a new dimension through a fully connected layer. Finally, obtain the combined vector mapping for questions and answers.
        input_fa_embedding = tf.reshape(tf.concat([self.feature_trans_embedding, input_answers_embedding], -1),
                                        [-1, hidden_size + self.embedding_size])  # embedding_size*2
        input_trans_embedding = tf.reshape(tf.layers.dense(input_fa_embedding, hidden_size),
                                           [-1, self.max_step, hidden_size])

        # Used to calculate vector mappings representing historical neighbor features
        if self.args.model == "ssei":
            if self.args.sim_emb == "skill_emb":
                # Calculate historical neighbor feature vector mappings using input skill embeddings (self.input_skills_embedding), next skill embeddings (self.next_skills_embedding), and input_transformed_embedding
                self.hist_neighbors_features = self.hist_neighbor_sampler1(self.input_skills_embedding,
                                                                           self.next_skills_embedding,
                                                                           input_trans_embedding)  # [self.batch_size,max_step,self.hist_neighbor_num,hidden_size] 6 199 3 100
            # Calculate historical neighbor feature vector mappings using input question embeddings (self.input_questions_embedding), next question embeddings (self.next_questions_embedding), and input_transformed_embedding
            elif self.args.sim_emb == "question_emb":
                # (batch_size, max_step, hist_neighbor_num, qa_emb.shape[-1])
                self.hist_neighbors_features = self.hist_neighbor_sampler1(self.input_questions_embedding,
                                                                           self.next_questions_embedding,
                                                                           input_trans_embedding)  # [self.batch_size,max_step,self.hist_neighbor_num,hidden_size]
            else:
                # Calculate historical neighbor feature vector mappings using input embeddings (self.feature_trans_embedding), next embeddings (self.next_trans_embedding), and input_transformed_embedding
                self.hist_neighbors_features = self.hist_neighbor_sampler1(self.feature_trans_embedding,
                                                                           self.next_trans_embedding,
                                                                           input_trans_embedding)  # [self.batch_size,max_step,self.hist_neighbor_num,hidden_size]

        # SG Embedding Module
        #GGNN
        self.output_series = []
        fin_state = tf.nn.embedding_lookup(self.feature_embedding, input_answers_index[:, 0])   # ?, 100
        cell = tf.nn.rnn_cell.GRUCell(self.embedding_size, dtype=tf.float32)    # 创建一个GRU单元
        with tf.variable_scope('gru'):
            for i in range(self.max_step):
                # Extract corresponding states from the embedding matrix based on question and answer indices
                question_state = tf.nn.embedding_lookup(self.feature_embedding, questions_index[:, i])  # ?, 100
                answer_state = tf.nn.embedding_lookup(self.feature_embedding, input_answers_index[:, i])  # ?, 100
                # Obtain the state through linear transformation of in_state using different weight matrices
                in_state = question_state + answer_state + input_trans_embedding[:, i, :]  # ?, 100
                fin_state_in = tf.reshape(tf.matmul(in_state, self.W_in) + self.b_in, [-1, self.embedding_size])
                fin_state_out = tf.reshape(tf.matmul(in_state, self.W_out) + self.b_out, [-1, self.embedding_size])
                # Concatenate fin_state_in and fin_state_out, representing the combination of input and transformed states
                av = tf.concat([fin_state_in, fin_state_out], axis=-1)
                # Add to in_state to capture more input information
                av = av + tf.concat([in_state, in_state], axis=-1)
                # The dynamic_rnn function is used to pass the current state av (c in the paper) and the initial state fin_state to the GRU unit, calculating the current time step's output state and the next time step's hidden state
                state_output, fin_state = \
                    tf.nn.dynamic_rnn(cell, tf.expand_dims(tf.reshape(av, [-1, 2 * self.embedding_size]), axis=1),
                                      initial_state=tf.reshape(fin_state, [-1, self.embedding_size]))

                ## Add Emotional Features -------------------------------------------------------------------------
                emotional_feature = self.input_time_embedding + self.input_frustrated_embedding + self.input_confused_embedding + self.input_concentrating_embedding + self.input_bored_embedding
                state_output_with_emotional = tf.concat(
                    [state_output, tf.expand_dims(emotional_feature[:, i, :], axis=1)], axis=-1)
                state_output_with_emotional = tf.compat.v1.layers.dense(state_output_with_emotional,
                                                                        units=self.embedding_size)

                gamma = tf.tanh(tf.matmul(state_output_with_emotional, self.W_2) + self.b_2)
                lg = tf.sigmoid(tf.matmul(state_output_with_emotional, self.W_3) + self.b_3)

                state = in_state
                learn_gates = lg * ((gamma + 1) / 2) * tf.expand_dims(state, axis=1)
                ## ------------------------------------------------------------------------------------

                # Append the output state of the current time step to the list
                self.output_series.append(learn_gates)
            # The output_series contains the output states of each time step, representing the sequence representation in the GGNN and GRU modules' propagation and update process
            self.output_series = tf.reshape(tf.concat(self.output_series, 1), [-1, self.max_step,
                                                                               self.embedding_size])  # (batch_size, max_step, self.embedding_size) 6, 199, 100

        #Self-attention with FM Module
        # (batch_size, max_step, hist_neighbor_num, input_embedding.shape[-1])
        E_ansewring_states = self.hist_neighbor_sampler(input_trans_embedding)


        ## difficulty ------------------------------------------------------
        d_t = tf.reshape(tf.matmul(self.feature_trans_embedding, self.W_d) + self.b_d, [-1, self.max_step, self.embedding_size])

        ## IRT
        beta = tf.reshape(tf.tanh(tf.matmul(d_t, self.W_beta) + self.b_beta),  [-1, self.max_step, self.embedding_size])# 6 199 100
        theta = tf.reshape(tf.tanh(tf.matmul(self.output_series, self.W_theta) + self.b_theta), [-1, self.max_step, self.embedding_size])# 6 199 100
        alpha = tf.reshape(tf.sigmoid(tf.matmul(tf.reshape(self.hist_neighbors_features + E_ansewring_states, [-1, self.max_step, self.embedding_size]), self.W_alpha) + self.b_alpha), [-1, self.max_step, self.embedding_size])

        #Prediction Module
        if self.next_neighbor_num != 0:
            Nn = self.next_neighbor_sampler(next_aggregate_embedding)  # [batch_size,max_step,N+1,embedding_size]
            Nn = tf.concat([tf.expand_dims(self.next_trans_embedding, 2), Nn], -2)
            next_neighbor_num = self.next_neighbor_num+1
        else:
            Nn = tf.expand_dims(self.next_trans_embedding, 2)
            next_neighbor_num = 1

        if self.hist_neighbor_num != 0:
            Nh = tf.concat([tf.expand_dims(self.output_series, 2), self.hist_neighbors_features + E_ansewring_states], 2)  # [self.batch_size,max_step,M+1,feature_trans_size]]
            logits = tf.reduce_sum(tf.expand_dims(Nh, 3) * tf.expand_dims(Nn, 2), axis=4)  # [-1,max_step,Nh,1,emb_size]*[-1,max_step,1,Nn,emb_size]
            logits = tf.reshape(logits, [-1, self.max_step, (self.hist_neighbor_num + 1) * next_neighbor_num])  # ====>[batch_size,max_step,Nu*Nv]
        else:
            Nh = tf.expand_dims(self.output_series, 2)  # [self.batch_size,max_step,1,feature_trans_size]
            logits = tf.reduce_sum(tf.expand_dims(Nh, 3) * tf.expand_dims(Nn, 2), axis=4)  # [-1,max_step,Nh,1,emb_size]*[-1,max_step,1,Nn,emb_size]
            logits = tf.reshape(logits, [-1, self.max_step, 1 * next_neighbor_num])  # ====>[batch_size,max_step,Nu*Nv]

        with tf.variable_scope('ni'):
            w1 = tf.get_variable('atn_weights_1', [hidden_size, 1], initializer=tf.contrib.layers.xavier_initializer())
            w2 = tf.get_variable('atn_weights_2', [hidden_size, 1], initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.get_variable('atn_bias_1', [1], initializer=tf.zeros_initializer())
            b2 = tf.get_variable('atn_bias_2', [1], initializer=tf.zeros_initializer())

        if select_size > 3:
            f1 = tf.reshape(tf.matmul(tf.reshape(Nh, [-1, hidden_size]), w1) + b1,
                            [-1, self.max_step, self.hist_neighbor_num + 1, 1])
            f2 = tf.reshape(tf.matmul(tf.reshape(Nn, [-1, hidden_size]), w2) + b2,
                            [-1, self.max_step, 1, next_neighbor_num])
            coefs = tf.nn.softmax(tf.nn.tanh(
                tf.reshape(f1 + f2, [-1, self.max_step, (self.hist_neighbor_num + 1) * next_neighbor_num])))  # temp=10
        else:
            f1 = tf.reshape(tf.matmul(tf.reshape(Nh, [-1, hidden_size]), w1) + b1,
                            [-1, self.max_step, self.hist_neighbor_num + 1, 1])
            f2 = tf.reshape(tf.matmul(tf.reshape(Nn, [-1, hidden_size]), w2) + b2,
                            [-1, self.max_step, 1, next_neighbor_num])
            coefs = tf.nn.softmax(tf.nn.tanh(
                tf.reshape(f1 + f2, [-1, self.max_step, (self.hist_neighbor_num + 1) * next_neighbor_num])))  # temp=10

        self.alpha = alpha[:self.batch_size, :, :]
        self.bt = theta - beta
        T_abt = self.alpha * self.bt
        p = tf.matmul(T_abt, self.W_p) + self.b_p
        p = tf.compat.v1.layers.dense(p, units=(self.hist_neighbor_num + 1) * next_neighbor_num)

        clp = p
        self.logits = tf.reduce_sum(clp, axis=-1)
        self.flat_target_logits = flat_target_logits = tf.reshape(self.logits, [-1])
        self.flat_target_correctness = tf.reshape(self.target_answers, [-1])
        self.pred = tf.sigmoid(tf.reshape(flat_target_logits, [-1, self.max_step]))
        self.binary_pred = tf.cast(tf.greater_equal(self.pred, 0.5), tf.int32)
        self.filling_seqs = tf.cast(tf.sequence_mask(self.sequence_lens - 1, self.max_step),
                                    dtype=tf.float32)  # [batch_size,seq_len]
        index = tf.where(tf.not_equal(tf.reshape(self.filling_seqs, [-1]), tf.constant(0, dtype=tf.float32)))
        clear_flat_target_logits = tf.gather(self.flat_target_logits, index)
        clear_flat_target_correctness = tf.gather(self.flat_target_correctness, index)

        self.loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=clear_flat_target_correctness,
                                                                          logits=clear_flat_target_logits))
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        trainable_vars = tf.compat.v1.trainable_variables()
        self.grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, trainable_vars), 50)
        self.train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr,
                                               beta1=0.9, beta2=0.999, epsilon=1e-8). \
            minimize(self.loss, global_step=self.global_step)
        self.finalTrainOp = tf.compat.v1.train.AdamOptimizer(learning_rate=0.8,
                                               beta1=0.9, beta2=0.999, epsilon=1e-8). \
            minimize(self.loss, global_step=self.global_step)
        print("initialize complete")

    # Self-attention with FM Module
    def hist_neighbor_sampler(self, input_embedding):
        temp = []
        # Define learnable parameters
        x = tf.get_variable('xita', shape=[self.max_step], dtype=tf.float32,
                            initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        xt1 = tf.get_variable('xt1', shape=[self.max_step, self.max_step], dtype=tf.float32,
                              initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        xt2 = tf.get_variable('xt2', shape=[self.max_step, self.max_step], dtype=tf.float32,
                              initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))

        # Calculate features for each vector mapping
        for i in range(self.hidden_neurons[-1]):
            temp.append(input_embedding[:, :, i] - input_embedding[:, :, 0])

        # Reshape the embedding and construct a new vector mapping
        input_embedding = tf.reshape(tf.stack(temp), [self.batch_size, self.max_step, self.hidden_neurons[-1]])
        zero_embeddings = tf.expand_dims(tf.zeros([self.batch_size, self.hidden_neurons[-1]], dtype=tf.float32),
                                         1)  # [batch_size,1,hidden_size]
        input_embedding = tf.concat([input_embedding, zero_embeddings], 1)  # [batch_size,max_step+1,hidden_size]

        # Apply additional transformations input_embedding:[batch_size,max_step,fa_trans_size]
        b = tf.get_variable('bias', shape=[self.max_step + 1, self.hidden_neurons[-1]], dtype=tf.float32,
                            initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        K = tf.get_variable('K', shape=[self.hidden_neurons[-1], self.hidden_neurons[-1]], dtype=tf.float32,
                            initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        Q = tf.get_variable('Q', shape=[self.hidden_neurons[-1], self.hidden_neurons[-1]], dtype=tf.float32,
                            initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        V = tf.get_variable('V', shape=[self.hidden_neurons[-1], self.hidden_neurons[-1]], dtype=tf.float32,
                            initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        EK = tf.matmul(input_embedding, K) + b
        EQ = tf.matmul(input_embedding, Q) + b
        EV = tf.matmul(input_embedding, V) + b

        # Calculate the attention weight matrix A and apply it to the feature representation to get the final historical neighbor node feature representation O
        A = tf.matmul(EQ, EK, transpose_b=True) / tf.sqrt(float(self.hidden_neurons[-1]))
        O = tf.matmul(A, EV)

        # Select the corresponding features from O based on the index of historical neighbor nodes, obtaining the historical neighbor features hist_neighbors_features
        temp_hist_index = tf.reshape(self.hist_neighbor_index,
                                     [-1, self.max_step * self.hist_neighbor_num])  # [self.batch_size, max_step*M]
        # (batch_size, max_step, hist_neighbor_num, input_embedding.shape[-1])
        hist_neighbors_features = tf.reshape(tf.batch_gather(O, temp_hist_index),
                                             [-1, self.max_step, self.hist_neighbor_num, input_embedding.shape[-1]])
        return hist_neighbors_features

    def hist_neighbor_sampler1(self, input_q_emb, next_q_emb, qa_emb):#sample based on question similarity
        #next_q_emb:[batch_size,ms,emb_size]
        mold_nextq = tf.sqrt(tf.reduce_sum(next_q_emb*next_q_emb,-1))#[bs,ms]
        next_q_emb = tf.expand_dims(next_q_emb,2)
        mold_inputq = tf.sqrt(tf.reduce_sum(input_q_emb*input_q_emb,-1))#[bs,ms]
        input_q_emb = tf.expand_dims(input_q_emb,1)
        q_similarity = tf.reduce_sum(next_q_emb*input_q_emb,-1)#[batch_size,ms,ms]
        molds = tf.expand_dims(mold_nextq,2)*tf.expand_dims(mold_inputq,1)#[bs,ms,ms]
        q_similarity = q_similarity/molds
        zero_embeddings = tf.expand_dims(tf.zeros([self.batch_size,self.hidden_neurons[-1]],dtype=tf.float32),1)#[batch_size,1,hidden_size]
        qa_emb = tf.concat([qa_emb, zero_embeddings], 1)#[batch_size,max_step+1,hidden_size]
        #mask future position
        seq_mask = tf.range(1,self.max_step+1)
        similarity_seqs = tf.tile(tf.expand_dims(tf.cast(tf.sequence_mask(seq_mask, self.max_step),
                                    dtype=tf.float32),0),[self.batch_size,1,1])  # [batch_size,ms,ms]
        q_similarity = q_similarity*similarity_seqs #only history q non zero# [batch_size,ms,ms]
        #setting lower similarity bount
        condition = tf.greater(q_similarity,self.args.att_bound)
        #condition = tf.greater(q_similarity,0.7)
        q_similarity = tf.where(condition,q_similarity,tf.zeros([self.batch_size,self.max_step,self.max_step]))
        self.q_similarity = q_similarity
        temp_hist_index = tf.nn.top_k(q_similarity, self.hist_neighbor_num)[1]# [batch_size,ms,hist_num]
        self.hist_attention_value = tf.nn.top_k(q_similarity, self.hist_neighbor_num)[0]# [batch_size,ms,hist_num]
        #q_similarity[temp_hist_index]>0
        temp_hist_index = tf.where(self.hist_attention_value>0,temp_hist_index,self.hist_neighbor_index)
        temp_hist_index = tf.reshape(temp_hist_index,[-1,self.max_step*self.hist_neighbor_num])
        hist_neighbors_features =tf.reshape(tf.batch_gather(qa_emb, temp_hist_index), [-1, self.max_step, self.hist_neighbor_num, qa_emb.shape[-1]])
        return hist_neighbors_features


    def next_neighbor_sampler(self,aggregate_embedding):
        temp_emb = tf.reshape(aggregate_embedding[1],[-1,self.question_neighbor_num,self.embedding_size])
        temp_emb = tf.transpose(temp_emb, [1, 0, 2])
        temp_emb = tf.transpose(
            tf.gather(temp_emb, tf.random.shuffle(tf.range(tf.shape(temp_emb)[0]))), [1, 0, 2])
        if self.question_neighbor_num>=self.next_neighbor_num:
            next_neighbors_embedding = tf.reshape(temp_emb[:,:self.next_neighbor_num,:],[self.batch_size,self.max_step,self.next_neighbor_num,self.embedding_size])
        else:
            tile_neighbor_embedding = tf.tile(temp_emb,[1, -(-self.next_neighbor_num // tf.shape(temp_emb)[0]), 1])
            next_neighbors_embedding = tf.reshape(tile_neighbor_embedding[:,:self.next_neighbor_num,:],[self.batch_size,self.max_step,self.next_neighbor_num,self.embedding_size])
        return next_neighbors_embedding

    def get_neighbors(self,n_hop, question_index):
        # question_index:[batch_size,seq_len]
        seeds = [question_index]  # [[batch_size,seq_len],[batch_size,seq_len,question_neighbor_num],batch_size,seq_len,question_neighbor_num,
        for i in range(n_hop):
            if i % 2 == 0:
                neighbor = tf.reshape(tf.gather(self.question_neighbors, tf.reshape(seeds[i], [-1])),
                                      [-1, self.max_step, self.question_neighbor_num])

            else:
                neighbor = tf.reshape(tf.gather(self.question_neighbors, tf.reshape(seeds[i], [-1])),
                                      [-1, self.max_step, self.skill_neighbor_num])
            seeds.append(neighbor)  # [batch_size,seq_len,neighbor_num],[batch_size,seq_len,neighbor_num*neighbor_num]
        return seeds

    def aggregate(self, input_neighbors, input_questions_embedding):
        # [[batch_size,seq_len],[batch_size,seq_len,q_neighbor_num],[batch_size,seq_len,q_neighbor_num*s_neighbor_num]]
        sq_neighbor_vectors = []
        for hop_i, neighbors in enumerate(input_neighbors):
            if hop_i % 2 == 0:  # question
                temp_neighbors = tf.reshape(tf.gather(self.feature_embedding, tf.reshape(neighbors, [-1])),
                                            [self.batch_size, self.max_step, -1, self.embedding_size])
                sq_neighbor_vectors.append(temp_neighbors)
            else:  # skill
                temp_neighbors = tf.reshape(tf.gather(self.feature_embedding, tf.reshape(neighbors, [-1])),
                                            [self.batch_size, self.max_step, -1, self.embedding_size])
                sq_neighbor_vectors.append(temp_neighbors)
        aggregators = []
        for i in range(self.n_hop):
            if i == self.n_hop - 1:
                aggregator = self.aggregator_class(self.batch_size, self.max_step, self.embedding_size, act=tf.nn.tanh,
                                                   dropout=self.keep_prob_gnn)
            else:
                aggregator = self.aggregator_class(self.batch_size, self.max_step, self.embedding_size, act=tf.nn.tanh,
                                                   dropout=self.keep_prob_gnn)
            aggregators.append(aggregator)
            for hop in range(self.n_hop - i):  # aggregate from outside to inside#layer
                if hop % 2 == 0:
                    shape = [self.batch_size, self.max_step, -1, self.question_neighbor_num, self.embedding_size]
                    vector = aggregator(self_vectors=sq_neighbor_vectors[hop],
                                        neighbor_vectors=tf.reshape(sq_neighbor_vectors[hop + 1], shape),
                                        question_embeddings=sq_neighbor_vectors[hop],
                                        )  # [batch_size,seq_len, -1, dim]
                else:
                    shape = [self.batch_size, self.max_step, -1, self.skill_neighbor_num, self.embedding_size]
                    vector = aggregator(self_vectors=sq_neighbor_vectors[hop],
                                        neighbor_vectors=tf.reshape(sq_neighbor_vectors[hop + 1], shape),
                                        question_embeddings=sq_neighbor_vectors[hop],
                                        )  # [batch_size,seq_len, -1, dim]
                sq_neighbor_vectors[hop] = vector
        res = sq_neighbor_vectors  # [[batch_size,max_step,-1,embedding_size]...]
        return res, aggregators

    # step on batch
    def train(self, sess, features_answer_index, target_answers, seq_lens, hist_neighbor_index, features_emotional_index):
        input_feed = {self.features_answer_index: features_answer_index, self.target_answers: target_answers, self.sequence_lens: seq_lens,
                      self.hist_neighbor_index: hist_neighbor_index, self.features_emotional_index: features_emotional_index, self.is_training: True}
        input_feed[self.keep_prob] = self.dropout_keep_probs[0]
        input_feed[self.keep_prob_gnn] = self.dropout_keep_probs[1]
        bin_pred, pred, train_loss, _, aaaa = sess.run(
            [self.binary_pred, self.pred, self.loss, self.train_op, self.flat_target_correctness], input_feed)
        return bin_pred, pred, train_loss



    def evaluate(self, sess, features_answer_index, target_answers, seq_lens, hist_neighbor_index, features_emotional_index, evaluate_step):
        input_feed = {self.features_answer_index: features_answer_index, self.target_answers: target_answers, self.sequence_lens: seq_lens,
                      self.hist_neighbor_index: hist_neighbor_index, self.features_emotional_index: features_emotional_index, self.is_training: False}
        input_feed[self.keep_prob] = self.dropout_keep_probs[-1]
        input_feed[self.keep_prob_gnn] = self.dropout_keep_probs[-1]
        bin_pred, pred, output, alpha, bt = sess.run([self.binary_pred, self.pred, self.output_series, self.alpha, self.bt], input_feed)
        print(self.output_series)
        return bin_pred, pred

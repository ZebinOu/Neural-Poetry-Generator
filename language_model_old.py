import tensorflow as tf
import utils
import wxpy
import math
import numpy as np
import time


# Monitoring the learning process via phone
# bot = wxpy.Bot(console_qr=True)
# logger = wxpy.get_wechat_logger(receiver=bot)
# logger.warning('Successfully login')

class RNNLM:

    def __init__(self, params):
        cells = {
        'LSTM':lambda:tf.contrib.rnn.BasicLSTMCell(embedding_size, forget_bias=0.0, state_is_tuple=True),
        }
        self.params = params

        num_layers = params['num_layers']
        rnn_cell = cells[params['rnn_cell']]
        embedding_size = params['embedding_size']
        vocabulary_size = params['vocabulary_size']

        
        self.graph = tf.Graph()
        with self.graph.as_default():
            # create model saver
            # self.saver = tf.train.Saver()

            #-----Model building-------
            with tf.variable_scope('Input'):
                # inputs & targets : (max_sequence_length, batch_size)
                self.inputs = tf.placeholder(tf.int32, shape=(None, None), name='input_sequence')
                self.targets = tf.placeholder(tf.int32, shape=(None, None), name='target_sequence')
                self.sequence_len = tf.placeholder(tf.int32, [None])
                self.keep_prob = tf.placeholder(tf.float32, shape=(), name='keep_prob')

            with tf.variable_scope('Embedding', reuse=True):
                self.embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name='embeddings')
                



            unit_cell = lambda:tf.contrib.rnn.DropoutWrapper(rnn_cell(), output_keep_prob=self.keep_prob)


            # Build multilayer cell
            with tf.variable_scope('RNN_Cell', reuse=True):
                cell = tf.contrib.rnn.MultiRNNCell([unit_cell() for _ in range(num_layers)], state_is_tuple=True)
    

            # ----------------------------Loss-----------------------------
            with tf.variable_scope('Loss'):
                # embed - (max_sequence_len, batch_size, embedding_size)
                embed = tf.nn.embedding_lookup(self.embeddings, self.inputs)
                
                # Dropout when training
                embed = tf.nn.dropout(embed, self.keep_prob)

                initial_state = cell.zero_state(tf.shape(self.inputs)[1], tf.float32)
                
                outputs, next_state = tf.nn.dynamic_rnn(cell, embed, time_major=True, initial_state=initial_state, sequence_length=self.sequence_len)
                # # Output: [batch_size, max_time, cell.output_size] for time_major=false
                # #         [max_time, batch_size, cell.output_size] for time_major=true
                # # For shorter sequences with zero pad, output after their sequence length will be set to zeros

                flattened_outputs = tf.reshape(outputs, [-1,embedding_size])

                w = tf.Variable(tf.truncated_normal((embedding_size, vocabulary_size),stddev=1.0 / math.sqrt(embedding_size)), name='softmax_w')
                b = tf.Variable(tf.zeros([vocabulary_size]), name='softmax_b')


                logits = tf.matmul(flattened_outputs, w) + b
                # Cross entropy between flattened_outputs and targets
                
                self.loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(self.targets, [-1]), logits=logits), name='loss')
                
                tf.summary.scalar('loss', self.loss)


            # ----------------------------Evaluation----------------------------------
            # The following computation assumes the feeding data to be one dimensional
            # sample_init: (sequence_len, batch_size)
            with tf.variable_scope('Evaluation'):
                # Compromise to tf.multinomial(), which returns label of int64
                self.sample_init = tf.placeholder(tf.int32, shape=(None,), name='sample_initialization')
                sample_len = tf.shape(self.sample_init)[0]

                # Compromise to tf.nn.raw_rnn(cell, loop_fn), which use int32 instead :)
                self.max_sample_len = tf.placeholder(tf.int32, shape=())

                # Unstack a tensor into TensorArray, along the first dimension
                inputs_ta = tf.TensorArray(dtype=tf.int32, size=sample_len)
                inputs_ta = inputs_ta.unstack(self.sample_init)
                # Each inputs_ta.read(0) is a scalar tensor

                # TensorArray used to store generated token
                sample_ta = tf.TensorArray(dtype=tf.int32, size=sample_len, dynamic_size=True)


                def loop_fn(time, cell_output, cell_state, loop_state):
                    '''## Missing docstring from tensorflow api
                    Helper function for tf.nn.raw_rnn(). tf.nn.raw_rnn() takes this function
                    and apply the cell function for each batch iteratively until finished flags
                    have been set. After finished flags are set, emit_output and next_cell_state will be all zeros.

                    time=zero_tensor_scalar, cell_output=None, cell_state=None, 
                    loop_state=None when initialized

                    ## function description
                    If current time step exceeds the input sequence, sample next word from the 
                    sigmoid activation, then feed the corresponding word embedding into the 
                    cell.
                    Otherwise read next word from the input sequence.
                    loop_state is a TensorArray that keeps the currently fed word.
                    Args:
                        time         - time step along the sequence for current loop
                        cell_output  - cell output from previous time step
                        cell_state   - cell state generated from previous time step
                        loop_state   - state that maintained across loop, here is sample_ta
                    Returns:
                        elements_finished - finished flag for each sequence in the batch
                        next_input        - next input to be fed into the cell
                        next_cell_state   - next hidden state of the cell
                        emit_output       - output of cell for current time
                        next_loop_state   - loop state for current time
                    '''

                    emit_output = cell_output  # == None for time == 0

                    if cell_output is None:  # time == 0
                        # initialization
                        next_loop_state = sample_ta
                        next_cell_state = cell.zero_state(1, tf.float32)
                        next_word = inputs_ta.read(time)

                    else:                  # time > 0
                        next_cell_state = cell_state
                        next_loop_state = loop_state
                        next_word = tf.cond(
                        tf.greater_equal(time, sample_len),
                            lambda:
                            tf.maximum(tf.constant(1,dtype=tf.int32),
                            tf.cast(
                                tf.multinomial(
                                tf.log(tf.nn.softmax(tf.matmul(cell_output, w) + b)), 
                                1)[0][0] # resulting multinomial sample is ((index)), taking the index constant
                            , dtype=tf.int32)
                            ),
                            lambda:inputs_ta.read(time)
                            )

                    elements_finished = (time >= self.max_sample_len)
                    next_loop_state = next_loop_state.write(time, next_word)
                    next_input = tf.reshape(tf.nn.embedding_lookup(self.embeddings, next_word), (1, embedding_size))
                    
                    return (elements_finished, next_input, next_cell_state,
                          emit_output, next_loop_state)


                _, _, final_sample_ta = tf.nn.raw_rnn(cell, loop_fn)
                self.final_sample = final_sample_ta.stack(name='sampled_sequence')

    def train(self, max_grad, learning_rate, batch_size, num_steps, sample=None):
        
        with self.graph.as_default():
            with tf.variable_scope('Optimization'):
                # -----------------Optimization setting-----------
                global_step = tf.Variable(0, trainable=False)
                gradients, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tf.trainable_variables()), max_grad, name='Gradients')
                learn_rate = tf.train.exponential_decay(learning_rate, global_step, 10000, 0.5, staircase=True)
                optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate)
                train_step = optimizer.apply_gradients(zip(gradients, tf.trainable_variables()), global_step=global_step)
                tf.summary.scalar('learning_rate', learning_rate)
    
                summary = tf.summary.merge_all()

                init = tf.global_variables_initializer()

                batch_feeder = utils.rnnlm_batch_feeder_setup(data, batch_size)
                

                # -------------------------Training--------------------------------------

                if sample != None:
                    s_feed = np.array(utils.sentance2tokens(sample, dictionary))
                    s_len = np.array(20)


            with tf.Session(graph=self.graph) as session:
                # We must initialize all variables before we use them.
                summary_writer = tf.summary.FileWriter('./tmp/rnnlog', session.graph)
                init.run()
                start_time = time.time()
                for step in range(num_steps):
                    input_batch, target_batch, sequence_len_batch = batch_feeder()
                    feed_dict = {
                    self.inputs: input_batch, 
                    self.targets: target_batch, 
                    self.sequence_len: sequence_len_batch, 
                    self.keep_prob: self.params['keep_prob']
                    }
                    
                    _, loss_val = session.run([train_step, self.loss], feed_dict=feed_dict)

                    if step % 10 == 0:
                        duration = time.time() - start_time
                        print("Step %d,\tLoss: %g,\tTime elapse: %.3f sec" % (step, loss_val, duration))
                        summary_str = session.run(summary, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, step)
                        summary_writer.flush()

                        if step % 20000 == 0 and step != 0:
                            # self.saver.save(session, './tmp/rnndata',global_step=step )
                            pass

                        if step % 100 == 0:
                            # logger.warning('Step: %d, loss: %.3f, time: %.3f sec\n' % (step, loss_val, duration))
                            feed_dict = {
                            self.sample_init    : s_feed, 
                            self.max_sample_len : s_len, 
                            self.keep_prob      : 1
                            }

                            sam = session.run(self.final_sample, feed_dict=feed_dict)
                            sam = utils.tokens2sentance(sam, reverse_dictionary)
                            print('Sampled sequence: %s' % (sam))
                            logger.warning('Step: %d, loss: %.3f, time: %.3f sec\n%s' % (step, loss_val, duration, sam))

                        start_time = time.time()

    def pretrain(self, num_sampled, batch_size, num_steps, valid_examples, num_skips, skip_window, valid_size):
        embedding_size = params['embedding_size']
        vocabulary_size = params['vocabulary_size']

        with self.graph.as_default():
        # ----------------------------Placeholders---------------------------------
            with tf.variable_scope('word2vec_input'):
                train_inputs = tf.placeholder(tf.int32, shape=[None], name='input_word')
                train_labels = tf.placeholder(tf.int32, shape=[None, 1], name='target_word')
                valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
            

            # ----------------------------nce and loss-----------------------------------
            with tf.variable_scope('nce'):
                # Construct the variables for the NCE loss
                word_embed = tf.nn.embedding_lookup(self.embeddings, train_inputs)
                
                nce_weights = tf.Variable(
                    tf.truncated_normal([vocabulary_size, embedding_size],
                                      stddev=1.0 / math.sqrt(embedding_size)), name='weight')
                nce_biases = tf.Variable(tf.zeros([vocabulary_size]), name='bias')

                # Compute the average NCE loss for the batch.
                # tf.nce_loss automatically draws a new sample of the negative labels each
                # time we evaluate the loss.
                loss = tf.reduce_mean(
                  tf.nn.nce_loss(weights=nce_weights,
                                 biases=nce_biases,
                                 labels=train_labels,
                                 inputs=word_embed,
                                 num_sampled=num_sampled,
                                 num_classes=vocabulary_size), name='loss')
                
                tf.summary.scalar('loss', loss)

            # ----------------------------Optimization setting-----------------------------
            with tf.variable_scope('optimizer'):
                global_step = tf.Variable(0, trainable=False)
                learning_rate = tf.train.exponential_decay(0.01, global_step,
                                                       10000, 0.5, staircase=True)
                

                # Construct the SGD optimizer using a learning rate of 1.0.
                # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
                
                tf.summary.scalar('learning_rate', learning_rate)
            # ----------------------------Evaluation-----------------------------
            with tf.variable_scope('eval'):
                # Compute the cosine similarity between minibatch examples and all embeddings.
                norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keep_dims=True))
                normalized_embeddings = self.embeddings / norm
                valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
                similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

                tf.summary.histogram('similarity', similarity)


            # --------------------Initializer-----------------------------
            summary = tf.summary.merge_all()
            init = tf.global_variables_initializer()

            # -------------------------Training-----------------------------

            # Some initialization work
            batch_feeder = utils.word2vec_batch_feeder_setup(data, batch_size, num_skips, skip_window)
            # saver = tf.train.Saver([embeddings])

            with tf.Session(graph=self.graph) as session:
              # We must initialize all variables before we use them.
              summary_writer = tf.summary.FileWriter('./tmp/log', session.graph)
              init.run()

              start_time = time.time()
              for step in range(num_steps):
                batch_center, batch_target = batch_feeder()
                feed_dict = {train_inputs: batch_center, train_labels: batch_target}


                _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)


                if step % 100 == 0:
                    duration = time.time() - start_time
                    print("Step %d,\tLoss: %g,\tTime elapse: %.3f sec" % (step, loss_val, duration))

                    summary_str = session.run(summary, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

                    if step % 20000 == 0 and step != 0:
                        # saver.save(session, './tmp/data',global_step=step )
                        pass

                    start_time = time.time()
                    if step % 1000 == 0:
                        sim = similarity.eval()
                        wechat_str = ''
                        for i in range(valid_size):
                            valid_word = reverse_dictionary[valid_examples[i]]
                            top_k = 8  # number of nearest neighbors
                            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                            log_str = "%s: " % valid_word
                            for k in range(top_k):
                                close_word = reverse_dictionary[nearest[k]]
                                log_str += close_word
                            print(log_str)
                            wechat_str += ('\n'+log_str)
                        # logger.warning('Step: %d, loss: %.3f, time: %.3f sec\n%s' % (step, loss_val, duration, wechat_str))

#--------------------------rnnlm Model setup------------------------------

vocabulary_size = utils.vocabulary_size # unique symbols in poems: 11873
embedding_size = utils.embedding_size

params = {
'embedding_size': embedding_size,
'vocabulary_size': vocabulary_size,
'num_layers': 4,  # num of RNN layers
'keep_prob': 0.5, # dropout
'rnn_cell':'LSTM' 
}


#--------------------word2vec------------------------------------------

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)


# ---------------------------Data feeding preparation---------------
# Read and tokenize data
data_dir = './data/'
texts = ['qts_tab.txt', 'qsc_tab.txt', 'qtais_tab.txt','qss_tab.txt']
# max and min length of poem sequence
maxlen = 200
minlen = 7

poems = []
for t in texts:
    poems.extend(utils.read_poem(data_dir + t))
data, count, dictionary, reverse_dictionary = utils.tokenize(poems, vocabulary_size, 'all_poems')


rnnlm = RNNLM(params)


rnnlm.pretrain(num_sampled=128, batch_size=256, num_steps=100000, valid_examples=valid_examples, num_skips=4, skip_window=2, valid_size=16)


poems = utils.chop_poems(poems, maxlen, minlen)
data, count, dictionary, reverse_dictionary = utils.tokenize(poems, vocabulary_size, 'all_poems')
s = '苟利国家生死以，'  
rnnlm.train(max_grad=1, learning_rate=0.001, batch_size=128, num_steps=50000, sample=s)



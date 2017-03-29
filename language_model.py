import tensorflow as tf
import utils, math, time, os
import numpy as np
# import wxpy
# Monitoring the learning process via phone
# bot = wxpy.Bot(console_qr=True)
# logger = wxpy.get_wechat_logger(receiver=bot)
# logger.warning('Successfully login')

class RNNLM:
    def __init__(self, params, data, count, dictionary, reverse_dictionary):
        self.data = data
        self.count = count
        self.dictionary = dictionary
        self.reverse_dictionary = reverse_dictionary
        self.params = params

        cells = {
        'LSTM':lambda:tf.contrib.rnn.BasicLSTMCell(params['embedding_size'], state_is_tuple=True),
        'GRU':lambda:tf.contrib.rnn.GRUCell(params['embedding_size'])
        }
   
        self.model_description = '#########Model setup###########\ncell type:\t %s\nnum_layers:\t %d\ndropout_prob: \t %f\nembed_size:\t %d\nvocab_size:\t %d\n###############################\n'%(params['rnn_cell'], params['num_layers'], params['keep_prob'], params['embedding_size'], params['vocabulary_size'])

        self.training_description = '########Training options#########\nmax_grad:\t%f\nbatch_size:\t%d\nmax_steps:\t%d\nlearning_rate:\t%g\ndecay_steps:\t%d\ndecay_rate:\t%g\noptimizer:\t%s\n#################################\n'%(params['max_grad'], params['batch_size'], params['num_steps'], params['learning_rate'], params['decay_steps'], params['decay_rate'], params['optimizer_name'])


        num_layers = params['num_layers']
        rnn_cell = cells[params['rnn_cell']]
        embedding_size = params['embedding_size']
        vocabulary_size = params['vocabulary_size']

        self.graph = tf.Graph()
        with self.graph.as_default():
            
            #-----Model building-------
            with tf.variable_scope('Input'):
                # inputs & targets : (max_sequence_length, batch_size)
                self.inputs = tf.placeholder(tf.int32, shape=(None, None), name='input_sequence')
                self.targets = tf.placeholder(tf.int32, shape=(None, None), name='target_sequence')
                self.sequence_len = tf.placeholder(tf.int32, [None])
                self.keep_prob = tf.placeholder(tf.float32, shape=(), name='keep_prob')

            with tf.variable_scope('Embedding', reuse=True):
                # self.embeddings = tf.Variable(
                # tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name='embeddings')
                # self.embeddings = tf.Variable(np.load('./tmp/w2vdata/step_30000.npy'), dtype=tf.float32)
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

                self.w = tf.Variable(tf.truncated_normal((embedding_size, vocabulary_size), stddev=1.0/math.sqrt(embedding_size)), name='softmax_w')
                self.b = tf.Variable(tf.zeros([vocabulary_size]), name='softmax_b')
                logits = tf.matmul(flattened_outputs, self.w) + self.b
                
                # Cross entropy between flattened_outputs and targets
                self.loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(self.targets, [-1]), logits=logits), name='loss')
                self.perplexity = tf.exp(self.loss)
                tf.summary.scalar('perplexity', self.perplexity)

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
                        def get_next_word():
                            '''select words with top k logits, eliminate <UNK> and perform softmax sampling'''
                            values, indices = tf.nn.top_k(tf.matmul(cell_output, self.w) + self.b, k=80)
                            nonzero = tf.where(tf.not_equal(indices, tf.zeros_like(indices)))
                            indices = tf.reshape(tf.gather_nd(indices, nonzero),(1, -1))
                            values = tf.reshape(tf.gather_nd(values, nonzero),(1, -1))
                            next_word = indices[0][tf.cast(tf.multinomial(values, 1), dtype=tf.int32)[0][0]]
                            return next_word

                        next_word = tf.cond(
                        tf.greater_equal(time, sample_len),
                            lambda:
                            get_next_word(),
                            lambda:inputs_ta.read(time)
                            )

                    elements_finished = (time >=self.max_sample_len) 
                    next_loop_state = next_loop_state.write(time, next_word)
                    next_input = tf.reshape(tf.nn.embedding_lookup(self.embeddings, next_word), (1, embedding_size))
                    return (elements_finished, next_input, next_cell_state,
                          emit_output, next_loop_state)

                _, _, final_sample_ta = tf.nn.raw_rnn(cell, loop_fn)
                self.final_sample = final_sample_ta.stack(name='sampled_sequence')

            self.saver = tf.train.Saver(tf.trainable_variables())

######################Training############################################3
    def train(self, sample_interval=1000, save_interval=5000, logger=None):
        max_grad = self.params['max_grad']
        learning_rate = self.params['learning_rate']
        batch_size = self.params['batch_size']
        num_steps = self.params['num_steps']
        sample = self.params['sample']
        max_sample_length = self.params['max_sample_length']
        decay_steps = self.params['decay_steps']
        decay_rate = self.params['decay_rate']
        optimizer_name = self.params['optimizer_name']


        with self.graph.as_default():
            with tf.variable_scope('Optimization'):
                # -----------------Optimization setting-----------
                global_step = tf.Variable(0, trainable=False)
                gradients, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tf.trainable_variables()), max_grad, name='Gradients')
                learn_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps=decay_steps, decay_rate=decay_rate, staircase=True)

                if optimizer_name == 'GradientDescent':
                    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate)
                elif optimizer_name == 'Adam':
                    optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate)
                else:
                    raise Exception('"optimizer_name" does not macth either "Adam" or "GradientDescent".')

                train_step = optimizer.apply_gradients(zip(gradients, tf.trainable_variables()), global_step=global_step) 

                tf.summary.scalar('learning_rate', learn_rate)
                summary = tf.summary.merge_all()

                init = tf.global_variables_initializer()
                batch_feeder = utils.rnnlm_batch_feeder_setup(self.data, batch_size)
                # -------------------------Training--------------------------------------
                if sample != None:
                    s_feed = np.array(utils.sentance2tokens(sample, self.dictionary))
                    s_len = np.array(max_sample_length)

            with tf.Session(graph=self.graph) as session:
                # Clear the log folder to avoid mess up tensorboard graph
                folder = './tmp/rnnlog'
                for the_file in os.listdir(folder):
                    file_path = os.path.join(folder, the_file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except Exception as e:
                        print(e)
                summary_writer = tf.summary.FileWriter('./tmp/rnnlog', session.graph)


                init.run()
                start_time = time.time()
                for step in range(num_steps):
                    input_batch, target_batch, sequence_len_batch, epochs= batch_feeder()
                    feed_dict = {
                    self.inputs: input_batch, 
                    self.targets: target_batch, 
                    self.sequence_len: sequence_len_batch, 
                    self.keep_prob: self.params['keep_prob']
                    }

                    _, lr, perplexity = session.run([train_step, learn_rate, self.perplexity], feed_dict=feed_dict)
                    if step % 5 == 0:
                        duration = time.time() - start_time
                        print('Epoch: %d, Step %d, lr: %g, perplexity: %g, Time: %.3f sec' % (epochs, step, lr, perplexity, duration))
                        summary_str = session.run(summary, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, step)
                        summary_writer.flush()

                        if step % save_interval == 0 and step != 0:
                            self.saver.save(session, './tmp/rnndata/model', global_step=step)

                        if step % sample_interval == 0:
                            feed_dict = {
                            self.sample_init    : s_feed, 
                            self.max_sample_len : s_len, 
                            self.keep_prob      : 1
                            }

                            sam = session.run(self.final_sample, feed_dict=feed_dict)
                            sam = utils.tokens2sentance(sam, self.reverse_dictionary)
                            print('Epoch: %d, Perplexity: %g, \nSampled sequence: %s\n' % (epochs, perplexity, sam))
                            
                            if logger:
                                # enable wechat logger
                                logger.warning('Epoch: %d, lr: %g, Perplexity: %.g, \n%s' % (epochs, lr, perplexity, sam))

                        start_time = time.time()

###########################sampler############################################3
    def sample(self, sample_len, checkpoint_dir='./tmp/rnndata/'):
        key = self.dictionary.keys()
        print(self.training_description)
        print(self.model_description)
        with tf.Session(graph=self.graph) as session:
            path = tf.train.latest_checkpoint(checkpoint_dir)
            self.saver.restore(session, path)
            while True:
                s = input('Type Chinese strings to warm up the model:\n')
                valid = True
                for i in s:
                    valid = valid and (i in key)
                if not valid:
                    continue
                s_feed = np.array(utils.sentance2tokens(s, self.dictionary))
                s_len = np.array(sample_len)
                feed_dict = {
                self.sample_init    : s_feed, 
                self.max_sample_len : s_len, 
                self.keep_prob      : 1
                }
                sam = session.run(self.final_sample, feed_dict=feed_dict)
                sam = utils.tokens2sentance(sam, self.reverse_dictionary)
                print(sam)    










import tensorflow as tf
import utils, math, time, os
import numpy as np
# import pickle
import wxpy
# Monitoring the learning process via phone
bot = wxpy.Bot(console_qr=True)
logger = wxpy.get_wechat_logger(receiver=bot)
logger.warning('Successfully login')


class RNNLM:

    def __init__(self, params):
        cells = {
        'LSTM':lambda:tf.contrib.rnn.BasicLSTMCell(embedding_size, state_is_tuple=True),
        'GRU':lambda:tf.contrib.rnn.GRUCell(embedding_size)
        }
        self.params = params

        num_layers = params['num_layers']
        rnn_cell = cells[params['rnn_cell']]
        embedding_size = params['embedding_size']
        vocabulary_size = params['vocabulary_size']
        
        self.model = '''#########Model setup###########              
cell type:\t %s
num_layers:\t %d
dropout_prob: \t %f
embed_size:\t %d
vocab_size:\t %d 
###############################
\n''' %(params['rnn_cell'], params['num_layers'], params['keep_prob'], embedding_size, vocabulary_size)

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
                # tf.summary.histogram('w', self.w)
                # tf.summary.histogram('b', self.b)
                logits = tf.matmul(flattened_outputs, self.w) + self.b

                # tf.summary.histogram('softmax_value', tf.nn.softmax(logits[5]))
                
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
                        def get_next_word():
                            '''select words with top 6 logits, eliminate <UNK> and perform softmax sampling'''
                            values, indices = tf.nn.top_k(tf.matmul(cell_output, self.w) + self.b, k=4)
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


    def train(self, max_grad, learning_rate, batch_size, num_steps, sample=None, max_sample_length=11, decay_steps=5000, decay_rate=0.5, opt='Adam'):

        with self.graph.as_default():
            with tf.variable_scope('Optimization'):
                # -----------------Optimization setting-----------
                global_step = tf.Variable(0, trainable=False)
                gradients, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tf.trainable_variables()), max_grad, name='Gradients')
                learn_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps=decay_steps, decay_rate=decay_rate, staircase=True)

                # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate)
                optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate)
                train_step = optimizer.apply_gradients(zip(gradients, tf.trainable_variables()), global_step=global_step) 
                # gradw = tf.gradients(self.loss, [self.w])
                # gradb = tf.gradients(self.loss, [self.b])
                # tf.summary.histogram('gradw', gradw)
                # tf.summary.histogram('gradb', gradb)
                # tf.summary.scalar('w_gradw', tf.norm(gradw)/tf.norm(self.w))
                # tf.summary.scalar('w_gradb', tf.norm(gradb)/tf.norm(self.b))
                    # tf.summary.scalar('ratio%d'%(i), tf.norm(a)/tf.norm(b))

                tf.summary.scalar('learning_rate', learn_rate)
                summary = tf.summary.merge_all()


                init = tf.global_variables_initializer()
                batch_feeder = utils.rnnlm_batch_feeder_setup(data, batch_size)
                # -------------------------Training--------------------------------------
                if sample != None:
                    s_feed = np.array(utils.sentance2tokens(sample, dictionary))
                    s_len = np.array(10)

            with tf.Session(graph=self.graph) as session:
                # We must initialize all variables before we use them.
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

                    _, lr, loss_val = session.run([train_step, learn_rate, self.loss], feed_dict=feed_dict)
                    if step % 1 == 0:
                        duration = time.time() - start_time
                        print('Epoch: %d, Step %d, lr: %g, perplexity: %g, Time: %.3f sec' % (epochs, step, lr, np.exp(loss_val), duration))
                        summary_str = session.run(summary, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, step)
                        summary_writer.flush()

                        if step % 5000 == 0 and step != 0:
                            self.saver.save(session, './tmp/rnndata/model', global_step=step)
                            pass
                        if step % 5000 == 0:
                            feed_dict = {
                            self.sample_init    : s_feed, 
                            self.max_sample_len : s_len, 
                            self.keep_prob      : 1
                            }

                            sam = session.run(self.final_sample, feed_dict=feed_dict)
                            sam = utils.tokens2sentance(sam, reverse_dictionary)
                            print('Epoch: %d, Loss: %g, \nSampled sequence: %s\n' % (epochs, loss_val, sam))
                            logger.warning('Epoch: %d, lr: %g, loss: %.3f, \n%s' % (epochs, lr, loss_val, sam))
                            pass
                        if step % 1500 == 0:
                            print('''########Training options#########
max_grad:\t%f
batch_size:\t%d
max_steps:\t%d
learning_rate:\t%g
decay_steps:\t%d
decay_rate:\t%g
optimizer:\t%s
#################################
            '''%(max_grad, batch_size, num_steps, learning_rate, decay_steps, decay_rate, opt))
                            print(self.model)
                        start_time = time.time()


#--------------------------rnnlm Model setup------------------------------

vocabulary_size = 8000 # unique symbols in poems: 11873
embedding_size = 1200

params = {
'embedding_size': embedding_size,
'vocabulary_size': vocabulary_size,
'num_layers': 4,  # num of RNN layers
'keep_prob': 0.8, # dropout
'rnn_cell':'LSTM', 
}

# ---------------------------Data feeding preparation---------------
# Read and tokenize data
data_dir = './data/'
texts = ['qts_tab.txt', 'qsc_tab.txt', 'qtais_tab.txt','qss_tab.txt']
# max and min length of poem sequence
maxlen = 150
minlen = 7
poems = []
for t in texts:
    poems.extend(utils.read_poem(data_dir + t))


poems = utils.chop_poems(poems, maxlen, minlen)
print('Number of poems: %d'%(len(poems)))
data, count, dictionary, reverse_dictionary = utils.tokenize(poems, vocabulary_size, 'all_poems')
# with open('./data/temp100','wb') as file:
#     pickle.dump([data, count, dictionary, reverse_dictionary], file)
# with open('./data/temp','rb') as file:
#     data, count, dictionary, reverse_dictionary = pickle.load(file)
rnnlm = RNNLM(params)

s = '苟'  
rnnlm.train(max_grad=8.0, learning_rate=0.0001, batch_size=64, num_steps=1000000, sample=s, max_sample_length=30, decay_steps=10000, decay_rate=0.1)


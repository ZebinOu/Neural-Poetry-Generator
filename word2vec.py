import tensorflow as tf
import numpy as np
import math
import time
import utils
# import wxpy 

# This is used for logging the training process
# bot = wxpy.Bot(console_qr=True)
# logger = wxpy.get_wechat_logger(receiver=bot)
# logger.warning('Successfully login!')

# ---------------------------Data feeding preparation--------------

# Read and tokenize data
data_dir = './data/'
texts = ['qts_tab.txt', 'qsc_tab.txt', 'qtais_tab.txt','qss_tab.txt']
poems = []
for t in texts:
    poems.extend(utils.read_poem(data_dir + t))

vocabulary_size = utils.vocabulary_size # unique symbols in poems: 11873
data, count, dictionary, reverse_dictionary = utils.tokenize(poems, vocabulary_size, 'all_poems')




#--------------------------Model setup------------------------------
embedding_size = utils.embedding_size
batch_size = 256
num_skips = 4
skip_window = 2     # should set according to min_poem_length in utils module
num_sampled = 128   # sample data to evaluate NCE
num_steps = 100000  # maximum training step



# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)




# --------------------------------Model building----------------------------
graph = tf.Graph()
with graph.as_default():
    # ----------------------------Placeholders---------------------------------
    with tf.variable_scope('input'):
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size], name='input_word')
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1], name='target_word')
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    

    # ----------------------------Embedding-------------------------------------
    with tf.variable_scope('embedding'):
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name='embeddings')
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)



    # ----------------------------nce and loss------------------------------------------
    with tf.variable_scope('nce'):
        # Construct the variables for the NCE loss
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
                         inputs=embed,
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
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

        tf.summary.histogram('similarity', similarity)


    # ----------------------------Initializer----------------------------------------------
    summary = tf.summary.merge_all()
    init = tf.global_variables_initializer()





# -------------------------Training----------------------------------------------------

# Some initialization work
batch_feeder = utils.word2vec_batch_feeder_setup(data, batch_size, num_skips, skip_window)
saver = tf.train.Saver([embeddings])

with tf.Session(graph=graph) as session:
  # We must initialize all variables before we use them.
  summary_writer = tf.summary.FileWriter('./tmp/w2vlog', session.graph)
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

        if step % 5000 == 0 and step != 0:
            saver.save(session, './tmp/w2vdata/model',global_step=step )
            e = session.run(embeddings)
            np.save('./tmp/w2vdata/step_%d'%(step), e)
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
                # wechat_str += ('\n'+log_str)
            # logger.warning('Step: %d, loss: %.3f, time: %.3f sec\n%s' % (step, loss_val, duration, wechat_str))
        


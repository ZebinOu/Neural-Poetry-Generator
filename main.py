import utils, language_model
import wxpy
# Monitoring the learning process via phone
# bot = wxpy.Bot(console_qr=True)
# logger = wxpy.get_wechat_logger(receiver=bot)
# logger.warning('Successfully login')



#--------------------------rnnlm Model setup------------------------------
params = {
# Model setup
'vocabulary_size': 8000,
'embedding_size': 1024,
'num_layers': 4,  # num of RNN layers
'keep_prob': 1, # dropout
'rnn_cell':'LSTM', 

# Training setup
'max_grad': 5.0, 
'learning_rate': 0.0001,
'batch_size': 64,
'num_steps': 100000,
'sample': '如', 
'max_sample_length': 50, 
'decay_steps': 2000, 
'decay_rate': 0.1,
'optimizer_name': 'Adam'  # Adam or GradientDescent
}



# ---------------------------Data feeding preparation---------------
# Read and tokenize data
texts = ['./data/qts_tab.txt', './data/qsc_tab.txt', './data/qtais_tab.txt','./data/qss_tab.txt']
# max and min length of poem sequence
maxlen = 100
minlen = 7
poems = []
# for t in texts:
#     poems.extend(utils.read_poem(t))
for t in texts:
    poems.extend(utils.read_regular_poem(t))

poems = utils.chop_poems(poems, maxlen, minlen)
data, count, dictionary, reverse_dictionary = utils.tokenize(poems, params['vocabulary_size'])


rnnlm = language_model.RNNLM(params, data, count, dictionary, reverse_dictionary)
rnnlm.train(sample_interval=1000, save_interval=5000, logger=None)
# rnnlm.sample(sample_len=100, checkpoint_dir='./tmp/rnndata/'
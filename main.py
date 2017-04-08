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
    'embedding_size': 512,
    'num_layers': 3,  # num of RNN layers
    'keep_prob': 0.5, # dropout
    'rnn_cell':'LSTM',

    # Training setup
    'max_grad': 2.0,
    'learning_rate': 0.0001,
    'batch_size': 128,
    'num_steps': 100000,
    'decay_steps': 1000,
    'decay_rate': 0.1,
    'optimizer_name': 'Adam',  # Adam or GradientDescent

    # Evaluation setup
    'sample': '如',
    'max_sample_length': 50,
    'sample_range': 2 # how many words in the dictionary to be considered when sampling
}

# -------------------------Data feeding preparation---------------
# Read and tokenize data
texts = ['./data/qts_tab.txt', './data/qsc_tab.txt', './data/qtais_tab.txt', './data/qss_tab.txt']
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
rnnlm.train(sample_interval=100, save_interval=5000, logger=None)
# rnnlm.sample(sample_len=100, checkpoint_dir='./tmp/rnndata/')
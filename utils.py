'''Defines cross tasks parameters and common utility functions
'''

import tensorflow as tf
import re
import collections
import random
import numpy as np

vocabulary_size = 8000
embedding_size = 800
texts = ['qts_tab.txt', 'qsc_tab.txt', 'qtais_tab.txt','qss_tab.txt']
min_poem_length = 7 # the shortest poem to be considered



def read_poem(filename):
    with tf.gfile.GFile(filename, "r") as f:
        words = f.read().decode("utf-8").split()
        temp=[]
        for segment in words:
            # find all words that satisfies [[(###,)*n][###。]]*m pattern
            candidates = re.findall('((?:(?:[\u4E00-\u9FFF]+，)*?[\u4E00-\u9FFF]+。)+)', segment)
            temp += [c for c in candidates if len(c) > min_poem_length]
        return temp

def tokenize(poems, vocabulary_size, name):
    '''
    Map word to index.
    Arg:
        poems:              List of poem strings
        vocabulary_size:    Number of words that are not replaced by UNK 
    Return:
        data:               List of poem strings, each string convert to list of index
        count:              List of [word, count of word]s
        dictionary:         dictionary[word] == index
        reverse_dictionary: reverse_dictionary[index] == word
    '''
    
    print('Building tokenized_data...')
    words = ''.join(poems)
    num_words = len(words)
    
    count = [['UNK', -1]]
    # Get the #vocabulary_size most frequent words, and set others to UNK
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))

    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = []
    unk_count = 0
    for s in poems:
        p = []
        for word in s:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0  # dictionary['UNK']
                unk_count += 1
            p.append(index)
        data.append(p)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    print('Data size: %d words.' % (num_words))

    return data, count, dictionary, reverse_dictionary

def chop_poems(poems, maxlen, minlen):
    '''Chop the poems longer than maxlen, discard ones that are shorter than minlen, sort poems in ascending length.
    Args:
        poems  - raw read poems
        maxlen - maximum length of single poem
        minlen - minimum length of single poem
    Return:
        chopped_poems - chopped and filtered poems
    '''
    
    pattern = '((?:[\u4E00-\u9FFF]+，)*?[\u4E00-\u9FFF]+。)' 
    chopped_poems = []
    for p in poems:
        if len(p) > maxlen:
            units = re.findall(pattern, p)
            units.reverse() # make it easy for pop
            chunk = [] # temporary list of units for a chunk
            chunk_length = 0
            while len(units) > 0:
                if chunk_length + len(units[-1]) < maxlen:
                    chunk_length += len(units[-1])
                    chunk.append(units.pop())
                else:
                    chopped_poems.append(''.join(chunk))
                    chunk.clear()
                    chunk_length = 0
            else:
                temp = ''.join(chunk)
                if len(temp) > minlen:
                    chopped_poems.append(temp)
        else:
            if len(p) > minlen:
                chopped_poems.append(p)

    # shuffle and sort by length to make the data easy to feed
    random.shuffle(chopped_poems)
    chopped_poems.sort(key=(lambda s: len(s)), reverse=True)
    
    return chopped_poems

def word2vec_batch_feeder_setup(data, batch_size, num_skips, skip_window):
    '''Return a data batch feeder. 
    Each window is of the form [skip_window center skip_window], where skip_window is number of target words.
    Windows at edge will be discarded. For left edge [num_words center skip_window] will be considered where num_words < skip_window, and so do right edge.
    Args:
        data          - A list of tokenized strings
        batch_size    - Number of data samples in the batch
        num_skips     - Number of (center_word, word_in_skip_window) pairs to be sampled
        skip_window   - Number of words to be considered to the left and the right of the center word
    Return:
        batch_feeder  - Function that returns data batches
    '''
    # Initialize batch function attributes
    buffer_queue = collections.deque(maxlen=(2 * skip_window + 1))
    pp = 0 
    wp = 0
    window_size = 2 * skip_window + 1
    min_poem_length = min([len(p) for p in data])

    # window_size <= min_poem_length guarantees that the shortest poem should at least fill up the buffer_queue.
    assert window_size <= min_poem_length   
    assert num_skips <= 2 * skip_window     # maximum num_skips a window can generates
    assert batch_size % num_skips == 0      # Avoid remembering targets_to_avoid between batches
    
    # Initialize the batches:
    batch_center = np.ndarray(shape=(batch_size), dtype=np.int32)
    batch_target = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    def batch_feeder():
        '''
        Generates a new batch on each call.
        Return:
            batch_center - list of center words
            batch_target - list of target words
        '''
        # Parameters used across multiple calls
        nonlocal batch_center, batch_target, buffer_queue, pp, wp
        
        # Initialization. 
        # Append new words to fix size buffer_queue until the buffer_queue is full
        # The maxlen of the buffer_queue == window_size
        while len(buffer_queue) < window_size:
            buffer_queue.append(data[pp][wp])
            wp += 1


        for i in range(batch_size // num_skips):
            target = skip_window
            targets_to_avoid = [skip_window]
            for j in range(num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, window_size - 1)
                targets_to_avoid.append(target)
                batch_center[i * num_skips + j] = buffer_queue[skip_window]
                batch_target[i * num_skips + j, 0] = buffer_queue[target]

            if wp < len(data[pp]):
                buffer_queue.append(data[pp][wp])
                wp += 1
                
            else:
                buffer_queue.clear()
                wp = 0
                pp = (pp + 1) % len(data)
                # Append new words to fix size buffer_queue until the buffer_queue is full
                while len(buffer_queue) < window_size:
                    buffer_queue.append(data[pp][wp])
                    wp += 1
        return batch_center, batch_target

    return batch_feeder

def rnnlm_batch_feeder_setup(data_sorted, batch_size):
    '''Return a data batch feeder. 
    Shorter sequences in the same batch will be padded with 0s according to the longest sequence.
    Args:
        data          - A list of tokenized strings, sorted in reverse length order
        batch_size    - Number of data samples in the batch
    Return:
        batch_feeder  - Function that returns data batches and sequence_length
    '''
    
    # Only consider num_steps*batch_size sequences ahead to avoid having longest and shortest sequences in the same batch
    num_steps = len(data_sorted) // batch_size
    step = 0
    epochs = 0
    batches = []
    def batch_feeder():
        '''
        Generates a new batch on each call.
        Return:
            input_batch - (batch_size, maxlen) input sequences
            target_batch - (batch_size, maxlen) output sequences
            sequence_length - (batch_size,) original input sequence length
        '''
        # Parameters used across multiple calls
        nonlocal step, epochs, batches
        next_step = step + 1
        if len(batches) == step:        
            # make copies 
            batch = [list(i) for i in data_sorted[batch_size*step:batch_size*next_step]]
            sequence_length = [len(p) for p in batch]
            for i in range(len(batch)):
                while len(batch[i]) < sequence_length[0]:
                    batch[i].append(0)
            batch = np.array(batch)
            sequence_length = np.array(sequence_length)
            # shift the batch
            input_batch = batch[:,:-1].T
            target_batch = batch[:,1:].T
            sequence_length -=  1
            batches.append((input_batch, target_batch, sequence_length))
        else:
            input_batch, target_batch, sequence_length = batches[step]
        step = next_step % num_steps
        if step == 0:
            random.shuffle(batches)
            epochs += 1
        # Output batch: (max_sequence_length, batch_size)
        return input_batch, target_batch, sequence_length, epochs
    return batch_feeder

def sentance2tokens(s, dictionary):
    return [dictionary[i] for i in s]

def tokens2sentance(tokens, reverse_dictionary):
    return ''.join([reverse_dictionary[i] for i in tokens])


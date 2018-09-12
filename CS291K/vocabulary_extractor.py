#! /usr/bin/env python

from tensorflow.contrib import learn
import tensorflow as tf
import numpy as np

def tokenizer(iterator):
  """Tokenizer generator.

  Args:
    iterator: Input iterator with strings.

  Yields:
    array of tokens per each value in the input.
  """
  for value in iterator:
    yield value

def loadGloVe(filename):
    vocab = []
    embd = []
    file = open(filename,'r', encoding='utf8')
    for line in file.readlines():
        row = line.strip().split(' ')
        if len(row) != 101:
            continue
        vocab.append(row[0].split('_'))
        embd.append(row[1:])

    print('Loaded words vectors!')
    file.close()
    return vocab,embd

def generateData(embiddingsFile, embeddingsDim, x_text, y, max_document_length):
    print("Loading words vectors!")
    embedding_dim = embeddingsDim
    filename = embiddingsFile


    vocab,embd = loadGloVe(filename)
    vocab_size = len(vocab)
    embedding_dim = len(embd[0])
    #embedding = np.asarray(embd)
    embedding = np.asarray([np.asarray(xi, dtype=np.float32) for xi in embd])
    W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]),
                    trainable=False, name="W")
    embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
    embedding_init = W.assign(embedding_placeholder)

    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=session_conf)
    feedDict = {embedding_placeholder: embedding}
    sess.run(embedding_init, feed_dict=feedDict)#[list(feedDict.keys())[0]])

    #init vocab processor
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length, tokenizer_fn=tokenizer)
    #fit the vocab from glove
    pretrain = vocab_processor.fit(vocab)
    #transform inputs
    x = np.array(list(vocab_processor.transform(x_text)))

    return x, y, vocab_processor
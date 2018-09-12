#! /usr/bin/env python

import sys
import tensorflow as tf
import batchgen
import numpy as np
from tensorflow.contrib import learn
import argparse
from vocabulary_extractor import generateData

def tokenizer(iterator):
  """Tokenizer generator.

  Args:
    iterator: Input iterator with strings.

  Yields:
    array of tokens per each value in the input.
  """
  for value in iterator:
    yield value


def main(emb_path, model_directory, model_metadata):
    """
        # print command line arguments
        for arg in sys.argv[1:]:
            print(arg)
        graph = tf.Graph()
        with graph.as_default():
             with tf.Session(graph=graph) as session:
                  ckpt = tf.train.get_checkpoint_state('./model/')
                  saver.restore(session, ckpt.model_checkpoint_path)
                  feed_dict = {tf_train_dataset : batch_data}
    """
    x_text,y = batchgen.get_dataset("good_test_astd-artwitter.csv", "bad_test_astd-artwitter.csv", 10000)
    
    x, y, _ = generateData(emb_path, 100, x_text, y, 176)
    
    sess=tf.Session()    
    #First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph(model_metadata)
    saver.restore(sess,tf.train.latest_checkpoint(model_directory))

    # Now, let's access and create placeholders variables and
    # create feed-dict to feed new data

    graph = tf.get_default_graph()
    x_input = graph.get_tensor_by_name("input_x:0")
    y_input = graph.get_tensor_by_name("input_y:0")
    dropout = graph.get_tensor_by_name("dropout_keep_prob:0")
    feed_dict ={x_input:x,y_input:y, dropout:0.5}

    #Now, access the op that you want to run. 
    op_to_restore = graph.get_tensor_by_name("output/predictions:0")

    out = sess.run(op_to_restore,feed_dict)
    predictions = [val[0] > val[1] for val in out]
    labels = [val[0] > val[1] for val in y]
    results = []
    correctPred = 0
    for i in range(len(labels)):
        results.append(labels[i]==predictions[i])
        if labels[i]==predictions[i]:
            correctPred += 1
    print(results)
    print(('Accuracy: %f' % (float(correctPred)/len(labels))))

    #out = sess.run(y,feed_dict)
    #print(out)
    #This will print 60 which is calculated 
    #using new values of w1 and w2 and saved value of b1. 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', action='store', dest='embeddings_path',
                    help='path to words vectors data set')
    parser.add_argument('-m', action='store', dest='model_dir',
                    help='path to saved model folder')
    parser.add_argument('-i', action='store', dest='model_metadata',
                    help='path to saved model metadata file')
    parser.add_argument('-d', action='store', dest='embdDim', type=int,
                    help='embeddings file vectors dimensions')
    results = parser.parse_args()
    main(results.embeddings_path, results.model_dir, results.model_metadata)
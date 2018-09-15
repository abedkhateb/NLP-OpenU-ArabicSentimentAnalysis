#! /usr/bin/env python

import sys
import tensorflow as tf
import batchgen
import numpy as np
from tensorflow.contrib import learn
import argparse
from vocabulary_extractor import generateData
import os
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def tokenizer(iterator):
  """Tokenizer generator.

  Args:
    iterator: Input iterator with strings.

  Yields:
    array of tokens per each value in the input.
  """
  for value in iterator:
    yield value

def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

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
    metadataFilePath = os.path.join(model_directory, model_metadata)
    saver = tf.train.import_meta_graph(metadataFilePath)
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

    cnf_matrix = confusion_matrix(labels, predictions)
    np.set_printoptions(precision=2)

    class_names = ['Positive', 'Negative']
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix for Sentiment classification')
    plt.show()
    #statsFilePath = os.path.join(model_directory, "stats.txt")
    #statsFile = open(statsFilePath, "w")
    #statsFile.write(('Accuracy: %f \n' % (float(correctPred)/len(labels))))
    #statsFile.close()
    #out = sess.run(y,feed_dict)
    #print(out)
    #This will print 60 which is calculated 
    #using new values of w1 and w2 and saved value of b1. 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', action='store', dest='embeddings_path',
                    help='path to words vectors data set')
    parser.add_argument('-m', action='store', dest='model_dir',
                    help='path to saved model checkpoints folder')
    parser.add_argument('-i', action='store', dest='model_metadata',
                    help='file name to saved model metadata file')
    parser.add_argument('-d', action='store', dest='embdDim', type=int,
                    help='embeddings file vectors dimensions')
    results = parser.parse_args()
    main(results.embeddings_path, results.model_dir, results.model_metadata)
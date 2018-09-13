#! /usr/bin/env python
import sys

#SELECT WHICH MODEL YOU WISH TO RUN:
from cnn_lstm import CNN_LSTM   #OPTION 0

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import batchgen
from tensorflow.contrib import learn
from vocabulary_extractor import generateData
import re
import argparse

from IPython import embed

# Parameters
# ==================================================

# Data loading params
dev_size = .10

# Model Hyperparameters
filter_sizes = [3,4,5]  #3
num_filters = 32
dropout_prob = 0.5 #0.5
l2_reg_lambda = 0.0

# Training parameters
batch_size = 128
#num_epochs = 20 #200
evaluate_every = 100 #100
checkpoint_every = 200 #100
num_checkpoints = 10 #Checkpoints to store

# Data Preparation
# ==================================================

filename = "astd-artwitter.csv"
goodfile = "good_astd-artwitter.csv"
badfile = "bad_astd-artwitter.csv"

def main(embdFilePath, embdDim, outFolderName, num_epochs):

    # Load data
    print("Loading data...")
    x_text, y = batchgen.get_dataset(goodfile, badfile, 5000) #TODO: MAX LENGTH

    max_document_length = max([len(x.split(" ")) for x in x_text])

    x, y, vocab_processor = generateData(embdFilePath, embdDim, x_text, y, max_document_length)
    # Randomly shuffle data
    np.random.seed(42)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(dev_size * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


    # Training
    # ==================================================

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            model = CNN_LSTM(x_train.shape[1],y_train.shape[1],len(vocab_processor.vocabulary_),embdDim,filter_sizes,num_filters,l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(model.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, outFolderName, timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", model.loss)
            acc_summary = tf.summary.scalar("accuracy", model.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            #TRAINING STEP
            def train_step(x_batch, y_batch,save=False):
                feed_dict = {
                  model.input_x: x_batch,
                  model.input_y: y_batch,
                  model.dropout_keep_prob: dropout_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, model.loss, model.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if save:
                    train_summary_writer.add_summary(summaries, step)

            #EVALUATE MODEL
            def dev_step(x_batch, y_batch, writer=None,save=False):
                feed_dict = {
                  model.input_x: x_batch,
                  model.input_y: y_batch,
                  model.dropout_keep_prob: 0.5
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, model.loss, model.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if save:
                    if writer:
                        writer.add_summary(summaries, step)

            #CREATE THE BATCHES GENERATOR
            batches = batchgen.gen_batch(list(zip(x_train, y_train)), batch_size, num_epochs)
        
            #TRAIN FOR EACH BATCH
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                if current_step % checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

            dev_step(x_dev, y_dev, writer=dev_summary_writer)

            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print("Saved model checkpoint to {}\n".format(path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', action='store', dest='embdFilePath',
                    help='path to words vectors data set')
    parser.add_argument('-d', action='store', dest='embdDim', type=int,
                    help='embeddings file vectors dimensions')
    parser.add_argument('-o', action='store', dest='outFolderName',
                    help='Output folder name')
    parser.add_argument('-p', action='store', dest='epochs', default=10,
                    type=int, help='num of epochs to run on batch')					
    results = parser.parse_args()
    main(results.embdFilePath, results.embdDim, results.outFolderName, results.epochs)
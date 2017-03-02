#
#
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys
import time

import numpy
from six.moves import urllib
from six.moves import xrange
import tensorflow as tf

WORK_DIRECTORY = 'data'#path of work directory
ACTION_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 3
VALIDATION_SIZE = 100
NUM_LABELS = 5
SEED = 66478
BATCH_SIZE = 8
NUM_EPOCHS = 10
EVAL_BATCH_SIZE = 8
EVAL_FREQUENCY =  10

tf.app.flags.DEFINE_boolean("self_test", False, "True if running a self test.")
FLAGS = tf.app.flags.FLAGS

#def load_data(filename): 
#  train_data_filename = open(filename, access_mode='r', buffering=-1)
#

def extract_data(filename, num_action):
  print('Extracting', filename)
  with open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_action)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
  return data

def extract_labels(filename, num_action):
  print('Extracting', filename)
  with open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_action)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
  return labels

# Generate a fake dataset that matches the dimensions of action set
def fake_data(num_action):
  data = numpy.ndarray(
      shape=(num_action, ACTION_SIZE, NUM_CHANNALS),
      dtype=numpy.float32)
  labels = numpy.zeros(shape=(num_action,), dtype=numpy.int64)
  for action in xrange(num_action):
    label = action % 2
    data[image, :, :, 0] = label - 0.5
    labels[action] = label
  return data, labels

# return the error rate based on dense predictions and sparse labels
def error_rate(predictions, labels):
  return 100.0 - (100 *
      numpy.sum(numpy.argmax(predictions, 1) == labels) / predictions.shape[0])

def main(argv=NOne, train_data_filepath, train_labels_filepath,
         test_data_filepath, test_labels_filepath):
  if FLAGS.self_test:
    # Construct fake dataset 
    print('Running self-test.')
    train_data, train_labels = fake_data(256)
    validation_data, validation_labels = fake_data(EVAL_BATCH_SIZE)
    test_data, test_labels = fake_data(EVAL_BATCH_SIZE)
    num_epochs = 1
  else:
    # Extract it into numpy arrays.
    train_data = extract_data(train_data_filepath, 100)
    train_labels = extract_labels(train_labels_filepath, 100)
    test_data = extract_data(test_data_filepath, 50)
    test_labels = extract_labels(test_labels_filepath, 50)
  
    # Generate a validation set
    validation_data = train_data[:VALIDATION_SIZE, ...]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_data = train_data[VALIDATION_SIZE:, ...]
    num_epochs = NUM_EPOCHS
  train_size = train_labels.shape[0]

  # The train samples and labels are fed to the graph.
  # These placeholder nodes will be fed 
  # a batch of training data at each training
  # step using the {feed_dict} argument to the Run() call below.
  train_data_node = tf.placeholder(
      tf.float32,
      shape=(BATCH_SIZE, ACTION_SIZE, ACTION_SIZE, NUM_CHANNELS))
  train_labels_node = tf.placeholder(tf.int64, shpae=(BATCH_SIZ,))
  eval_data = tf.placeholder(
      tf.float32,
      shape=(EVAL_BATCH_SIZE, ACTION_SIZE, ACTION_SIZE, NUM_CHANNELS))

  conv1_weights = tf.Variable(
      tf.truncated_normal([5, 5, NUM_CHANNELS, 32],
                          stddev=0.1,
                          seed=SEED))
  conv1_biases = tf.Variable(tf.zeros([32]))
  conv2_weights = tf.Variable(
      tf.truncated_normal([5, 5, 32, 64],
                          stddev=0.1,
                          seed=SEED))
  conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))
  
  # fully connected, depth 512.
  fc1_weights = tf.variable(
      tf.truncated_normal(
          [ACTION_SIZE // 4 * ACTION_SIZE // 4 * 64, 512],
          stddev=0.1,
          seed=SEED))
  fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))
  fc2_weights = tf.Variable(
      tf.truncated_normal([512, NUM_LABELS],
                          stddev=0.1,
                          seed=SEED))
  fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))

  # The model definition.
  # 2D convolution, with 'SAME' padding (i.e. the output feature map has
  # the same size as the input).
  # Note that {strides} is a 4D array whose shape matches 
  # the data layout: [action index, y, x, depth].
  def model(data, train=False)
    conv = tf.nn.conv2d(data,
                        conv1_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    # Bias and rectified linear non-linearity.
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))

    # Max pooling.
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    conv = tf.nn.conv2d(pool,
                        conv2_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')

    # Reshape the feature map cuboid into a 2D matrix to 
    # feed it to the fully connected layers.
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(
        pool,
        [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])

    # Fully connected layer.
    # Note that the "+' operation automatically broadcasts the biases.
    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
    # Add a 50% dropput during training only.
    # Dropout scales activations such that 
    # no rescaling is needed at evaluation time.
    if train:
      hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
    return tf.matmul(hidden, fc2_weights) + fc2_biases

  # Training computation: logits + cross-entropy loss.
  logits = model(train_data_node， True)
  loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, train_labels_node))

  # L2 regularization for the fully connected parameters.
  regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                  tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))    
  # Add the regularization term to the loss.
  loss += 5e-4 * regularizers

  # Optimizer: set up a variable that's incremented once per batch
  # and controls the learning rate decay.
  batch = tf.Variable(0)
  # Decay once per epoch, using an exponential schedule starting at 0.01.
  learning_rate = tf.train.exponential_decay(
      0.01,                    # Base learning rate.
      batch * BATCH_SIZE,      # Current index into the dataset.   
      train_size,              # Decay step.
      0.95,                    # Decay rate.
      staircase=True)
  # Use simple momentum for the optimization.
  optimizer = tf.train.MomentumOptimizer(learning_rate,
                                         0.9).minimize(loss,
                                                       global_step=batch)

  # Predictions for the current training minibatch.
  train_prediction = tf.nn.softmax(logits)
  # Predictions for the test and validation.
  eval_prediction = tf.nn.softmax(model(eval_data))

  # Small utility function to evaluate a dataset by feeding batches of data
  # to {eval_data} and pulling the results from {eval_prediction}.
  # Saves memory and enables this to run on smaller GPUs.
  # Get all predictions for a dataset by running it in small batches.
  def eval_in_batches(data, sess):
    size = data.shape[0]
    if size < EVAL_BATCH_SIZE:
      raise ValueError("batch size for evals larger than dataset: %d"      
                       % size)
    predictions = numpy.ndarray(shape=(size, NUM_LABELS), 
                                dtype=numpy.float32)
    for begin in xrange(0, size, EVAL_BATCH_SIZE):
      end = begin + EVAL_BATCH_SIZE
      if end <= size:
        predictions[begin:end, :] = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[begin:end, ...]})
      else:
        batch_predictions = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
        predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions

  # Create a local session to run the training
  start_time = time.time()
  with tf.Session() as sess:
    # Run all the initializers to prepare the trainable parameters.
    tf.initialize_all_variables().run()
    print('Initialized!')

    # Loop through training steps.
    for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
      # Compute the offset of the current minibatch in the data.
      # Note that we could use better randomization across epochs.
      offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
      batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
      batch_labels = train_labels[offset:(offset + BATCH_SIZE)]

      # This dictionary maps the batch data to the node 
      # in the graph it should be fed to.
      feed_dict = {train_data_node: batch_data,
                   train_labels_node: batch_labels}
      
      # Run the graph and fetch some of the nodes.
      _, l, lr, predictions = sess.run(
          [optimizer, loss, learning_rate, train_prediction],
          feed_dict=feed_dict)
      if step % EVAL_FREQUENCY == 0:
        elapsed_time = time.time() - start_time
        start_time = time.time()
        print('Step %d (epoch %.2f, %.1f ms' %
              (step, float(step) * BATCH_SIZE / train_size,
               1000 * elapsed_time / EVAL_FREQUENCY))
        print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
        print('Minibatch error: %.1f%%' % error_rate(predictions, 
                                                     batch_labels))
        print('Validation error: %.1f%%' % error_rate(
            eval_in_batches(validation_data, sess), validation_labels))
        sys.stdout.flush()

    # Final result of test error rate.
    test_error = error_rate(eval_in_batches(test_data, sess), test_labels)
    print('Test error: %.1f%%' % test_error)
    if FLAGS.self_test:
      print('test_error', test_error)
      assert test_error == 0.0, 'expected 0.0 test_error, got %.2f' 
          % (test_error,)

# Run main.
if __name__ == '__main__':
  tf.app.run()

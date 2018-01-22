import math
import pickle as p
import tensorflow as tf
import numpy as np
import json
from deepsent import utils

# set variables
tweet_size = 20
hidden_size = 100
vocab_size = 7597
batch_size = 64

# this just makes sure that all our following operations will be placed in the right graph.
tf.reset_default_graph()

# create a session variable that we can run later.
session = tf.Session()

# make placeholders for data we'll feed in
tweets = tf.placeholder(tf.float32, [None, tweet_size, vocab_size])
labels = tf.placeholder(tf.float32, [None])

# make the lstm cells, and wrap them in MultiRNNCell for multiple layers
lstm_cell_1 = tf.contrib.rnn.LSTMCell(hidden_size)
lstm_cell_2 = tf.contrib.rnn.LSTMCell(hidden_size)
multi_lstm_cells = tf.contrib.rnn.MultiRNNCell(cells=[lstm_cell_1, lstm_cell_2], state_is_tuple=True)

# define the op that runs the LSTM, across time, on the data
_, final_state = tf.nn.dynamic_rnn(multi_lstm_cells, tweets, dtype=tf.float32)


# a useful function that takes an input and what size we want the output
# to be, and multiples the input by a weight matrix plus bias (also creating
# these variables)
def linear(input_, output_size, name, init_bias=0.0):
    shape = input_.get_shape().as_list()
    with tf.variable_scope(name):
        W = tf.get_variable("weight_matrix", [shape[-1], output_size], tf.float32,
                            tf.random_normal_initializer(stddev=1.0 / math.sqrt(shape[-1])))
    if init_bias is None:
        return tf.matmul(input_, W)
    with tf.variable_scope(name):
        b = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(init_bias))
    return tf.matmul(input_, W) + b


# define that our final sentiment logit is a linear function of the final state
# of the LSTM
sentiment = linear(final_state[-1][-1], 1, name="output")

sentiment = tf.squeeze(sentiment, [1])

# define cross entropy loss function
loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=sentiment, labels=labels)
loss = tf.reduce_mean(loss)

# round our actual probabilities to compute error
prob = tf.nn.sigmoid(sentiment)
prediction = tf.to_float(tf.greater_equal(prob, 0.5))
pred_err = tf.to_float(tf.not_equal(prediction, labels))
pred_err = tf.reduce_sum(pred_err)

# define our optimizer to minimize the loss
optimizer = tf.train.AdamOptimizer().minimize(loss)

# initialize any variables
tf.global_variables_initializer().run(session=session)

# load our data and separate it into tweets and labels
train_data = json.load(open("/home/iamukasa/PycharmProjects/NLP/data/testtweets.json", 'r'))
train_data = list(map(lambda row: (np.array(row[0], dtype=np.int32), str(row[1])), train_data))

train_tweets = np.array([t[0] for t in train_data])
train_labels = np.array([int(t[1]) for t in train_data])

test_data = json.load(open("/home/iamukasa/PycharmProjects/NLP/data/testtweets.json", 'r'))
test_data = map(lambda row: (np.array(row[0], dtype=np.int32), str(row[1])), test_data)

# we are just taking the first 1000 things from the test set for faster evaluation
test_data = test_data[0:1000]
test_tweets = np.array([t[0] for t in test_data])
one_hot_test_tweets = utils.one_hot(test_tweets, vocab_size)
test_labels = np.array([int(t[1]) for t in test_data])

# we'll train with batches of size 128.  This means that we run
# our model on 128 examples and then do gradient descent based on the loss
# over those 128 examples.
num_steps = 1000

for step in range(num_steps):
    # get data for a batch
    offset = (step * batch_size) % (len(train_data) - batch_size)
    batch_tweets = utils.one_hot(train_tweets[offset: (offset + batch_size)], vocab_size)
    batch_labels = train_labels[offset: (offset + batch_size)]

    # put this data into a dictionary that we feed in when we run
    # the graph.  this data fills in the placeholders we made in the graph.
    data = {tweets: batch_tweets, labels: batch_labels}

    # run the 'optimizer', 'loss', and 'pred_err' operations in the graph
    _, loss_value_train, error_value_train = session.run(
        [optimizer, loss, pred_err], feed_dict=data)

    # print stuff every 50 steps to see how we are doing
    if (step % 50 == 0):
        print("Minibatch train loss at step", step, ":", loss_value_train)
        print("Minibatch train error: %.3f%%" % error_value_train)

        # get test evaluation
        test_loss = []
        test_error = []
        for batch_num in range(int(len(test_data) / batch_size)):
            test_offset = (batch_num * batch_size) % (len(test_data) - batch_size)
            test_batch_tweets = one_hot_test_tweets[test_offset: (test_offset + batch_size)]
            test_batch_labels = test_labels[test_offset: (test_offset + batch_size)]
            data_testing = {tweets: test_batch_tweets, labels: test_batch_labels}
            loss_value_test, error_value_test = session.run([loss, pred_err], feed_dict=data_testing)
            test_loss.append(loss_value_test)
            test_error.append(error_value_test)

        print("Test loss: %.3f" % np.mean(test_loss))
        print("Test error: %.3f%%" % np.mean(test_error))
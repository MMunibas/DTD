import tensorflow as tf
import numpy as np
import sys
from datetime import datetime
import utlty as util

# read in parameters for training the NN
if len(sys.argv) != 6:
    print("usage: script.py nt nv seed n_epochs batch_size")
    print("nt:         number of training   samples")
    print("nv:         number of validation samples")
    print("seed:       seed for selecting training samples etc. (to make runs reproducible)")
    print("n_epochs:   number of training epochs")
    print("batch_size: number of data sets to include in one batch for training (must not be bigger than nt)")
    quit()

nt = int(sys.argv[1])
nv = int(sys.argv[2])
seed = int(sys.argv[3])
n_epochs = int(sys.argv[4])
batch_size = int(sys.argv[5])

# saves approximately every 10% of the training
nsave = n_epochs//200
assert nsave != 0

# load datasets
print("Loading Datasets...\n")
data = util.DataContainer(nt, nv, seed, False)
print("total number of data:", data.num_data)
print("training examples:   ", data.num_train)
print("validation examples: ", data.num_valid)
print("test examples:       ", data.num_test)
print()

# fire up tensorflow
print("Loading TensorFlow...\n")

# specify further parameters of the NN and training schedule

# retrain the model from an existing file
retrain_model = True
model_parameter_save = "save/NN-seed1-nt3000-nv600-43-6-6-44_loss0.29111156_2020-06-06-19-46-11-40"

# number of features
n_inputs = data.num_features

# number of neurons in each hidden layer (and how many hidden layers)
n_hidden = [6, 6]

# check that we have at least 1 hidden layer
assert len(n_hidden) >= 1

# number of outputs
n_outputs = data.num_outputs

# set the learning rate parameters
# the default value I use is 1e-3
global_step = tf.Variable(0, trainable=False)
zero_global_step_op = tf.assign(global_step, 0)
starter_learning_rate = 8.0e-4
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           100000, 0.5, staircase=False)
# passing global_step to minimize() will increment it at each step.

# lambda multiplier for L2 regularization (0 -> no regularization)
l2_lambda = 0.0

# for saving logs
now = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")
root_logdir = "logs"

# encode NN architecture in string
NNstr = "NN-seed"+str(seed)+"-nt"+str(data.num_train)+"-nv" + \
    str(data.num_valid)+"-"+str(n_inputs)+"-"
for i in range(len(n_hidden)):
    NNstr = NNstr + str(n_hidden[i]) + "-"
NNstr = NNstr + str(n_outputs)
logdir = "{}/run_".format(root_logdir)+NNstr+"_{}/".format(now)

# create NN

# define a function to create a layer of neurons (getting X as an input with n_out outputs)


def neuron_layer(X, n_out, activation_fn=lambda x: x, scope=None, factor=2.0):
    with tf.variable_scope(scope):

        # define a layer
        n_in = X.shape[1].value
        W = tf.Variable(tf.truncated_normal(
            [n_in, n_out], stddev=tf.sqrt(factor/(n_in+n_out))), name="W")
        b = tf.Variable(tf.truncated_normal([n_out], stddev=0), name="b")
        y = activation_fn(tf.add(tf.matmul(X, W), b))

        # l2 loss term for regularization
        l2_W = tf.nn.l2_loss(W, name="l2_W")

        # add to collections
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, l2_W)
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)
        tf.add_to_collection(tf.GraphKeys.BIASES, b)
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, y)

        # create histogram summaries for monitoring the weights and biases
        tf.summary.histogram("weights", W)
        tf.summary.histogram("biases",  b)

        return y

# define a function to create a Multilayer Perceptron getting X as a feature input, n_hidden is a list containing the number of neurons in the hidden layers (at least 1 hidden layer)
# and n_outputs specifies the number of outputs of the MLP


def MLP(X, n_hidden, n_outputs, activation_fn=lambda x: x, scope=None, factor=2.0):

    # check that there is at least 1 hidden layer
    assert len(n_hidden) >= 1
    with tf.name_scope(scope):

        # list that stores the hidden layers
        hidden = []

        # create and append the hidden layers

        hidden.append(neuron_layer(
            X, n_hidden[0], activation_fn=activation_fn, scope="hidden0", factor=factor))
        hidden.append(neuron_layer(hidden[0], n_hidden[1],
                                   activation_fn=activation_fn, scope="hidden1", factor=factor))
        return tanh(neuron_layer(hidden[len(n_hidden)-1], n_outputs, scope="output", factor=2.0))

# activation function to use for the output layer


def tanh(x):
    return 8.0*tf.tanh(x)

# activation function to use for the hidden layers


def actf(x):
    return tf.nn.softplus(x)-tf.log(2.0)


# create neural network (and placeholders for feeding)
print("Creating neural network...\n")
X = tf.placeholder(tf.float32, shape=[None, n_inputs],  name="X")
y = tf.placeholder(tf.float32, shape=[None, n_outputs], name="y")

# (factor = 1 is only for the self-normalizing input functions!)
yhat = MLP(X, n_hidden, n_outputs, activation_fn=actf,
           scope="neuralnetwork", factor=2.0)

# define loss function: here RMSE + regularization loss
with tf.name_scope("loss"):
    l2_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    # loss = tf.reduce_mean(tf.squared_difference(tf.log(y+1.0),tf.log(y+tf.abs(tf.subtract(y,yhat))+1.0))) + l2_lambda*l2_loss
    loss = tf.reduce_mean(tf.squared_difference(y, yhat))

with tf.name_scope("score"):
    score = tf.reduce_mean(tf.abs(tf.subtract(y, yhat)))

with tf.name_scope("rmsd"):
    rmsd = tf.reduce_mean(tf.squared_difference(y, yhat))

# define training method
with tf.name_scope("train"):
    training_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

# for logging stats

# mean absolute error
score_for_train = tf.constant(0.0)
score_for_valid = tf.constant(0.0)
score_for_best = tf.constant(0.0)
tf.summary.scalar("score-train", score_for_train)
tf.summary.scalar("score-valid", score_for_valid)
tf.summary.scalar("score-best",  score_for_best)

# loss function
loss_for_train = tf.constant(0.0)
loss_for_valid = tf.constant(0.0)
loss_for_best = tf.constant(0.0)
tf.summary.scalar("loss-train", loss_for_train)
tf.summary.scalar("loss-valid", loss_for_valid)
tf.summary.scalar("loss-best", loss_for_best)

# root mean squared error
rmsd_for_train = tf.constant(0.0)
rmsd_for_valid = tf.constant(0.0)
rmsd_for_best = tf.constant(0.0)
tf.summary.scalar("rmsd-train", rmsd_for_train)
tf.summary.scalar("rmsd-valid", rmsd_for_valid)
tf.summary.scalar("rmsd-best", rmsd_for_best)

# merged summary op
summary_op = tf.summary.merge_all()

# create file writer for writing out summaries
file_writer = tf.summary.FileWriter(logdir=logdir,
                                    graph=tf.get_default_graph(),
                                    flush_secs=120)

# define saver nodes (max_to_keep=None lets the saver keep everything)

# save only the best model
saver_best = tf.train.Saver(name="saver_best", max_to_keep=50)

# save a checkpoint every few steps
saver_step = tf.train.Saver(name="saver_step", max_to_keep=200)

# counter that keeps going up for the best models
number_best = 0

# initialize the best score/loss/validation to huge values
score_best = np.finfo(dtype=float).max
loss_best = np.finfo(dtype=float).max
rmsd_best = np.finfo(dtype=float).max

all_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

# evaluating the model to store the weights and biases of the NN
print("Starting evaluation...\n")

with tf.Session() as sess:

    # initialize all variables
    if retrain_model:
        saver_step.restore(sess, model_parameter_save)
        sess.run(zero_global_step_op)

    else:
        tf.global_variables_initializer().run()

    varlist = sess.run([all_variables])
    vardictionary = {}
    i = 0
    print(len(varlist[0]))
    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        vardictionary[var.name] = varlist[0][i]
        print(var.name, varlist[0][i].shape)
        i = i+1

    print(vardictionary.keys())
    h0W = vardictionary['neuralnetwork/hidden0/W:0']
    h0b = vardictionary['neuralnetwork/hidden0/b:0']
    h1W = vardictionary['neuralnetwork/hidden1/W:0']
    h1b = vardictionary['neuralnetwork/hidden1/b:0']
    outW = vardictionary['neuralnetwork/output/W:0']
    outb = vardictionary['neuralnetwork/output/b:0']

    # save weights and biases of the NN
    np.savetxt('./NN_parameters/Coeff_h0W.dat', h0W, delimiter=',')
    np.savetxt('./NN_parameters/Coeff_h0b.dat', h0b, delimiter=',')
    np.savetxt('./NN_parameters/Coeff_h1W.dat', h1W, delimiter=',')
    np.savetxt('./NN_parameters/Coeff_h1b.dat', h1b, delimiter=',')
    np.savetxt('./NN_parameters/Coeff_outW.dat', outW, delimiter=',')
    np.savetxt('./NN_parameters/Coeff_outb.dat', outb, delimiter=',')

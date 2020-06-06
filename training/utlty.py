import numpy as np

# define a DataContainer class which stores data


class DataContainer:
    def __repr__(self):
        return "DataContainer"

    def __str__(self):
        return self._data.to_string()

    def __init__(self, nt, nv, seed=None, load_test=False):

        # define the total number of data sets
        self._num_data = 3600

# ---------------------------------------------------------------
# use the following data structure for the input file:

# line 1 for data set 1: feature 1,feature 2,...,feature n,output 1,output 2,
# ...,output m
# line 2 for data set 2: feature 1,feature 2,...,feature n,output 1,output 2,
# ...,output m
# ...
# line l for data set l: feature 1,feature 2,...,feature n,output 1,output 2,
# ...,output m

# ---------------------------------------------------------------

        # total number of features and outputs
        self._num_features = 43
        self._num_outputs = 44

        # whether to load the test set or not
        self._load_test = load_test
        self._random_state = np.random.RandomState(seed=seed)

        # load the data sets
        data = np.genfromtxt('input_and_reference_train_valid.txt', delimiter=',')

        # create a shuffled list of indices
        indices = self._random_state.permutation(np.arange(0, self._num_data))

        # store indices of training, validation and test data sets
        indices_train = indices[0:nt]
        indices_valid = indices[nt:nt+nv]
        indices_test = indices[nt+nv:]

        # np.savetxt('valid_indices.txt', indices_valid, delimiter=',')
        # np.savetxt('test_indices.txt', indices_test, delimiter=',')

        # save number of training, validation and test data sets
        self._num_train = indices_train.shape[0]
        self._num_valid = indices_valid.shape[0]
        self._num_test = indices_test.shape[0]

        # store the 3 different sets of data
        self._data_train = data[indices_train, :]
        self._data_valid = data[indices_valid, :]
        if self._load_test:
            self._data_test = data[indices_test, :]

        # calculate and save running mean/stdev of the features
        # for standardization
        n = 0
        S = np.zeros(self._num_features, dtype=float)
        m = np.zeros(self._num_features, dtype=float)
        for i in range(self._num_train):

            # loop through the descriptor and update mean/stdev
            # keeps track of how many samples have been analyzed for mean/stdev
            n += 1
            for j in range(self._num_features):

                # update mean/stdev
                m_prev = m[j]
                m[j] += (self._data_train[i, j]-m[j])/n
                S[j] += (self._data_train[i, j]-m[j]) * (self._data_train[i, j]-m_prev)

        stdev = np.sqrt(S/n)

        np.savetxt('./NN_parameters/Coeff_mval_input.txt', m, delimiter=',')
        np.savetxt('./NN_parameters/Coeff_stdv_input.txt', stdev, delimiter=',')

        # standardize the inputs

        # training set
        for i in range(self._num_train):
            for j in range(self._num_features):
                if stdev[j] > 0.0:
                    self._data_train[i, j] = (self._data_train[i, j]-m[j])/stdev[j]
                else:
                    self._data_train[i, j] = (self._data_train[i, j]-m[j])

        # validation set
        for i in range(self._num_valid):
            for j in range(self._num_features):
                if stdev[j] > 0.0:
                    self._data_valid[i, j] = (self._data_valid[i, j]-m[j])/stdev[j]
                else:
                    self._data_valid[i, j] = (self._data_valid[i, j]-m[j])

        # test set
        if self._load_test:
            for i in range(self._num_test):
                for j in range(self._num_features):
                    if stdev[j] > 0.0:
                        self._data_test[i, j] = (self._data_test[i, j]-m[j])/stdev[j]
                    else:
                        self._data_test[i, j] = (self._data_test[i, j]-m[j])

        # calculate and save running mean/stdev of the outputs
        # for standardization

        n = 0
        S = np.zeros(self._num_outputs, dtype=float)
        m = np.zeros(self._num_outputs, dtype=float)
        for i in range(self._num_train):

            # loop through the descriptor and update mean/stdev
            # keeps track of how many samples have been analyzed for mean/stdev
            n += 1
            for j in range(self._num_features, self._num_features + self._num_outputs):

                # Update mean/stdev
                m_prev = m[j-self._num_features]
                m[j-self._num_features] += (self._data_train[i, j]-m[j-self._num_features])/n
                S[j-self._num_features] += (self._data_train[i, j] -
                                            m[j-self._num_features]) * (self._data_train[i, j]-m_prev)

        self.mval = m
        self.stdv = np.sqrt(S/n)
        stdev = np.sqrt(S/n)

        np.savetxt('./NN_parameters/Coeff_mval_output.txt', m, delimiter=',')
        np.savetxt('./NN_parameters/Coeff_stdv_output.txt', stdev, delimiter=',')

        # standardize the outputs

        # training set
        for i in range(self._num_train):
            for j in range(self._num_features, self._num_features + self._num_outputs):
                if stdev[j-self._num_features] > 0.0:
                    self._data_train[i, j] = (
                        self._data_train[i, j]-m[j-self._num_features])/stdev[j-self._num_features]
                else:
                    self._data_train[i, j] = (self._data_train[i, j]-m[j-self._num_features])

        # validation set
        for i in range(self._num_valid):
            for j in range(self._num_features, self._num_features + self._num_outputs):
                if stdev[j-self._num_features] > 0.0:
                    self._data_valid[i, j] = (
                        self._data_valid[i, j]-m[j-self._num_features])/stdev[j-self._num_features]
                else:
                    self._data_valid[i, j] = (self._data_valid[i, j]-m[j-self._num_features])

        # test set
        if self._load_test:
            for i in range(self._num_test):
                for j in range(self._num_features, self._num_features + self._num_outputs):
                    if stdev[j-self._num_features] > 0.0:
                        self._data_test[i, j] = (
                            self._data_test[i, j]-m[j-self._num_features])/stdev[j-self._num_features]
                    else:
                        self._data_test[i, j] = (self._data_test[i, j]-m[j-self._num_features])

# ------------------------------------------------------------------------------

        # for retrieving batches
        self._index_in_epoch = 0

    # total number of data sets
    @property
    def num_data(self):
        return self._num_data

    # number of training data sets
    @property
    def num_train(self):
        return self._num_train

    # number of validation data sets
    @property
    def num_valid(self):
        return self._num_valid

    # number of test data sets
    @property
    def num_test(self):
        return self._num_test

    @property
    def num_features(self):
        return self._num_features

    @property
    def num_outputs(self):
        return self._num_outputs

    # shuffle the training data sets
    def shuffle_train_data(self):
        indices = self._random_state.permutation(np.arange(0, self._num_train))
        self._data_train = self._data_train[indices, :]

    # return a batch of samples from the training data sets
    def next_batch(self, batch_size=1):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        # epoch is finished, hence the data needs to be shuffled
        if self._index_in_epoch > self.num_train:

            # shuffle training data
            self.shuffle_train_data()
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_train
        end = self._index_in_epoch
        return self._data_train[start:end, :self._num_features], self._data_train[start:end, self._num_features:self._num_features+self._num_outputs]

    # define get - functions of the DataContainer class

    def get_train_data(self):
        return self._data_train[:, :self._num_features], self._data_train[:, self._num_features:self._num_features+self._num_outputs]

    def get_valid_data(self):
        return self._data_valid[:, :self._num_features], self._data_valid[:, self._num_features:self._num_features+self._num_outputs]

    def get_test_data(self):
        assert self._load_test
        return self._data_test[:, :self._num_features], self._data_test[:, self._num_features:self._num_features+self._num_outputs]

    def get_one_data(self, nindx):
        return self._data_valid[nindx, :self._num_features], self._data_valid[nindx, self._num_features:self._num_features+self._num_outputs]

    def get_onetest_data(self, nindx):
        return self._data_test[nindx, :self._num_features], self._data_test[nindx, self._num_features:self._num_features+self._num_outputs]

    def get_onetrain_data(self, nindx):
        return self._data_train[nindx, :self._num_features], self._data_train[nindx, self._num_features:self._num_features+self._num_outputs]

    def get_mval(self):
        return self.mval

    def get_stdv(self):
        return self.stdv

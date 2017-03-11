import os
import numpy as np

num_hidden_units = 300      #number of nodes in the hidden unit
minibatch_size = 100    #size of each mini_batch
regularization_rate = 0.01      #coefficient for regularization
learning_rate = 0.001       #coefficient to decide the rate of learning

#function to implement the ReLu (Rectified Linear Unit) function
def relu_function(matrix_content, matrix_dim_x, matrix_dim_y):

    ret_vector = np.zeros((matrix_dim_x, matrix_dim_y))

    for i in range(0, matrix_dim_x):
        for j in range(0, matrix_dim_y):
            if matrix_content[i, j]> 0:
                ret_vector[i,j] = matrix_content[i,j]
            else:
                ret_vector[i,j] = 0

    return ret_vector

#function to implement the gradient of ReLu (Rectified Linear Unit) function
def grad_relu(matrix_content, matrix_dim_x, matrix_dim_y):

    ret_vector = np.zeros((matrix_dim_x, matrix_dim_y))

    for i in range(matrix_dim_x):
        for j in range(matrix_dim_y):
            if matrix_content[i,j] > 0:
                ret_vector[i,j] = 1
            else:
                ret_vector[i,j] = 0

    return ret_vector

#function to implement Softmax
def softmax_function(vector_content):
    return np.exp(vector_content - np.max(vector_content)) / np.sum(np.exp(vector_content - np.max(vector_content)), axis=0)

#function to create mini_batches while training the MLP model
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]

    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)

    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)

        yield inputs[excerpt], targets[excerpt]

#function to train the MLP model
def train(trainX, trainY):

    #initializing the parameters
    w1_mat = np.random.randn(num_hidden_units, 28*28) *np.sqrt(2./(num_hidden_units+28*28))
    w2_mat = np.random.randn(10, num_hidden_units) *np.sqrt(2./(10+num_hidden_units))
    b1_vec = np.zeros((num_hidden_units, 1))
    b2_vec = np.zeros((10, 1))

    trainX = np.reshape(trainX, (trainX.shape[0], 28*28))
    trainY = np.reshape(trainY, (trainY.shape[0], 1))

    for num_epochs in range(25) :
        if num_epochs%2==0:
            print "Current epoch number : ", num_epochs

        for batch in iterate_minibatches (trainX, trainY, minibatch_size, shuffle=True):

            x_batch, y_batch = batch
            x_batch = x_batch.T
            y_batch = y_batch.T

            #forward propagation to get the intermediate values and the final output value
            z1 =  np.dot(w1_mat, x_batch) + b1_vec
            a1 = relu_function(z1, num_hidden_units, minibatch_size)
            z2 =  np.dot(w2_mat, a1) + b2_vec
            a2_softmax = softmax_function(z2)

            #ground truth used to find the cross_entropy error
            gt_vector = np.zeros((10, minibatch_size))
            for example_num in range(minibatch_size):
                gt_vector[y_batch[0, example_num], example_num] = 1

            #applying regularization to keep the magnitude of weights within a limit
            d_w2_mat = regularization_rate*w2_mat
            d_w1_mat = regularization_rate*w1_mat


            #backpropagation
            delta_2 = (a2_softmax - gt_vector)
            d_w2_mat = d_w2_mat + np.dot(delta_2, (np.matrix(a1)).T)
            d_b2_vec = np.sum(delta_2, axis = 1, keepdims=True)

            delta_1 = np.multiply((np.dot(w2_mat.T, delta_2)), grad_relu(z1, num_hidden_units, minibatch_size))
            d_w1_mat = d_w1_mat + np.dot(delta_1, np.matrix(x_batch).T)
            d_b1_vec = np.sum(delta_1, axis = 1, keepdims=True)

            d_w2_mat = d_w2_mat/minibatch_size
            d_w1_mat = d_w1_mat/minibatch_size
            d_b2_vec = d_b2_vec/minibatch_size
            d_b1_vec= d_b1_vec/minibatch_size

            w2_mat = w2_mat - learning_rate*d_w2_mat
            b2_vec = b2_vec - learning_rate*d_b2_vec

            w1_mat = w1_mat - learning_rate*d_w1_mat
            b1_vec = b1_vec - learning_rate*d_b1_vec

        #writing the final parameter values obtained in the folder "weights"
        params_dir ="./weights"
        fd_w1 = open(os.path.join(params_dir,'w1_values'), "w")
        fd_b1 = open(os.path.join(params_dir,'b1_values'), "w")
        fd_w2 = open(os.path.join(params_dir,'w2_values'), "w")
        fd_b2 = open(os.path.join(params_dir,'b2_values'), "w")

        w1_mat.tofile(fd_w1)
        b1_vec.tofile(fd_b1)
        w2_mat.tofile(fd_w2)
        b2_vec.tofile(fd_b2)

        fd_w1.close()
        fd_b1.close()
        fd_w2.close()
        fd_b2.close()

#function to test the MLP model
def test(testX):
    output_labels = np.zeros(testX.shape[0])

    num_examples = testX.shape[0]

    testX = np.reshape(testX, (num_examples, 28*28))
    testX = testX.T

    #read the parameter values
    params_dir ="./weights"
    fd_w1 = open(os.path.join(params_dir,'w1_values'))
    fd_b1 = open(os.path.join(params_dir,'b1_values'))
    fd_w2 = open(os.path.join(params_dir,'w2_values'))
    fd_b2 = open(os.path.join(params_dir,'b2_values'))

    loaded = np.fromfile(file=fd_w1, dtype=np.float)
    w1_mat = loaded.reshape((num_hidden_units, 28*28)).astype(np.float)

    loaded = np.fromfile(file=fd_b1, dtype=np.float)
    b1_vec = loaded.reshape((num_hidden_units, 1)).astype(np.float)

    loaded = np.fromfile(file=fd_w2, dtype=np.float)
    w2_mat = loaded.reshape((10, num_hidden_units)).astype(np.float)

    loaded = np.fromfile(file=fd_b2, dtype=np.float)
    b2_vec = loaded.reshape((10, 1)).astype(np.float)

    #forward propagation to get the predicted values
    z1 =  np.dot(w1_mat, testX) + b1_vec    #b1_vec ->200 X 1
    a1 = relu_function(z1, num_hidden_units, num_examples)

    z2 =  np.dot(w2_mat, a1) + b2_vec

    a2_softmax = softmax_function(z2)

    for i in range(num_examples):
        pred_col = a2_softmax[:, [i]]
        output_labels[i] = np.argmax(pred_col)

    return output_labels
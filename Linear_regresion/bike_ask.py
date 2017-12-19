import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
rng = np.random


data_path = 'Bike-Sharing-Dataset/hour.csv'

rides = pd.read_csv(data_path)


# Dummy variables with get_dummies()
dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
    rides = pd.concat([rides, dummies], axis=1)

fields_to_drop = ['instant', 'dteday', 'season', 'weathersit', 
                  'weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = rides.drop(fields_to_drop, axis=1)


# Scaling target variable
quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
# Store scalings in a dictionary so we can convert back later
scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean)/std

# Splitting the data into training, testing
# save the data approximately the last 21 days
test_data = data[-21*24:]

# remove the test data from the data set
data = data[:-21*24]

# separate the data into features and targets
target_fields = ['cnt', 'casual', 'registered']

features, targets = data.drop(target_fields, axis=1), data[target_fields]
test_features, test_targets = test_data.drop(target_fields, axis=1), test_data['cnt']


# Building the Network

# Parametres

epochs = 1000
batch_size = 128
display_step = 50
learning_rate = 0.001

n_hidden = 15
n_output = 1
n_input = features.shape[1]

n_samples = features.shape[0]

# tf graph input
inputs = tf.placeholder(tf.float32)
targets = tf.placeholder(tf.float32)


# set model weights
weights = {
    'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden])),
    'out': tf.Variable(tf.truncated_normal([n_hidden, n_output]))
}
biases = {
    'h1': tf.Variable(tf.truncated_normal([n_hidden])),
    'out': tf.Variable(tf.truncated_normal([n_output]))
}

# Create the model

def neural_net(x):
    # Hidden fully connected layer and sigmoid activation function
    layer_1 = tf.sigmoid(tf.add(tf.matmul(x, weights['h1']), biases['h1']))
    
    # Output fully connected layer
    layer_out = tf.add(tf.matmul(layer_1, weights['out']), biases['out'])
    
    return layer_out


# Mean squared error
cost = tf.reduce_sum(tf.pow(neural_net(inputs)-targets, 2))/(2*n_samples)

# Define loss and optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(cost)

# Evaluate model (with test logits, for dropout to ve disabled)
correct_pred = tf.equal(tf.argmax(cost, 1), tf.argmax(targets, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variable
init = tf.global_variables_initializer()


# Start training
with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(epochs):
        
        # mini-batching with .sample() 
        target_fields = ['cnt', 'casual', 'registered']
        batch_data = data.sample(n=batch_size)
        batch_features, batch_targets = batch_data.drop(target_fields, axis=1), \
                                        batch_data['cnt']
        batch_X, batch_y = batch_features.values, batch_targets
        
        for (X, y) in zip(batch_X,batch_y):
            
            # run optimization
            sess.run(train_op, feed_dict={inputs: X, targets: y})
            if step % display_step == 0 or step == 1:

            	
                # Calculate batch loss and accuracy
                loss, acc = sess.run([cost, accuracy], feed_dict={inputs: X, 
                                                              targets: y})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                        "{:.4f}".format(loss) + ", Training Accuracy= " + \
                          "{:.3f}".format(acc))
    
    print("Optimization Finished!")

    # Calculate accuracy for MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={inputs: test_features,
                                      targets: test_targets}))



from __future__ import division, print_function, absolute_import

import tensorflow as tf

import numpy as np

from numpy import genfromtxt

#define hyper-parameters,no. of epochs, batch size
learning_rate = 0.0001
num_steps = 500
batch_size = 32
display_step = 10


num_input =28*28  # Z500 data input (img shape: 28*28)
num_classes = 4 # weather patterns total classes is 4
dropout = 0.5 # Dropout, probability to keep units. Don't change this 
# These are placeholders in computation graph
X = tf.placeholder(tf.float32, [None, num_input*3]) 
Y = tf.placeholder(tf.float32, [None, num_classes])
# placeholder for dropout in feed forward layer
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

#define convolution layer in 2D. Requires data input weights, and stride always equ#al to 1 

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

#max pool function. k=2 always unless you wanna change more stuff
def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

#this is the function that defines the conv-net computation graph
def conv_net(x, weights, biases, dropout):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 28, 28, 3])
    
    conv01=conv2d(x, weights['wc01'], biases['bc01']) #no maxpool
       
    conv02=conv2d(conv01, weights['wc02'], biases['bc02']) #no maxpool
    # Convolution Layer
    conv1 = conv2d(conv02, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    
    'wc01': tf.Variable(tf.random_normal([5, 5, 3, 8])),
    
    'wc02': tf.Variable(tf.random_normal([5, 5, 8, 16])),
    'wc1': tf.Variable(tf.random_normal([5, 5, 16, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 4 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, num_classes]))
}

biases = {
  
    'bc01': tf.Variable(tf.random_normal([8])),
    'bc02': tf.Variable(tf.random_normal([16])),
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Construct model
logits = conv_net(X, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))+0.2*tf.nn.l2_loss(weights['wc1']) +0.2*tf.nn.l2_loss(weights['wc2'])+0.2*tf.nn.l2_loss(weights['wd1']) +0.2*tf.nn.l2_loss(weights['out'])
#loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.initialize_all_variables()
saver = tf.train.Saver()

with tf.Session() as sess:
    

    # Run the initializer
    sess.run(init)
    count=1
    
    train=np.load('training_40ensemble_4classes_fuzzy.npy')
  #  train=trainfull[:,range(0,int(3*(np.size(trainfull,1)/4)))]
    label=np.load('labels_40ensemble_4classes_fuzzy.npy')
  #  label=labelfull[range(0,int(3*(np.size(trainfull,1))/4)),:]
    for step in range(1, num_steps+1):
    
      for step2 in range(0,int(np.size(label,0)/batch_size)):
             
        batch_x=np.transpose(train[:,range(step2*batch_size,(step2+1)*batch_size)])   
        
        batch_y=label[range(step2*batch_size,(step2+1)*batch_size),:]
       
      
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y,
                                                                 keep_prob: 1.0})
#            loss_fn[count]=loss
#            count=count+1
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
    saver.save(sess,'/pylon5/at560kp/achatto2/Tensorflow/Clustering/fuzzy_40ensemble/save')
    print("Optimization Finished!")
    # Calculate accuracy for 256 MNIST test images
    test_x=np.load('test_40ensemble_4classes_fuzzy.npy')
   
    test_y=np.load('test_labels_40ensemble_4classes_fuzzy.npy')
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: np.transpose(test_x),
                                      Y: test_y,
                                      keep_prob: 1.0})) 


 

    
    pred=sess.run(prediction, feed_dict={X: np.transpose(test_x),keep_prob: 1.0});
    np.savetxt("prediction_4layers.csv", pred, delimiter=",")
        
  #  np.savetxt("test_y_run1.csv", test_y, delimiter=",")
  #  np.savetxt("loss_newdata_200epochs_kernelsize_5x5_8_16_colored_regularization_data.csv", loss_fn, delimiter=",")

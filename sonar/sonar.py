import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns


def read_dataset():
    df = pd.read_csv("sonar.csv", header=None)
    X = df.iloc[:,:-2]
    Y = df.iloc[:,-1]
    Y = pd.get_dummies(Y)  # One Hot Encoding
    return(X,Y,df)


#initialize variables
X, Y, df = read_dataset()
X, Y = shuffle(X, Y, random_state=1)
train_X, test_X, train_y, test_y = train_test_split(X, Y, test_size=.20, random_state=415)
learning_rate = 1
training_epochs = 1501  #Loops
n_features = X.shape[1]    # features in X
n_class = Y.shape[1]    #classes in y
n_hidden_1 = 60
n_hidden_2 = 60
n_hidden_3 = 60
n_hidden_4 = 60
n_hidden_5 = 60


x = tf.placeholder(tf.float32, [None, n_features])
y = tf.placeholder(tf.float32, [None, n_class])



Weights = {
    'W1' : tf.Variable(tf.truncated_normal([n_features, n_hidden_1])),
    'W2' : tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
    'W3' : tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3])),
    'W4' : tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4])),
    'W5' : tf.Variable(tf.truncated_normal([n_hidden_4, n_hidden_5])),
    'out' : tf.Variable(tf.truncated_normal([n_hidden_5, n_class]))
}
biases = {
    'b1' : tf.Variable(tf.truncated_normal([n_hidden_1])),
    'b2' : tf.Variable(tf.truncated_normal([n_hidden_2])),
    'b3' : tf.Variable(tf.truncated_normal([n_hidden_3])),
    'b4' : tf.Variable(tf.truncated_normal([n_hidden_4])),
    'b5' : tf.Variable(tf.truncated_normal([n_hidden_5])),
    'out' : tf.Variable(tf.truncated_normal([n_class]))
}



def forward_propogation(X, Weights, biases):
    Layer1 = tf.nn.relu(tf.add(tf.matmul(X, Weights['W1']), biases['b1']))
    Layer2 = tf.nn.relu(tf.add(tf.matmul(Layer1, Weights['W2']), biases['b2']))
    Layer3 = tf.nn.tanh(tf.add(tf.matmul(Layer2, Weights['W3']), biases['b3']))
    Layer4 = tf.nn.tanh(tf.add(tf.matmul(Layer3, Weights['W4']), biases['b4']))
    Layer5 = tf.nn.tanh(tf.add(tf.matmul(Layer4, Weights['W5']), biases['b5']))
    y_ = tf.nn.sigmoid(tf.add(tf.matmul(Layer5, Weights['out']), biases['out']))
    return y_



init = tf.global_variables_initializer()



#initializing all variables
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)



y_ = forward_propogation(x, Weights, biases)
cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)
test_log=[]
train_log=[]
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        sess.run(optimizer, feed_dict={x:train_X, y:train_y})
        training_cost = sess.run(cost_function, feed_dict={x:train_X, y:train_y})
        test_cost = sess.run(cost_function, feed_dict={x:test_X, y:test_y})
        train_log.append(training_cost)
        test_log.append(test_cost)
        if(epoch%100 == 0):
            print(epoch,"  training cost",training_cost,"  test_cost",test_cost)
    test_pred = sess.run( y_, feed_dict={ x:test_X } )



plt.plot(range(training_epochs), train_log, 'b')
plt.plot(range(training_epochs), test_log, 'r')
plt.legend(("train cost","test cost"))
plt.xlabel("epochs")
plt.ylabel("cost")



test_pred_normal = np.argmax(test_pred, axis=1)
test_y_normal = np.argmax(test_y.as_matrix(), axis=1)


print(confusion_matrix(test_y_normal,test_pred_normal))
from _future_ import print_function

import numpy as numpyimport matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf


from tensorflow.contrib import learn
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn import metrics

boston = learn.datasets.load_dataset('boston')

x, y = boston.data, boston.target

y.resize(y.size, 1) #make y a 506 by 1 tensor

#20% of data is test data, 80% is training
tain_x, test_x, train_y, test_y = cross_validation.train_test_split(x, y, test_xize = 0.2)

#Scale the inputs to have mean 0 and unit standard error
scaler = preprocessing.StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.fit_transform(test_x)

#13 features
numFeatures = train_x.shape[1]

#Dimensions of train_x = (404, 13)
#Dimensions of train_y = (404, 1)
#Dimensions of test_x = (102, 13)
#Dimensions of test_y = (102, 1)

with tf.name_scope("IO"):
	#no height, width of 13
	inputs = tf.placeholder(tf.float32, [None, numFeatures], name = "x")
	outputs = tf.placeholder(tf.float32, [None, 1], name = "yhat")

def neural_network_model(data):
	Layers = [numFeatures, 100, 1]

	hidden_1_layer = {'weights': tf.Variable(tf.random_normal([Layers[0], Layers[1]], mean = 0, stddev = .1, dtype = tf.float32)),
						'biases':tf.Variable(tf.random_normal([Layers[1]], mean = 0, stddev = .1, dtype = tf.float32))}

	output_layer = {'weights': tf.Variable(tf.random_normal([Layers[1], Layers[2]], mean = 0, stddev = .1, dtype = tf.float32)),
						'biases':tf.Variable(tf.random_normal([Layers[2]], mean = 0, stddev = .1, dtype = tf.float32))}

	l1 =  tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.sigmoid(l1)

	output = tf.add(tf.matmul(l1, output_layer['weights']), output_layer['biases'])

	return output

with tf.name_scope("train"):
	learning_rate = 0.01
	y_out = neural_network_model(inputs)

	#cost is the squared error
	cost_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_op)

epoch = 0
last_cost = 0
max_epochs = 50000

print("Beginning Training")
sess = tf.Session()

with sess.as_default():
	#Graphs
	writer = tf.summary.FileWriter("./logs/nn_logs", sess.graph)
	merged = tf.summary.merge_all()

	#initalize the variables
	init = tf.global_variables_initializer()
	sess.run(init)
	#start training until we stop, weither because the max number of epochs
	#is reached, or successive errors are close enough to each other (less than tolerance)

	costs = []
	epochs = []
	while epoch <= max_epochs:
		#Do the training
		sess.run(train_op, feed_dict = {inputs: train_x, outputs: train_y})

		#Update the user every 1000 epochs
		if epoch % 1000 == 0:
			cost = sess.run(cost_op, feed_dict = {inputs:train_x, outputs: train_y})
			costs.append(cost)
			epochs.append(epoch)

			print("Epoch: ", epoch, "\tError:", cost)

			last_cost = cost

		epoch += 1

	print("Test Cost =", sess.run(cost_op, feed_dict={inputs: text_x, oututs: test_y}))

	#compute the predicted output for test_x
	pred_y = sess.run(y_out, feed_dict={inputs: test_x, outputs: test_y})

	for (y, yHat) in list(zip(test_y, pred_y)) [0:10]:
		print("%1.1f\t%1.1f"% (y, yHat))

	r2 = metrics.r2_score(test_y, pred_y)
	print("mean squared error =", metrics.mean_squared_error(test_y, pred_y))
	print("r2 scpre (coef determination) = ", metrics.r2_score(test_y, pred_y))
import os, cv2, sys, numpy as np, tensorflow as tf, warnings, keras
from keras.datasets import cifar10
from skimage import exposure

#disable warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#class labels (used for testing images)
image_classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#define the directory to which I store the model
modelDir = os.path.dirname(os.path.abspath(__file__)) + '/model/Eldin_Neural_Net.ckpt'

#parameters for the neural network
hidden_layers = [128] #[1024]
convolutional_layers = [([32], [2, 2], 2, 5), ([64], [2, 2], 2, 5)] #, (128, 5)
learningRate = 1e-3
momentumRate = 0.9
iterations = 100000
drop_out_rate = 0.0
cifar10_mean = None
sizeBatch = 128
n_classes = 10
img_size = 32
num_channels = 3
img_size_flat = img_size * img_size * num_channels
img_shape_full = (img_size, img_size, num_channels)

#function that first performs a Guassian Blur using a 3x3 kernel
#then performs histogram equalization
def histogram_equalize(img):
    img = cv2.GaussianBlur(img,(3,3),0)
    img_cdf, bin_centers = exposure.cumulative_distribution(img)
    return np.interp(img, bin_centers, img_cdf)

#a function that processes the input test image
#first subtract the mean training image, then
#perform gaussian blurring and histogram equalization
#lastly, flatten the image so that it may be fed as 
#input into the neural network
def pre_process_test(image):
    global cifar10_mean
    
    image = image.astype(np.float32)
    image -= cifar10_mean
    image = histogram_equalize(image)
    image = image.flatten()
    
    return image

#a function that processes the images used during the training process
#first we load the cifar10 dataset. Next we convert the data types from
#uint8 to floats or ints. Next, we compute the mean training image and subtract
#that from both the training and testing datasets. Once this is done, we perform
#Gaussian Blurring and histogram equalization on each image. Lastly, we reshape the images
#and the labels to fit our neural network structure
def pre_process():
	global cifar10_mean

	(xTrain, yTrain), (xTest, yTest) = cifar10.load_data()
	xTrain, yTrain, xTest, yTest = xTrain.astype(np.float32), yTrain.astype(np.int64), xTest.astype(np.float32), yTest.astype(np.int64)

	cifar10_mean = np.mean(xTrain, axis=0)
	xTrain -= cifar10_mean
	xTest -= cifar10_mean

	#xTrain = np.array(list(map(lambda x: histogram_equalize(x), xTrain)))
	#xTest = np.array(list(map(lambda x: histogram_equalize(x), xTest)))

	xTrain /= 255
	xTest /= 255 

	xTrain, xTest = np.reshape(xTrain, (xTrain.shape[0], -1)), np.reshape(xTest, (xTest.shape[0], -1))
	yTrain, yTest = np.squeeze(yTrain), np.squeeze(yTest)

	return (xTrain, yTrain, xTest, yTest)

#This function builds the neural network structure by iteratively adding layers.
#We iterate through the hidden_nodes list containing the number of nodes that should
#be in each layer. For each hidden layer we use the relu activation function and an
#elastic net regularizer (i.e. l1 Norm + l2 Norm). Once this layer is added to the 
#neural network structure, we add a dropout layer that's used in training. The last
#layer (i.e. output layer) uses the softmax function activation function instead of
#the relu activation function 
def build_model(input_layer, hidden_nodes, conv_nodes, num_classes, drop_rate, phase):
	prev_layer = tf.reshape(input_layer, [-1, *img_shape_full])

	for num_filters, pooling_size, stride_len, size_kernel in conv_nodes:
		for filter_size in num_filters:
			prev_layer = tf.layers.conv2d(inputs=prev_layer, filters=filter_size, kernel_size=[size_kernel, size_kernel], padding="same", activation=tf.nn.relu, kernel_initializer=tf.contrib.keras.initializers.he_normal()) #kernel_regularizer=tf.contrib.layers.sum_regularizer(regularizer_list=[tf.contrib.layers.l2_regularizer(scale=1e-5), tf.contrib.layers.l1_regularizer(scale=1e-5)])
			prev_layer = tf.layers.batch_normalization(prev_layer, training=phase) #, center=True, scale=True

		prev_layer = tf.layers.max_pooling2d(inputs=prev_layer, pool_size=pooling_size, strides=stride_len)
		prev_layer = tf.layers.dropout(inputs=prev_layer, rate=drop_rate, training=phase)
	
	prev_layer = tf.contrib.layers.flatten(prev_layer)
	
	for num_nodes in hidden_nodes:
		prev_layer = tf.layers.dense(inputs=prev_layer, units=num_nodes, activation=tf.nn.relu, kernel_initializer=tf.contrib.keras.initializers.he_normal())
		prev_layer = tf.layers.batch_normalization(prev_layer, training=phase) #, center=True, scale=True
		prev_layer = tf.layers.dropout(inputs=prev_layer, rate=drop_rate, training=phase)
	
	output = tf.layers.dense(inputs=prev_layer, units=num_classes, activation=tf.nn.softmax)
	return output

'''prev_layer = tf.reshape(input_layer, [-1, 32, 32, 3])

# Convolutional Layer #1 and Pooling Layer #1
prev_layer = tf.layers.conv2d(inputs=prev_layer, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
prev_layer = tf.layers.max_pooling2d(inputs=prev_layer, pool_size=[2, 2], strides=2)
prev_layer = tf.layers.batch_normalization(prev_layer, center=True, scale=True, training=phase)
prev_layer = tf.layers.dropout(inputs=prev_layer, rate=drop_rate, training=phase)

# Convolutional Layer #1 and Pooling Layer #1
prev_layer = tf.layers.conv2d(inputs=prev_layer, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
prev_layer = tf.layers.max_pooling2d(inputs=prev_layer, pool_size=[2, 2], strides=2)
prev_layer = tf.layers.batch_normalization(prev_layer, center=True, scale=True, training=phase)
prev_layer = tf.layers.dropout(inputs=prev_layer, rate=drop_rate, training=phase)

# Convolutional Layer #1 and Pooling Layer #1
prev_layer = tf.layers.conv2d(inputs=prev_layer, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
prev_layer = tf.layers.max_pooling2d(inputs=prev_layer, pool_size=[2, 2], strides=2)
prev_layer = tf.layers.batch_normalization(prev_layer, center=True, scale=True, training=phase)
prev_layer = tf.layers.dropout(inputs=prev_layer, rate=drop_rate, training=phase)

# Convolutional Layer #1 and Pooling Layer #1
prev_layer = tf.layers.conv2d(inputs=prev_layer, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
prev_layer = tf.layers.max_pooling2d(inputs=prev_layer, pool_size=[2, 2], strides=2)
prev_layer = tf.layers.batch_normalization(prev_layer, center=True, scale=True, training=phase)
prev_layer = tf.layers.dropout(inputs=prev_layer, rate=drop_rate, training=phase)

prev_layer = tf.contrib.layers.flatten(prev_layer)

# Dense Layers
prev_layer = tf.layers.dense(inputs=prev_layer, units=256, activation=tf.nn.relu)
prev_layer = tf.layers.batch_normalization(prev_layer, center=True, scale=True, training=phase)
prev_layer = tf.layers.dropout(inputs=prev_layer, rate=drop_rate, training=phase)

prev_layer = tf.layers.dense(inputs=prev_layer, units=128, activation=tf.nn.relu)
prev_layer = tf.layers.batch_normalization(prev_layer, center=True, scale=True, training=phase)
prev_layer = tf.layers.dropout(inputs=prev_layer, rate=drop_rate, training=phase)

prev_layer = tf.layers.dense(inputs=prev_layer, units=64, activation=tf.nn.relu)
prev_layer = tf.layers.batch_normalization(prev_layer, center=True, scale=True, training=phase)
prev_layer = tf.layers.dropout(inputs=prev_layer, rate=drop_rate, training=phase)

output = tf.layers.dense(inputs=prev_layer, units=num_classes, activation=tf.nn.softmax)
return output'''

# Convolutional Layer #1 and Pooling Layer #1
'''	prev_layer = tf.layers.conv2d(inputs=prev_layer, filters=128, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
prev_layer = tf.layers.max_pooling2d(inputs=prev_layer, pool_size=[2, 2], strides=2)
prev_layer = tf.layers.batch_normalization(prev_layer, center=True, scale=True, training=phase)
prev_layer = tf.layers.dropout(inputs=prev_layer, rate=drop_rate, training=phase)'''

'''	# Convolutional Layer #2 and Pooling Layer #2
prev_layer = tf.layers.conv2d(inputs=prev_layer, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
prev_layer = tf.layers.max_pooling2d(inputs=prev_layer, pool_size=[2, 2], strides=2)

# Convolutional Layer #3 and Pooling Layer #3
prev_layer = tf.layers.conv2d( inputs=prev_layer, filters=128, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
prev_layer = tf.layers.max_pooling2d(inputs=prev_layer, pool_size=[2, 2], strides=2)'''

'''
prev_layer = input_layer
for nodes in hidden_nodes:
prev_layer = tf.layers.dense(inputs=prev_layer, units=nodes, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.sum_regularizer(regularizer_list=[tf.contrib.layers.l2_regularizer(scale=1e-5), tf.contrib.layers.l1_regularizer(scale=1e-5)]))
prev_layer = tf.layers.dropout(inputs=prev_layer, rate=drop_rate, training=True)

output = tf.layers.dense(inputs=prev_layer, units=num_classes, activation=tf.nn.softmax)
return output
'''

#this function trains the neural network and saves the model to the specified directory
#We begin by importing the cifar10 dataset and processing the images by calling
#pre_process(). The description for this function may be found above. Next, we create
#the placeholders for the input layer, the output layer, and the dropout rate. Now we
#pass these as parameters to the model and create the model using the build_model()
#function described above. Next, we compute the loss of our neural network and try
#to minimize it with the AdamOptimizer(). Next we get the model's predictions and 
#use reduce_mean(), defining one of the inputs as our predictions. Once all this has
#been setup we then begin the tensorflow session and initiate all the variables.
#Now we iterate over the specified number of iterations. Within each iteration, we
#randomly choose batches of the specified size (in this case we chose 128). Once the
#batches have been retrieved, we run the training procedure on the model then,
#every 1000 iterations, we check the accuracy of our model against the test set.
#Once we are finished iterating (i.e. done training), we save the model and save the mean
#cifar10 image so that we may reload these during testing. 
def train():
    global cifar10_mean

    (xTrain, yTrain, xTest, yTest) = pre_process()
    image_size = len(xTrain[0].flatten())
    
    input_ = tf.placeholder(tf.float32, [None, img_size_flat])
    output_ = tf.placeholder(tf.int64, [None])
    drop_rate = tf.placeholder(tf.float32)
    phase = tf.placeholder(tf.bool, name='phase')

    model = build_model(input_, hidden_layers, convolutional_layers, n_classes, drop_rate, phase)
    loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=output_, logits=model))
    train_step = tf.train.AdamOptimizer(learningRate).minimize(loss) #tf.train.AdadeltaOptimizer(learning_rate=learningRate).minimize(loss) #tf.train.MomentumOptimizer(learning_rate=learningRate, momentum=momentumRate, use_nesterov=True).minimize(loss) #

    correct_prediction = tf.equal(tf.argmax(model, 1), output_)
    test_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print("Epoch\t\tAccuracy\n------------------------")

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        for iteration in range(iterations):
            #s = np.arange(xTrain.shape[0])
            batch_idx = np.random.choice(img_shape_full, size=sizeBatch)
            batch_x, batch_y = xTrain[batch_idx, :], yTrain[batch_idx,]

            train_step.run(feed_dict={input_: batch_x, output_: batch_y, drop_rate: drop_out_rate, phase: True})

            if not(iteration%10):
                accuracy = test_step.eval(feed_dict={input_: xTest, output_: yTest, drop_rate: 0.0, phase: False})
                print("{0}\t\t{1}%".format(str(iteration), str(round(accuracy*100, 4))))
                #print("Iteration: {0}; Accuracy: {1}".format(str(iteration), str(accuracy)))

        saver = tf.train.Saver()
        s_path = saver.save(session, modelDir)
        cv2.imwrite('cifar10_mean.png', cifar10_mean)

#This function is used to load the saved model and run it on an input test image
#first we create input and dropout rate placeholder then run the build_model() function.
#Now we begin the tensorflow session. Within this session, we restore the model from the
#saved file, then we evaluate our model against the input image. Once this is done, we 
#simply output the predicted class
def test(image):
    image_size = len(image)
    input_ = tf.placeholder(tf.float32, [None, image_size])
    drop_rate = tf.placeholder(tf.float32)
    model = build_model(input_, hidden_layers, n_classes, drop_rate)

    with tf.Session() as session:
        saver = tf.train.Saver()
        saver.restore(session, modelDir)

        predict_step = tf.argmax(model, 1)
        index = predict_step.eval(feed_dict={input_: [image], drop_rate: 0.0})[0]
        print('Image Is Predicted As: {0}'.format(image_classes[index]))

#this is used to get inputs from the command line (if there are any).
#If there are no command line arguments, then we simply train the model
#if the command line argument is 'train', then we train the model
#if the command line argument is 'test', then we load the cifar10 mean image
#process the input image, and pass that image to the test function
if (__name__ == '__main__') and (len(sys.argv) > 1):
    if 'train' == sys.argv[1]:
        train()

    elif 'test' == sys.argv[1]:
        cifar10_mean = cv2.imread(os.path.dirname(os.path.abspath(__file__)) + '/cifar10_mean.png')
        test(pre_process_test(cv2.imread(os.path.dirname(os.path.abspath(__file__)) + sys.argv[2])))

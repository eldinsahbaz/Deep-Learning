import os, cv2, sys, numpy as np, tensorflow as tf, warnings, math, matplotlib
from tensorflow.python.keras.layers import Conv2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.regularizers import l1, l2, l1_l2
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import Adam, SGD
from matplotlib import pyplot as plt
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.datasets import cifar10
from skimage import exposure
import keras

# disable warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# class labels (used for testing images)
image_classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# define the directory to which I store the model
modelDir = os.path.dirname(os.path.abspath(__file__)) + '/model/'
modelFileName = 'Eldin_Sahbaz_CNN.ckpt'
# parameters for the neural network
hidden_layers = [256, 128, 64]  # [120, 84] #[500, 500, 500]
convolutional_layers = [32, 32, 64, 64]  # [36, 56] #[6, 16]
learningRate = 1e-3
iterations = 100
drop_out_rate = 0.2
cifar10_mean = None
n_classes = 10
img_size = 32
img_size_flat = img_size * img_size * 3
img_shape_full = (img_size, img_size, 3)
drop_rate = 0.2


#This function uses matplotlib to get the output of the first convolutional
#layer in the CNN for a given input image. The my_plot function makes a
#square grid corresponding to the number of filters in the first convolutional
# layer. It then plots the output of each of the filters onto the grid. This
# then saves the grid to CONV_rslt.png and shows the plots on the screen.
def plot_conv_output(model, image):
    def my_plot(values):
        # Retrieve the number of filters in the convolutional layer.
        num_filters = values.shape[3]

        # Set the number of grids needed for this convolutional layer.
        num_grids = math.ceil(math.sqrt(num_filters))

        # Create figure with a grid of sub-plots.
        fig, axes = plt.subplots(num_grids, num_grids)

        # Plot the outputs for each filter.
        for i, ax in enumerate(axes.flat):
            if i < num_filters:
                # Get the ith filter.
                img = values[0, :, :, i]

                # Plot the filter's values onto the corresponding grid.
                ax.imshow(img, interpolation='nearest', cmap='binary')

            # Remove ticks from the plot.
            ax.set_xticks([])
            ax.set_yticks([])

        # Save the figure to a PNG file
        # show the figure
        plt.savefig('CONV_rslt.png')
        plt.show()

    # Use the K function to get the output of the first convolutional layer
    output_conv = K.function(inputs=[model.layers[0].input], outputs=[model.get_layer('layer_conv0').output])

    # Use the function above and pass it the test image
    layer_output = output_conv([[image]])[0]

    # Use the output from the convolutional layer and plot the filters
    my_plot(values=layer_output)


# a function that processes the input test image
# first subtract the mean training image, then
# normalize the pixel values to a range [0, 1]
def pre_process_test(image):
    global cifar10_mean

    image = image.astype('float32')
    image -= cifar10_mean
    image /= 255
    return image


# a function that processes the images used during the training process
# first we load the cifar10 dataset. Next we convert the labels to categorical
# vectors (i.e. one-hot encoding). Next, we convert the training and testing samples
# from uint8 to float32. Next, we compute the mean training image and subtract
# that from both the training and testing datasets. Lastly, we normalize the pixel values
# to the range [0, 1].
def pre_process():
    global cifar10_mean
    global img_shape_full

    (xTrain, yTrain), (xTest, yTest) = cifar10.load_data()
    img_shape_full = (lambda x: (x[1], x[2], x[3]))(xTrain.shape)

    yTrain, yTest = keras.utils.to_categorical(yTrain, n_classes), keras.utils.to_categorical(yTest, n_classes)
    xTrain, xTest = xTrain.astype('float32'), xTest.astype('float32')

    cifar10_mean = np.mean(xTrain, axis=0)
    xTrain -= cifar10_mean
    xTest -= cifar10_mean

    xTrain /= 255
    xTest /= 255

    return (xTrain, yTrain, xTest, yTest)


# This function builds the neural network structure by iteratively adding layers.
# We iterate through the convolutional_nodes list. The length of this list correpsonds
# to the number of convolutions that will be added and each element in the list corrponds
# to the number of filters in that convolutional layer. Along with the convolutional layers
# we also add maxpooling, batch normalization, and drop out layers. Once these layers are added
# we flatten the output from the convolutions and feed them into fully connected layers/
# For this we also iterate through a list (hidden_nodes) whose length correpsonds to the
# number of fully connected layers to be added and whose elements specify the number of nodes
# in each fully connected layer. We also use batch normalization and drop out with the fully
# connected layers as well. The last layer is the output layer. which has 10 nodes (for the 10 classes).
# the output of this last layer is passed through a softmax. This network is trained with SGD+Momentum to
# optimize the cetgorical cross-entropy. Thie function then returns the constructed model.
def build_model(hidden_nodes, convolutional_nodes, num_classes, drop_rate, flat_size, full_size):
    model = Sequential()
    model.add(InputLayer(input_shape=full_size))

    for y, x in enumerate(convolutional_nodes):
        model.add(Conv2D(kernel_size=5, strides=1, filters=x, padding='same', activation='relu',
                         kernel_initializer='he_normal', name='layer_conv{0}'.format(str(y))))
        model.add(MaxPooling2D(pool_size=2, strides=2))
        model.add(BatchNormalization())
        model.add(Dropout(drop_rate, input_shape=(
            (lambda z: (z[1], z[2], z[3]))(model.get_layer('layer_conv{0}'.format(str(y))).output_shape))))

    model.add(Flatten())

    for y, x in enumerate(hidden_nodes):
        model.add(Dense(x, activation='relu', kernel_initializer='he_normal', name='layer_dense{0}'.format(str(y))))
        model.add(BatchNormalization())
        model.add(Dropout(drop_rate, input_shape=(
            (lambda z: (z[1],))(model.get_layer('layer_dense{0}'.format(str(y))).output_shape))))

    model.add(Dense(num_classes, activation='softmax'))
    optimizer = SGD(lr=.1, momentum=0.9, nesterov=True)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# this function trains the neural network and saves the model to the specified directory
# We begin by importing the cifar10 dataset and processing the images by calling
# pre_process(). The description for this function may be found above. Next, we create
# the model by calling the build_model() function described above. Once this is done,
# we create a tensorflow session and initialize all the variables. Now we use the model
# we previously built and train it using the training data set - with a batch size fo 128
# and 100 epochs. For each batch, we evaluate the model against the testing data to track
# the accuracy. After this is complete, we check to see if our model directory already exists
# if it doesn't, we create it. Now that we know it exists, we save the model structure and weights
# to that directory so that we may load it later for testing. We also save the cifar10 mean training
# image.
def train():
    global cifar10_mean
    global img_shape_full

    (xTrain, yTrain, xTest, yTest) = pre_process()
    image_size = len(xTrain[0].flatten())
    model = build_model(hidden_layers, convolutional_layers, n_classes, drop_rate, image_size, img_shape_full)

    with tf.Session() as session:
        #session.run(tf.global_variables_initializer())
        keras.backend.get_session().run(tf.global_variables_initializer())
        model.fit(x=xTrain, y=yTrain, epochs=iterations, batch_size=128, validation_data=(xTest, yTest), workers=10,
                  shuffle=True)
        result = model.evaluate(x=xTest, y=yTest)

        for name, value in zip(model.metrics_names, result):
            print(name, value)

        try:
            os.stat(modelDir)
        except:
            os.mkdir(modelDir)

        model_json = model.to_json()
        with open(modelDir + modelFileName, "w") as json_file:
            json_file.write(model_json)
            model.save_weights(modelDir + "weights.h5")

        cv2.imwrite(modelDir + 'cifar10_mean.png', cifar10_mean)
        # print(model.summary())


# This function is used to load the saved model (and its weights) and run it on an input test image.
# Once we load the model, we pass the test image to the model and get the CNN's classification.
# We print the classification and save the output from the first convolutional layer by using the
# plot_conv_output(.) function described above.
def test(image):
    # load json and create model
    with tf.Session() as session:
        with open(modelDir + modelFileName, 'r') as json_file:
            model = json_file.read()

        model = model_from_json(model)
        model.load_weights(modelDir + "weights.h5")
        print('Image Is Predicted As: {0}'.format(image_classes[np.argmax(model.predict(np.array([image])))]))
        plot_conv_output(model, image)


# this is used to get inputs from the command line (if there are any).
# If there are no command line arguments, then we do nothing
# if the command line argument is 'train', then we train the model
# if the command line argument is 'test', then we load the cifar10
# mean image, trained model, and test image and process the input
# image, and pass that image to the test function
if (__name__ == '__main__') and (len(sys.argv) > 1):
    if 'train' == sys.argv[1]:
        train()

    elif 'test' == sys.argv[1]:
        cifar10_mean = cv2.imread(modelDir + 'cifar10_mean.png')
        test(pre_process_test(cv2.imread(os.path.dirname(os.path.abspath(__file__)) + '\\' + sys.argv[2])))

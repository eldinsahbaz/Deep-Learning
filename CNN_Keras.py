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

#disable warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#class labels (used for testing images)
image_classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#define the directory to which I store the model
modelDir = os.path.dirname(os.path.abspath(__file__)) + '/model/'
modelFileName = 'Eldin_Sahbaz_CNN.ckpt'
#parameters for the neural network
hidden_layers = [256, 128, 64] #[120, 84] #[500, 500, 500]
convolutional_layers = [32, 32, 64, 64] #[36, 56] #[6, 16]
learningRate =  1e-3
iterations = 50
drop_out_rate = 0.2
cifar10_mean = None
n_classes = 10
img_size = 32
img_size_flat = img_size * img_size * 3
img_shape_full = (img_size, img_size, 3)
drop_rate = 0.2

def plot_conv_output(model, image):
    def my_plot(values):
        # Number of filters used in the conv. layer.
        num_filters = values.shape[3]

        # Number of grids to plot.
        # Rounded-up, square-root of the number of filters.
        num_grids = math.ceil(math.sqrt(num_filters))
        
        # Create figure with a grid of sub-plots.
        fig, axes = plt.subplots(num_grids, num_grids)

        # Plot the output images of all the filters.
        for i, ax in enumerate(axes.flat):
            # Only plot the images for valid filters.
            if i<num_filters:
                # Get the output image of using the i'th filter.
                img = values[0, :, :, i]

                # Plot image.
                ax.imshow(img, interpolation='nearest', cmap='binary')
            
            # Remove ticks from the plot.
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Ensure the plot is shown correctly with multiple plots
        # in a single Notebook cell.
        plt.savefig('CONV_rslt.png')
        plt.show()

    output_conv = K.function(inputs=[model.layers[0].input], outputs=[model.get_layer('layer_conv0').output])
    layer_output = output_conv([[image]])[0]
    my_plot(values=layer_output)

#a function that processes the input test image
#first subtract the mean training image, then
#perform gaussian blurring and histogram equalization
#lastly, flatten the image so that it may be fed as 
#input into the neural network
def pre_process_test(image):
    global cifar10_mean
    
    image = image.astype('float32')
    image -= cifar10_mean
    image /= 255    
    return image

#a function that processes the images used during the training process
#first we load the cifar10 dataset. Next we convert the data types from
#uint8 to floats or ints. Next, we compute the mean training image and subtract
#that from both the training and testing datasets. Once this is done, we perform
#Gaussian Blurring and histogram equalization on each image. Lastly, we reshape the images
#and the labels to fit our neural network structure
def pre_process():
    global cifar10_mean
    global img_shape_full

    (xTrain, yTrain), (xTest, yTest) = cifar10.load_data()
    img_shape_full =  (lambda x: (x[1], x[2], x[3]))(xTrain.shape)

    yTrain, yTest = keras.utils.to_categorical(yTrain, n_classes), keras.utils.to_categorical(yTest, n_classes)
    xTrain, xTest = xTrain.astype('float32'), xTest.astype('float32')
    
    cifar10_mean = np.mean(xTrain, axis=0)
    xTrain -= cifar10_mean
    xTest -= cifar10_mean
    
    xTrain /= 255
    xTest /= 255

    return (xTrain, yTrain, xTest, yTest)

#This function builds the neural network structure by iteratively adding layers.
#We iterate through the hidden_nodes list containing the number of nodes that should
#be in each layer. For each hidden layer we use the relu activation function and an
#elastic net regularizer (i.e. l1 Norm + l2 Norm). Once this layer is added to the 
#neural network structure, we add a dropout layer that's used in training. The last
#layer (i.e. output layer) uses the softmax function activation function instead of
#the relu activation function 
def build_model(hidden_nodes, convolutional_nodes, num_classes, drop_rate, flat_size, full_size):
    model = Sequential()
    model.add(InputLayer(input_shape=full_size))

    for y, x in enumerate(convolutional_nodes):    
        model.add(Conv2D(kernel_size=5, strides=1, filters=x, padding='same', activation='relu', kernel_initializer='he_normal', name='layer_conv{0}'.format(str(y))))
        model.add(MaxPooling2D(pool_size=2, strides=2))
        model.add(BatchNormalization())
        model.add(Dropout(drop_rate, input_shape=((lambda z: (z[1], z[2], z[3]))(model.get_layer('layer_conv{0}'.format(str(y))).output_shape))))

    model.add(Flatten())

    for y, x in enumerate(hidden_nodes):
        model.add(Dense(x, activation='relu', kernel_initializer='he_normal', name='layer_dense{0}'.format(str(y))))
        model.add(BatchNormalization())
        model.add(Dropout(drop_rate, input_shape=((lambda z: (z[1],))(model.get_layer('layer_dense{0}'.format(str(y))).output_shape))))

    model.add(Dense(num_classes, activation='softmax'))
    optimizer = SGD(lr=.1, momentum=0.9, nesterov=True)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

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
    global img_shape_full

    (xTrain, yTrain, xTest, yTest) = pre_process()
    image_size = len(xTrain[0].flatten())
    model = build_model(hidden_layers, convolutional_layers, n_classes, drop_rate, image_size, img_shape_full)    

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        model.fit(x=xTrain, y=yTrain, epochs=iterations, batch_size=128,  validation_data=(xTest, yTest), workers=10, shuffle=True)
        result = model.evaluate(x=xTest, y=yTest)
        
        for name, value in zip(model.metrics_names, result):
            print(name, value)

        try:
            os.stat(modelDir)
        except:
            os.mkdir(modelDir)

        model_json = model.to_json()
        with open(modelDir+modelFileName, "w") as json_file:
            json_file.write(model_json)
            model.save_weights(modelDir+"weights.h5")


        cv2.imwrite(modelDir+'cifar10_mean.png', cifar10_mean)
        #print(model.summary())

#This function is used to load the saved model and run it on an input test image
#first we create input and dropout rate placeholder then run the build_model() function.
#Now we begin the tensorflow session. Within this session, we restore the model from the
#saved file, then we evaluate our model against the input image. Once this is done, we 
#simply output the predicted class
def test(image):
    # load json and create model
    with tf.Session() as session:
        with open(modelDir+modelFileName, 'r') as json_file:
            model = json_file.read()
        
        model = model_from_json(model)
        model.load_weights(modelDir+"weights.h5")
        print('Image Is Predicted As: {0}'.format(image_classes[np.argmax(model.predict(np.array([image])))]))
        plot_conv_output(model, image)

#this is used to get inputs from the command line (if there are any).
#If there are no command line arguments, then we simply train the model
#if the command line argument is 'train', then we train the model
#if the command line argument is 'test', then we load the cifar10 mean image
#process the input image, and pass that image to the test function
if (__name__ == '__main__') and (len(sys.argv) > 1):
    if 'train' == sys.argv[1]:
        train()

    elif 'test' == sys.argv[1]:
        cifar10_mean = cv2.imread(modelDir+'cifar10_mean.png')
        test(pre_process_test(cv2.imread(os.path.dirname(os.path.abspath(__file__)) + sys.argv[2])))

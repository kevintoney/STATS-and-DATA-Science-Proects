
# import dependencies
# source (copied most from: https://www.kaggle.com/twhitehurst3/stanford-dogs-keras-vgg16)
import matplotlib.pyplot as plt
import numpy as np
import os,shutil,math
from PIL import Image
import random as rn
from tqdm import tqdm
from IPython.display import SVG
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# from keras.models import *
# from keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import *
from tensorflow.keras.callbacks import *
# from keras.preprocessing.image import ImageDataGenerator
# from keras.applications.vgg16 import VGG16,preprocess_input
# from keras.utils.vis_utils import model_to_dot
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.keras.utils import plot_model
from tensorflow.keras.backend import get_session
import shap

# helpful functions
def label_assignment(img,label):
    return label


def training_data(label, data_dir):
    for img in tqdm(os.listdir(data_dir)):
        label = label_assignment(img, label)
        path = os.path.join(data_dir, img)
        img = Image.open(path)
        img = img.resize((imgsize, imgsize))

        X.append(np.array(img))
        Z.append(str(label))


def show_final_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(15,5))
    ax[0].set_title('loss')
    ax[0].plot(history.epoch, history.history["loss"], label="Train loss")
    ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax[1].set_title('acc')
    ax[1].plot(history.epoch, history.history["acc"], label="Train acc")
    ax[1].plot(history.epoch, history.history["val_acc"], label="Validation acc")
    ax[0].legend()
    ax[1].legend()


# get the training data directories for a few dogs
chihuahua_dir = "Images/n02085620-Chihuahua"
japanese_spaniel_dir = 'Images/n02085782-Japanese_spaniel'
maltese_dir = 'Images/n02085936-Maltese_dog'
pekinese_dir = 'Images/n02086079-Pekinese'
shitzu_dir = 'Images/n02086240-Shih-Tzu'
blenheim_spaniel_dir = 'Images/n02086646-Blenheim_spaniel'
papillon_dir = 'Images/n02086910-papillon'
toy_terrier_dir = 'Images/n02087046-toy_terrier'
afghan_hound_dir = 'Images/n02088094-Afghan_hound'
basset_dir = 'Images/n02088238-basset'


# prepare matrices
X = []
Z = []
imgsize = 150



# read in training data
training_data('chihuahua',chihuahua_dir)
training_data('japanese_spaniel',japanese_spaniel_dir)
training_data('maltese',maltese_dir)
training_data('pekinese',pekinese_dir)
training_data('shitzu',shitzu_dir)
training_data('blenheim_spaniel',blenheim_spaniel_dir)
training_data('papillon',papillon_dir)
training_data('toy_terrier',toy_terrier_dir)
training_data('afghan_hound',afghan_hound_dir)
training_data('basset',basset_dir)

# prepare the matrices more
label_encoder= LabelEncoder()
Y = label_encoder.fit_transform(Z)
Y = to_categorical(Y,10)
X = np.array(X)
X=X/255

# show some random images
fig, ax = plt.subplots(5, 2) # 5 by 2 plot setup
fig.set_size_inches(15, 15) # the figure is 15 inches by 15 inches
for i in range(5):
    for j in range(2):
        l = rn.randint(0, len(Z)) # get a random integer
        ax[i, j].imshow(X[l]) # show the image in the specific position in the plot array
        ax[i, j].set_title('Dog: ' + Z[l]) # set a title for each image

plt.tight_layout()


############
#next steps


#train-test split
x_train,x_test,y_train,y_test = train_test_split(X, Y, test_size=0.3, random_state=15)

#image generator
augs_gen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=10,
        zoom_range = 0.1,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=False)

augs_gen.fit(x_train)

#CNN model using the VGG16 framework
base_model = VGG16(include_top=False,
                   input_shape=(imgsize, imgsize, 3),
                   weights='imagenet')

for layer in base_model.layers:
    layer.trainable = False

for layer in base_model.layers:
    print(layer, layer.trainable)

# set up the CNN model with a VGG16 model first, a global average pooling layer,
# a dropout layer and then a dense softmax activation layer.
model = models.Sequential()
model.add(base_model)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))
model.summary()

# SVG(model_to_dot(model).create(prog='dot', format='svg'))
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


#train the model
checkpoint = ModelCheckpoint(
    './base.model',
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min',
    save_weights_only=False,
    period=1
)
earlystop = EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,
    patience=30,
    verbose=1,
    mode='auto'
)
tensorboard = TensorBoard(
    log_dir = './logs',
    histogram_freq=0,
    batch_size=16,
    write_graph=True,
    write_grads=True,
    write_images=False,
)

csvlogger = CSVLogger(
    filename= "training_csv.log",
    separator = ",",
    append = False
)

reduce = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=3,
    verbose=1,
    mode='auto'
)

callbacks = [checkpoint,tensorboard,csvlogger,reduce]

#-----------Optimizers-----------#
opt = SGD(lr=1e-4,momentum=0.99)
opt1 = Adam(lr=1e-2)
#----------Compile---------------#
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
#-----------Training------------#
history = model.fit_generator(
    augs_gen.flow(x_train,y_train,batch_size=16),
    validation_data  = (x_test,y_test),
    validation_steps = 500,
    steps_per_epoch  = 500,
    epochs = 15,
    verbose = 1,
    callbacks=callbacks
)


#Validate


#test accuracy
show_final_history(history)
model.load_weights('./base.model')
model_score = model.evaluate(x_test, y_test)
print("Model Test Loss:", model_score[0])
print("Model Test Accuracy:", model_score[1])

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save("model.h5")
print("Weights Saved")




##########
#SHAP with the Deep Explainer


# select a set of background examples to take an expectation over
background = x_train[np.random.choice(x_train.shape[0], 100, replace=False)]

# explain predictions of the model on four images
e = shap.DeepExplainer(model, background)
# ...or pass tensors directly
# e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
shap_values = e.shap_values(x_test[1:5])

# plot the feature attributions
shap.image_plot(shap_values, -x_test[1:5], show=False)
plt.savefig('shapValues.png')
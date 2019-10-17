
# import dependencies
# source (copied most from: https://www.kaggle.com/twhitehurst3/stanford-dogs-keras-vgg16)
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import random as rn
from tqdm import tqdm
from IPython.display import SVG
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import *
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121,preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils.vis_utils import plot_model
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard,CSVLogger,ReduceLROnPlateau,LearningRateScheduler
from keras.optimizers import Adam,SGD
from keras import backend as K
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
imgsize = 224



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
base_model = DenseNet121(input_tensor=inp,
                       weights="./DenseNet-BC-121-32-no-top.h5",
                       include_top=False)

for layer in base_model.layers:
    layer.trainable = False

for layer in base_model.layers:
    print(layer, layer.trainable)

# set up the CNN model with a VGG16 model first, a global average pooling layer,
# a dropout layer and then a dense softmax activation layer.
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(1024, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.summary()

SVG(model_to_dot(model).create(prog='dot', format='svg'))
plot_model(model, to_file='model_plotG.png', show_shapes=True, show_layer_names=True)


#train the model
checkpoint = ModelCheckpoint(
    './base.modelGD',
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
    log_dir = './logsGD',
    histogram_freq=0,
    batch_size=16,
    write_graph=True,
    write_grads=True,
    write_images=False,
)

csvlogger = CSVLogger(
    filename= "training_csvGD.log",
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


# Validate


# test accuracy
# show_final_history(history)
model.load_weights('./base.modelGD')
model_score = model.evaluate(x_test, y_test)
print("Model Test Loss:", model_score[0])
print("Model Test Accuracy:", model_score[1])

model_json = model.to_json()
with open("modelGD.json", "w") as json_file:
    json_file.write(model_json)

model.save("GradDensemodel.h5")
print("Weights Saved")


##########
# SHAP with the GradientExplainer


# explain how the input to the 2nd layer of the model explains the top two classes
def map2layer(x, layer, nodeIndex):
    feed_dict = dict(zip([model.layers[0].input], [preprocess_input(x.copy())]))
    return K.get_session().run(model.layers[layer].get_input_at(nodeIndex), feed_dict)


e = shap.GradientExplainer((model.layers[2].get_input_at(3), model.layers[-1].output), map2layer(preprocess_input(x_train.copy()), 2, 3))
shap_values,indexes = e.shap_values(map2layer(x_test[1:3], 2, 3), ranked_outputs=2)


# plot the explanations
shap.image_plot(shap_values, x_test[1:3])
plt.savefig('./shapValuesGradiDense.png')
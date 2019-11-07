
# import dependencies
# source (copied the preprocessing and model evaluation code
# from: https://www.kaggle.com/twhitehurst3/stanford-dogs-keras-vgg16)
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
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.utils.vis_utils import model_to_dot
from keras.utils.vis_utils import plot_model
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard,CSVLogger,ReduceLROnPlateau,LearningRateScheduler
from keras.optimizers import Adam,SGD
from keras import backend as K
from skimage.segmentation import slic
from matplotlib.colors import LinearSegmentedColormap
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
    ax[1].set_title('accuracy')
    ax[1].plot(history.epoch, history.history["accuracy"], label="Train acc")
    ax[1].plot(history.epoch, history.history["val_accuracy"], label="Validation acc")
    ax[0].legend()
    ax[1].legend()

# convert numpy array to RGBA with some transparency (ALPHA EQUALS 85, not 255)
def array2PIL(arr, size):
    mode = 'RGBA'
    arr = arr.reshape(arr.shape[0]*arr.shape[1], arr.shape[2])
    if len(arr[0]) == 3:
        arr = np.c_[arr, 85*np.ones((len(arr),1), np.uint8)]
    return Image.frombuffer(mode, size, arr.tostring(), 'raw', mode, 0, 1)


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
imgsize = 224 # maybe need to do 224 here



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
classNames = np.unique(Z)
label_encoder= LabelEncoder()
Y = label_encoder.fit_transform(Z)
Y = to_categorical(Y,10)
X = np.array(X)
X=X/255


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
base_model = InceptionV3(include_top=False, classes=10)

for layer in base_model.layers:
    layer.trainable = False

for layer in base_model.layers:
    print(layer, layer.trainable)

# set up the CNN model with a squeezenet model first, a global average pooling layer,
# a dropout layer and then a dense softmax activation layer.
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.summary()

SVG(model_to_dot(model).create(prog='dot', format='svg'))
plot_model(model, to_file='model_plotKernInc255.png', show_shapes=True, show_layer_names=True)


#train the model
checkpoint = ModelCheckpoint(
    './base.modelKInc255',
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
    log_dir = './logsKInc255',
    histogram_freq=0,
    batch_size=16,
    write_graph=True,
    write_grads=True,
    write_images=False,
)

csvlogger = CSVLogger(
    filename= "training_csvKInc255.log",
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
model.load_weights('./base.modelKInc255')
model_score = model.evaluate(x_test, y_test)
print("Model Test Loss:", model_score[0])
print("Model Test Accuracy:", model_score[1])


##########
# SHAP with the KernalExplainer

#get an example image to explain
img = x_test[3]
img_orig = img

# segment the image so we don't have to explain every pixel
segments_slic = slic(img, n_segments=50, compactness=10, sigma=3)


# define a function that depends on a binary mask representing if an image region is hidden
def mask_image(zs, segmentation, image, background=None):
    if background is None:
        background = image.mean((0,1))
    out = np.zeros((zs.shape[0], image.shape[0], image.shape[1], image.shape[2]))
    print(zs.shape[0])
    for i in range(zs.shape[0]):
        out[i,:,:,:] = image
        for j in range(zs.shape[1]):
            if zs[i,j] == 0:
                out[i][segmentation == j,:] = background
    return out


def f(z):
    return model.predict(preprocess_input(mask_image(z, segments_slic, img_orig, 255)))
# I changed the background to 255. 


explainer = shap.KernelExplainer(f, np.zeros((1,50)))

shap_values = explainer.shap_values(np.ones((1,50)), nsamples=1000)
# beforehand, I tried to do 10 samples. I think I will try 1000 
# to see if I can get my code to run

preds = model.predict(preprocess_input(np.expand_dims(img_orig.copy(), axis=0)))
top_preds = np.argsort(-preds)
print(top_preds[0])
# plot the explanations
# make a color map

colors = []
for l in np.linspace(1,0,100):
    colors.append((245/255,39/255,87/255,l))

for l in np.linspace(0,1,100):
    colors.append((24/255,196/255,93/255,l))


cm = LinearSegmentedColormap.from_list("shap", colors)


def fill_segmentation(values, segmentation):
    out = np.zeros(segmentation.shape)
    for i in range(len(values)):
        out[segmentation == i] = values[i]
    return out


# plot our explanations

img=img*255
img = img.astype('uint8')
# convert np-array to RGBA
imgRGBA = array2PIL(img, [224, 224])

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12,4))
inds = top_preds[0]
print(inds[0])
print(inds[1])
print(shap_values[inds[0]][0])
print(shap_values[inds[1]][0])
print(shap_values[inds[2]][0])
print(shap_values[inds[3]][0])
# print(shap_values[inds[0]][1])
# print(shap_values[inds[0]][2])
axes[0].imshow(img)
axes[0].axis('off')
max_val = np.max([np.max(np.abs(shap_values[i][:,:-1])) for i in range(len(shap_values))])

for i in range(3):
    m = fill_segmentation(shap_values[inds[i]][0], segments_slic)
    axes[i+1].set_title(classNames[inds[i]])
    axes[i+1].imshow(imgRGBA)
    im = axes[i+1].imshow(m, cmap=cm, vmin=-max_val, vmax=max_val)
    axes[i+1].axis('off')

cb = fig.colorbar(im, ax=axes.ravel().tolist(), label="SHAP value", orientation="horizontal", aspect=60)
cb.outline.set_visible(False)

# save the figure as a file
plt.savefig('./shapValuesKernalIncep255.png')
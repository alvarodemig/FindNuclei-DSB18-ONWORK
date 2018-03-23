
from skimage.transform import resize
import pandas as pd
import cv2, json, os
from skimage.color import rgb2grey
from sklearn.preprocessing import MinMaxScaler
from skimage.exposure import rescale_intensity
from skimage.transform import resize
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.models import Input, Model
from keras.layers import UpSampling2D, Dropout, BatchNormalization
from keras.layers import Conv2D, concatenate, MaxPooling2D
from keras.callbacks import ModelCheckpoint, Callback
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from skimage import io
from skimage.morphology import label
from sklearn.model_selection import train_test_split
import keras
import imgaug as ia
from imgaug import augmenters as iaa
from random import uniform

basepath = os.getcwd()

#List of names of the train and test images
trainList = os.listdir(os.path.join(basepath, 'stage1_train'))
testList = os.listdir(os.path.join(basepath, 'stage1_test'))
trainPics = len(trainList)
testPics = len(testList)

#Grey scale images and masks join
counter = 0
for f in trainList:
    img = io.imread(os.path.join(basepath, 'stage1_train', f, f + '.png'))
    img = rescale_intensity(rgb2grey(img))
    io.imsave(os.path.join(basepath, 'stage1_train', f, f + '_GREY.png', img))
    
    masks_path = os.listdir(os.path.join(basepath, 'stage1_train', f, 'masks'))
    masks = os.listdir(masks_path)
    img_mask = np.zeros(img.shape)
    for m in masks:
        mask = io.imread(os.path.join(masks_path, m))
        img_mask = np.maximum(img_mask, mask[:,:])
    img_mask /= 255
    io.imsave(os.path.join(os.getcwd(), 'stage1_train', f, f + '_MASK.png', img_mask))
    counter +=1
    print(counter, '/', trainPics, 'COMPLETED')

counter = 0
for f in testList:
    img = io.imread(os.path.join(basepath, 'stage1_train', f, f + '.png'))
    img = rescale_intensity(rgb2grey(img))
    io.imsave(os.path.join(basepath, 'stage1_train', f, f + '_GREY.png', img))
    counter +=1
    print(counter, '/', trainPics, 'COMPLETED')


#Image crop function and reshape generator
def imgCrops(img_name, img_path, img_folder, final_sq_shape, mask_path = None, mask_folder = None):
    img = io.imread(img_path)
    if mask_path != None: mask = io.imread(mask_path)
    height, width = img.shape
    div = int(min([height, width])/final_sq_shape + 0.5)
    if [height, width] != [final_sq_shape*div, final_sq_shape*div]:
        img = resize(img, (final_sq_shape*div, final_sq_shape*div))
        if mask_path != None: mask = resize(mask, (final_sq_shape*div, final_sq_shape*div))
    for i in range(div):
        for e in range(div):
            cropped_img = img[final_sq_shape*i:final_sq_shape*(i+1),
                          final_sq_shape*e:final_sq_shape*(e+1)]
            io.imsave(os.path.join(img_folder, img_name + '_' + str(i+1) + '_' + str(e+1) + '.png'), cropped_img)
            if mask_path != None: cropped_mask = mask[final_sq_shape*i:final_sq_shape*(i+1),
                          final_sq_shape*e:final_sq_shape*(e+1)]
            if mask_path != None: io.imsave(os.path.join(mask_folder, img_name + '_' + str(i+1) + '_' + str(e+1) +'.png'), cropped_mask)

img_crop_path = os.path.join(basepath, 'imgCrops')
mask_crop_path = os.path.join(basepath, 'maskCrops')
os.makedirs(img_crop_path)
os.makedirs(mask_crop_path)


'''
#####################################################
---------------- UNet STRUCTURE ---------------------
#####################################################
'''

def dbl_conv(prev_layer, channels, activ, do = 0, batch_norm = 0, kernel_initializer='he_normal'):
    # prev_layer : the previous layer that will be used as input
    # channels: number of channels in the current convolution
    # activ: activation function
    # batch_norm: batch normalization [0 or 1), not included initially
    # do: dropout between convolutions? from 0 to 1
    m = Conv2D(channels, (3, 3), kernel_initializer = kernel_initializer, activation=activ, padding='same')(prev_layer)
    if batch_norm != 0: m = BatchNormalization()(m)
    if do > 0 and do < 1: m = Dropout(do)(m)
    m = Conv2D(channels, (3, 3),kernel_initializer = kernel_initializer, activation=activ, padding='same')(m)
    if batch_norm != 0: m = BatchNormalization()(m)
    return m
    
def UNet(img_shape, levels=5, initial_channels = 32, channels_rate = 2, activ = 'relu', batch_norm = 0, do = 0):
    # img_shape: shape of input images
    # levels: total number of levels that the UNet will have
    # initial_channels: the number of channels in the 1st level
    # channels_rate: how the number of channels will be modified per level
    # activ: activation function
    # batch_norm: batch normalization [0 or 1), not included initially
    # do: dropout between convolutions? from 0 to 1
    inputs = Input(shape=img_shape)
    channels = initial_channels
    UNet = {'maxpool0': inputs}
    # Level down part of the model
    for i in range(1, levels):
        UNet['conv'+str(i)] = dbl_conv(UNet['maxpool'+str(i-1)], channels, activ, do = do, batch_norm = batch_norm)
        UNet['maxpool'+str(i)] = MaxPooling2D(pool_size=(2, 2))(UNet['conv'+str(i)])
        channels *= channels_rate
    # Lowest Level of the model
    UNet['conv'+str(levels)] = dbl_conv(UNet['maxpool'+str(levels-1)], channels, activ, do = do, batch_norm = batch_norm)
    # Level up part of the model
    for i in range(levels+1, levels*2):
        channels //= channels_rate
        UNet['up'+str(i)] = UpSampling2D(2)(UNet['conv'+str(i-1)])
        UNet['up'+str(i)] = Conv2D(channels, (3, 3), activation=activ, padding='same')(UNet['up'+str(i)])
        UNet['up'+str(i)] = concatenate([UNet['up'+str(i)], UNet['conv'+str(2*levels-i)]], axis=3)
        UNet['conv'+str(i)] = dbl_conv(UNet['up'+str(i)], channels, activ, do = do, batch_norm = batch_norm)
    UNet['conv'+str(2*levels)] = Conv2D(1, (1, 1), activation='sigmoid')(UNet['conv'+str((2*levels)-1)])    
    model = Model(inputs=[inputs], outputs=[UNet['conv'+str(2*levels)] ])
    UNet['inputs'] = UNet.pop('maxpool0')
    return model

'''
#####################################################
--------------- MODEL CALLBACKS ---------------------
#####################################################
'''
class AccuracyHistory(Callback):
    def on_train_begin(self, logs={}):
        self.acc = []
        self.val_acc = []
        self.loss = []
        self.val_loss = []
        
    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
'''        
#####################################################
------------------ IOU METRIC -----------------------
#####################################################
'''

# https://www.kaggle.com/aglotero/another-iou-metric
def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = label(y_true_in > 0.5)
    y_pred = label(y_pred_in > 0.5)
    
    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred))
    
    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    
    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)

def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.array(np.mean(metric), dtype=np.float32)

def my_iou_metric(label, pred):
    metric_value = tf.py_func(iou_metric_batch, [label, pred], tf.float32)
    return metric_value

'''        
#####################################################
----------------- MODEL COMPILE ---------------------
#####################################################
'''

def modelCompile(model, optimizer = 'rmsprop', lr= 0.00001, metrics = 'accuracy'):   
    if optimizer == 'adam': optCompile = Adam(lr=lr)
    else: optCompile = 'rmsprop'
    
    model.compile(loss= binary_crossentropy,
                  optimizer= optCompile,
                  metrics=[metrics])
    print('Model Compiled')
    return model

'''
#####################################################
------------------- MODEL RUN -----------------------
#####################################################
'''
def modelHistory(model, batch_size, epochs, X_train, y_train_cat, X_test, y_test_cat,
                 bestModelPathLoss, bestModelPathAcc):
    history = AccuracyHistory()
    bestModelAcc = ModelCheckpoint(bestModelPathAcc, monitor="val_my_iou_metric",
                      save_best_only=True, save_weights_only=False)
    bestModelLoss = ModelCheckpoint(bestModelPathLoss, monitor="val_loss",
                      save_best_only=True, save_weights_only=False)
    return model.fit(X_train, y_train_cat,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(X_test, y_test_cat),
            callbacks=[history, bestModelAcc, bestModelLoss]), history
                     

#List of cropped images and masks
cropList = os.listdir(os.path.join(basepath, 'imgCrops'))
Xtrain = [os.path.join(basepath, 'imgCrops', x) for x in cropList]
ytrain = [os.path.join(basepath, 'maskCrops', x) for x in cropList]

#Reading files and adapting sizes
X_train = np.array([np.expand_dims(io.imread(x)/65535, axis = 2) for x in Xtrain])
y_train = np.array([np.expand_dims(io.imread(x)/65535, axis = 2) for x in ytrain])

os.makedirs(os.path.join(basepath, 'UNET_models'))

#Model training
levels = 5
initial_channels = 16

unet_name = 'unet_l' + str(levels) + '_c' + str(initial_channels)

keras.backend.clear_session()
bestModelPath = os.path.join(basepath, 'UNET_models')
bestModelPathIOU = bestModelPath + 'model_IOU_' + unet_name + '.hdf5'
bestModelPathLoss = bestModelPath + 'model_loss_' + unet_name + '.hdf5'

bestModelIOU = ModelCheckpoint(bestModelPathIOU, monitor="val_my_iou_metric",
                      save_best_only=True, save_weights_only=False)
bestModelLoss = ModelCheckpoint(bestModelPathLoss, monitor="val_loss",
                      save_best_only=True, save_weights_only=False)

model = UNet((None, None,1), levels=levels, initial_channels = initial_channels, channels_rate = 2,
             activ = 'relu')
model.summary()
model = modelCompile(model, optimizer = 'rmsprop', metrics = my_iou_metric)

history=[]
history = AccuracyHistory()
modelHistory = model.fit(X_train, y_train,
          batch_size=5,
          epochs=20,
          verbose=1,
          validation_split=0.1,
          callbacks=[history, bestModelIOU, bestModelLoss])


#Load model and continue training (if you want to train it more)
umodel = load_model(bestModelPathLoss,
                    custom_objects={'my_iou_metric': my_iou_metric})

UmodelHistory = umodel.fit(X_train, y_train,
          batch_size=5,
          epochs=10,
          verbose=1,
          validation_split=0.1,
          callbacks=[history, bestModelIOU, bestModelLoss])
          
 

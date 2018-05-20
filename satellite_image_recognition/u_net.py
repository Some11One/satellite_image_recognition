import keras
from keras.backend import binary_crossentropy
from keras.layers import merge, MaxPooling2D, UpSampling2D, Convolution2D, BatchNormalization
from keras.layers.core import Lambda
from keras.models import *
from keras.optimizers import *

import os
from .settings import BASE_DIR

def slice_batch(x, n_gpus, part):
    """
    Divide the input batch into [n_gpus] slices, and obtain slice no.
    [part].

    i.e. if len(x)=10, then slice_batch(x, 2, 1) will return x[5:].

    """

    sh = K.shape(x)

    L = sh[0] / n_gpus
    L = tf.cast(L, tf.int32)

    if part == n_gpus - 1:
        return x[part * L:]

    return x[part * L:int(part + 1) * L]


def to_multi_gpu(model, n_gpus=4):
    """Given a keras [model], return an equivalent model which parallelizes

    the computation over [n_gpus] GPUs.



    Each GPU gets a slice of the input batch, applies the model on that
    slice

    and later the outputs of the models are concatenated to a single
    tensor,

    hence the user sees a model that behaves the same as the original.

    """

    with tf.device('/cpu:0'):
        x = Input(model.input_shape[1:], name=model.input_names[0])

    towers = []

    for g in range(n_gpus):
        with tf.device('/gpu:' + str(g)):
            slice_g = Lambda(slice_batch, lambda shape: shape, arguments={'n_gpus': n_gpus, 'part': g})(x)

            towers.append(model(slice_g))

    with tf.device('/cpu:0'):
        merged = merge(towers, mode='concat', concat_axis=0)

    new_model = Model(input=[x], output=merged)

    funcType = type(model.save)

    # monkeypatch the save to save just the underlying model
    def new_save(self_, filepath, overwrite=True):
        model.save(filepath, overwrite)

    new_model.save = funcType(new_save, new_model)

    return new_model


smooth = 1e-12


def jaccard_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def jaccard_coef_int(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def jaccard_coef_loss(y_true, y_pred):
    return -K.log(jaccard_coef(y_true, y_pred)) + binary_crossentropy(y_pred, y_true)


class myUnet(object):

    def __init__(self, tp='buildings', img_rows=512, img_cols=512):

        self.type = tp
        self.img_rows = img_rows
        self.img_cols = img_cols

    # Define the neural network
    def get_unet(self):
        inputs = Input((512, 512, 1))
        #
        conv1 = Convolution2D(32, 3, 3, border_mode='same', kernel_initializer='he_normal')(inputs)
        conv1 = BatchNormalization(mode=0, axis=1)(conv1)
        conv1 = keras.layers.advanced_activations.ELU()(conv1)
        conv1 = Convolution2D(32, 3, 3, border_mode='same', kernel_initializer='he_normal')(conv1)
        conv1 = BatchNormalization(mode=0, axis=1)(conv1)
        conv1 = keras.layers.advanced_activations.ELU()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        #
        conv2 = Convolution2D(64, 3, 3, border_mode='same', kernel_initializer='he_normal')(pool1)
        conv2 = BatchNormalization(mode=0, axis=1)(conv2)
        conv2 = keras.layers.advanced_activations.ELU()(conv2)
        conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', kernel_initializer='he_normal')(conv2)
        conv2 = BatchNormalization(mode=0, axis=1)(conv2)
        conv2 = keras.layers.advanced_activations.ELU()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        #
        conv3 = Convolution2D(128, 3, 3, border_mode='same', kernel_initializer='he_normal')(pool2)
        conv3 = BatchNormalization(mode=0, axis=1)(conv3)
        conv3 = keras.layers.advanced_activations.ELU()(conv3)
        conv3 = Convolution2D(128, 3, 3, border_mode='same', kernel_initializer='he_normal')(conv3)
        conv3 = BatchNormalization(mode=0, axis=1)(conv3)
        conv3 = keras.layers.advanced_activations.ELU()(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        #
        conv4 = Convolution2D(256, 3, 3, border_mode='same', kernel_initializer='he_normal')(pool3)
        conv4 = BatchNormalization(mode=0, axis=1)(conv4)
        conv4 = keras.layers.advanced_activations.ELU()(conv4)
        conv4 = Convolution2D(256, 3, 3, border_mode='same', kernel_initializer='he_normal')(conv4)
        conv4 = BatchNormalization(mode=0, axis=1)(conv4)
        conv4 = keras.layers.advanced_activations.ELU()(conv4)
        #         drop4 = Dropout(0.5)(conv?4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        #
        conv5 = Convolution2D(512, 3, 3, border_mode='same', kernel_initializer='he_normal')(pool4)
        conv5 = BatchNormalization(mode=0, axis=1)(conv5)
        conv5 = keras.layers.advanced_activations.ELU()(conv5)
        conv5 = Convolution2D(512, 3, 3, border_mode='same', kernel_initializer='he_normal')(conv5)
        conv5 = BatchNormalization(mode=0, axis=1)(conv5)
        conv5 = keras.layers.advanced_activations.ELU()(conv5)
        #         drop5 = Dropout(0.5)(conv5)
        #
        up1 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=-1)
        conv6 = Convolution2D(256, 3, 3, border_mode='same', kernel_initializer='he_normal')(up1)
        conv6 = BatchNormalization(mode=0, axis=1)(conv6)
        conv6 = keras.layers.advanced_activations.ELU()(conv6)
        conv6 = Convolution2D(256, 3, 3, border_mode='same', kernel_initializer='he_normal')(conv6)
        conv6 = BatchNormalization(mode=0, axis=1)(conv6)
        conv6 = keras.layers.advanced_activations.ELU()(conv6)
        #
        up2 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=-1)
        conv7 = Convolution2D(128, 3, 3, border_mode='same', kernel_initializer='he_normal')(up2)
        conv7 = BatchNormalization(mode=0, axis=1)(conv7)
        conv7 = keras.layers.advanced_activations.ELU()(conv7)
        conv7 = Convolution2D(128, 3, 3, border_mode='same', kernel_initializer='he_normal')(conv7)
        conv7 = BatchNormalization(mode=0, axis=1)(conv7)
        conv7 = keras.layers.advanced_activations.ELU()(conv7)
        #
        up3 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=-1)
        conv8 = Convolution2D(64, 3, 3, border_mode='same', kernel_initializer='he_normal')(up3)
        conv8 = BatchNormalization(mode=0, axis=1)(conv8)
        conv8 = keras.layers.advanced_activations.ELU()(conv8)
        conv8 = Convolution2D(64, 3, 3, border_mode='same', kernel_initializer='he_normal')(conv8)
        conv8 = BatchNormalization(mode=0, axis=1)(conv8)
        conv8 = keras.layers.advanced_activations.ELU()(conv8)
        #
        up4 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=-1)
        conv9 = Convolution2D(32, 3, 3, border_mode='same', kernel_initializer='he_normal')(up4)
        conv9 = BatchNormalization(mode=0, axis=1)(conv9)
        conv9 = keras.layers.advanced_activations.ELU()(conv9)
        conv9 = Convolution2D(32, 3, 3, border_mode='same', kernel_initializer='he_normal')(conv9)
        conv9 = BatchNormalization(mode=0, axis=1)(conv9)
        conv9 = keras.layers.advanced_activations.ELU()(conv9)
        conv9 = Convolution2D(2, 3, 3, border_mode='same', kernel_initializer='he_normal')(conv9)
        #
        conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

        model = Model(input=inputs, output=conv10)

        model.compile(optimizer=Nadam(lr=1e-3), loss=jaccard_coef_loss,
                      metrics=['binary_crossentropy', jaccard_coef_int])
        # model = to_multi_gpu(model)
        # model.compile(optimizer=Nadam(lr=1e-3), loss=jaccard_coef_loss,
        #               metrics=['binary_crossentropy', jaccard_coef_int])

        return model

    def predict(self, imgs_test):

        model = self.get_unet()
        if self.type == 'buildings':
            model.load_weights(os.path.join(BASE_DIR, 'model_weights/unet_b.hdf5'))
        else:
            model.load_weights(os.path.join(BASE_DIR, 'model_weights/unet_c.hdf5'))
        model.compile(optimizer=Nadam(lr=1e-3), loss=jaccard_coef_loss,
                      metrics=['binary_crossentropy', jaccard_coef_int])

        imgs_mask_test = model.predict(imgs_test, batch_size=8, verbose=1)
        return imgs_mask_test

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 01:38:53 2019

@author: LAB528
"""
import tensorflow as tf
from tensorflow.compat.v1.keras.initializers import he_normal
from tensorflow.keras.utils import plot_model
from tensorflow.keras import regularizers,metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D,  Flatten
from tensorflow.keras.optimizers import Adam

def akcu_model_seq(lr, weight_decay):
    
    model = Sequential()
    model.add(Conv2D(16,
               kernel_size=7,
               strides=2,
               padding='same',
               activation='relu',
               input_shape=(64,64,1),
               kernel_initializer=he_normal(),
               kernel_regularizer=regularizers.l2(weight_decay)
               ))
    model.add(Conv2D(32,
               kernel_size=3,
               strides=2,
               padding='valid',
               activation='relu',
               kernel_initializer=he_normal(),
               kernel_regularizer=regularizers.l2(weight_decay)
               ))
    model.add(Conv2D(32,
               kernel_size=3,
               strides=2,
               padding='valid',
               activation='relu',
               kernel_initializer=he_normal(),
               kernel_regularizer=regularizers.l2(weight_decay)
               ))
    model.add(Flatten())
    model.add(Dense(96,
                    activation='relu',
                    kernel_initializer=he_normal()))
    model.add(Dense(16,
                    activation='relu',
                    kernel_initializer=he_normal()))
    model.add(Dense(1,
                    activation='sigmoid',
                    kernel_initializer=he_normal()))
    
    print(model.summary())
    #plot_model(model, to_file='test01.png')
    
    #sgd = SGD(lr=lr,momentum=0.9)
    adam = Adam(lr=lr,epsilon=1e-10,amsgrad=True)
    model.compile(optimizer=adam,
              loss='binary_crossentropy',
              #loss='sparse_categorical_crossentropy',
              #loss='mse',
              metrics=['accuracy'])
    return model

# =============================================================================
# model = akcu_model_seq(lr=1, weight_decay=1)
# =============================================================================

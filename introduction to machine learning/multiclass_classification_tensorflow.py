from __future__ import absolute_import, division, print_function


#To void error message spam (CPU-Programming not optimized for Tensorflow, not sure how to fix it. It's fast anyway!)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'

# TensorFlow and tf.keras
import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint,EarlyStopping

#Try to make it use all CPUs
import tensorflow as tf
K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=32, inter_op_parallelism_threads=32)))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.layers import ELU

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#Hyperas
from hyperas import optim
from hyperas.distributions import choice, uniform
from hyperopt import Trials, STATUS_OK, tpe

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
print(keras.__version__)

from keras import backend as K
K.tensorflow_backend._get_available_gpus()

# https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
# Function to create model, required for KerasClassifier


def data():
    #import, standardize and split data
    train = pd.read_csv("train.csv")

    y = train.loc[:,'y']
    X = train.loc[:,'x1'::]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def create_model(X_train,y_train,X_test,y_test):
# create model

    K.clear_session()
    activation_function = {{choice(['LeakyReLU', 'ELU'])}}
    input_shape = X_train.shape[1]
    

    model = Sequential()
    model.add(Dense({{choice([60,280,500,720,940,1160])}}, input_dim = input_shape ))
    if activation_function == 'LeakyReLU':
        model.add(LeakyReLU())
    else:
        model.add(ELU())
    model.add(BatchNormalization())
    
    model.add(Dropout({{uniform(0, 1)}}, seed=42))
    model.add(Dense({{choice([60,280,500,720,940,1160])}}))
    if activation_function == 'LeakyReLU':
        model.add(LeakyReLU())
    else:
        model.add(ELU())
    model.add(BatchNormalization())
    
    model.add(Dropout({{uniform(0, 1)}}, seed=42))
    model.add(Dense({{choice([60,280,500,720,940,1160])}}))
    if activation_function == 'LeakyReLU':
        model.add(LeakyReLU())
    else:
        model.add(ELU())
    model.add(BatchNormalization())
    
    model.add(Dropout({{uniform(0, 1)}}, seed=42))
    model.add(Dense({{choice([60,280,500,720,940,1160])}}))
    if activation_function == 'LeakyReLU':
        model.add(LeakyReLU())
    else:
        model.add(ELU())
    model.add(BatchNormalization())


    nr_hlayers = {{choice(['three_layers', 'four_layers','five_layers'])}}

    if nr_hlayers=='three_layers':
        pass

    if nr_hlayers== 'four_layers':
        model.add(Dropout({{uniform(0, 1)}}, seed=42))
        model.add(Dense({{choice([60,280,500,720,940,1160])}}))
        if activation_function == 'LeakyReLU':
            model.add(LeakyReLU())
        else:
            model.add(ELU())
        model.add(BatchNormalization())

    if nr_hlayers== 'five_layers':
        model.add(Dropout({{uniform(0, 1)}}, seed=42))
        model.add(Dense({{choice([60,280,500,720,940,1160])}}))
        if activation_function == 'LeakyReLU':
            model.add(LeakyReLU())
        else:
            model.add(ELU())
        model.add(BatchNormalization())

        model.add(Dropout({{uniform(0, 1)}}, seed=42))
        model.add(Dense({{choice([60,280,500,720,940,1160])}}))
        if activation_function == 'LeakyReLU':
            model.add(LeakyReLU())
        else:
            model.add(ELU())
        model.add(BatchNormalization())


    optimizer= {{choice(['adam','adagrad','sgd','sgd_nest'])}}
    lr = {{choice([0.001,0.0001])}}

    if optimizer=="adam":
            opt=keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999)
    if optimizer=="adagrad":
            opt=keras.optimizers.Adagrad(lr=lr, epsilon=None)
    if optimizer=="sgd":
            opt=keras.optimizers.SGD(lr=lr, nesterov=False)
    if optimizer=="sgd_nest":
            opt=keras.optimizers.SGD(lr=lr, nesterov=True)


    model.add(Dropout({{uniform(0, 1)}}, seed=42)),
    model.add(Dense(5, activation='softmax'))
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    result = model.fit(X_train, y_train,
          batch_size=256,
          epochs=256,
          verbose=0,
          callbacks=[EarlyStopping(monitor='val_loss', patience=9)],
          validation_data = (X_test,y_test))



    loss, acc = model.evaluate(X_test,y_test, batch_size = 256, verbose = 0)
    print('Val acc:',acc, 'Loss:',loss)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


# Start and analyze result and create submission-file:
if __name__ == '__main__':
    trials= Trials() 
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=85,
                                          trials=trials)

    train = pd.read_csv("train.csv")
    y = train.loc[:,'y']
    X = train.loc[:,'x1'::]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)


    #Sometimes keras gives back an error while trying to evaluate and creating a CSV-file. Why? Don't know. Sometimes it works.
    try:
        print("Evalutation of best performing model (on all data):")
        print(best_model.evaluate(X, y))
    except:
        print("Error with evaluate")

    print("Best performing model chosen hyper-parameters:")
    print(best_run)

    try:
        #Create Submission-File:
        test = pd.read_csv("test.csv")
        sample = pd.read_csv('sample.csv')
    
        scaler = StandardScaler()
    
        X_submission = test.loc[:,:]
        X_submission = scaler.fit_transform(X_submission)
    
    
        y_res = pd.DataFrame()
        y_res['Id']  = sample.loc[:,'Id']
        y_res['y'] = best_model.predict_classes(X_submission)
    

        y_res.to_csv('result.csv', index = False)
    except:
        print("Error with file-creating")

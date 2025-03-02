import numpy as np
from keras_preprocessing.image import ImageDataGenerator #type:ignore
from keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,Dropout #type:ignore
from tensorflow.keras.models import Model #type:ignore
from glob import glob
import os
import argparse
from get_data import read_params,get_data
import matplotlib.pyplot as plt 
from keras.applications.vgg16 import VGG16 #type:ignore
import tensorflow as tf #type:ignore
import mlflow #type:ignore
from urllib.parse import urlparse
import mlflow.keras #type:ignore


def train_model_mlflow(config_file):
    config = get_data(config_file)
    train = config['model']['trainable']
    if train:
        img_size = config['model']['image_size']
        train_set = config['model']['train_path']
        test_set = config['model']['test_path']
        num_cls = config['load_data']['num_classes']
        rescale = config['img_augment']['rescale']
        shear_range = config['img_augment']['shear_range']
        zoom_range = config['img_augment']['zoom_range']
        horizontal_flip = config['img_augment']['horizontal_flip']
        vertical_flip = config['img_augment']['vertical_flip']
        class_mode = config['img_augment']['class_mode']
        batch = config['img_augment']['batch_size']
        loss = config['model']['loss']
        optimizer = config['model']['optimizer']
        metrics = config['model']['metrics']        
        epochs = config['model']['epochs']        
        model_path = config['model']['sav_dir']

        print(type(batch))

        resnet = VGG16(input_shape = img_size+[3], weights = 'imagenet',include_top=False)
        for p in resnet.layers:
            p.trainable = False
        
        op = Flatten()(resnet.output)
        prediction = Dense(num_cls, activation='softmax')(op)
        mod = Model(inputs=resnet.input,outputs = prediction)
        print(mod.summary())

        img_size = tuple(img_size)

        mod.compile(loss = loss, optimizer = optimizer, metrics = metrics)

        train_gen = ImageDataGenerator(
            rescale = rescale,
            shear_range = shear_range,
            zoom_range = zoom_range,
            horizontal_flip = horizontal_flip,
            vertical_flip = vertical_flip,
            rotation_range = 90
        )

        test_gen = ImageDataGenerator(rescale=rescale)

        train_set = train_gen.flow_from_directory(train_set,
            target_size = img_size,
            batch_size = batch,
            class_mode = class_mode
        )

        test_set = test_gen.flow_from_directory(test_set,
            target_size = img_size,
            batch_size = batch,
            class_mode = class_mode
        )

        ########################### Start of ML flow #####################################

        mlflow_config = config['mlflow_config']
        remote_server_uri = mlflow_config['remote_server_uri']
        mlflow.set_tracking_uri(remote_server_uri)
        mlflow.set_experiment(mlflow_config['experiment_name'])

        with mlflow.start_run():
            history = mod.fit(train_set,
                epochs = epochs,
                steps_per_epoch = len(train_set),
                validation_data = test_set,
                validation_steps = len(test_set)
            )

            train_loss = history.history['loss'][-1]
            val_loss = history.history['val_loss'][-1]
            train_acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']
            
            mlflow.log_param('epochs',epochs)
            mlflow.log_param('loss',loss)
            mlflow.log_param('val_loss',val_loss)
            mlflow.log_param('train_loss',train_loss)
            mlflow.log_param('val_acc',val_acc)
            mlflow.log_param('train_acc',train_acc)
            mlflow.log_param("metrics",metrics)

            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            
            if tracking_url_type_store != 'file':
                mlflow.keras.log_model(mod,'model',registered_model_name=mlflow_config['register_model_name'])
            else:
                mlflow.keras.log_model(mod,'model')
    
    else:
        print("Model is not trainable")








if __name__=='__main__':
    args=argparse.ArgumentParser()
    args.add_argument('--config',default='params.yaml')
    passed_args=args.parse_args()
    train_model_mlflow(config_file=passed_args.config)
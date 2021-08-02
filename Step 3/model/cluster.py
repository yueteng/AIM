#!usr/bin/env python
#coding:utf-8
"""
@author: Haidong Zhang
@contact: haidong_zhang14@yahoo.com
@time: 2021/1/15 9:34
@project: MalariaDetection
@description: https://blog.keras.io/building-autoencoders-in-keras.html
"""
from keras.layers import *      #For adding convolutional layer
from keras.layers import Dense, ZeroPadding2D, BatchNormalization,  Activation        #For adding layers to NN
from keras.optimizers import adam
from keras.models import *                  #for loading the model
from keras.backend import stack, squeeze, expand_dims
from keras.backend import sum as keras_sum
from utils import module
from keras import initializers
import os, csv

from time import time
import numpy as np
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from sklearn.cluster import KMeans

from model import metrics


kernel = 3
stride = 2


def CAE(input_shape, conv_filters=[64, 128, 256, 512, 1024]):
    model = Sequential()
    model.add(Conv2D(conv_filters[0], (kernel, kernel), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D((stride, stride), padding='same'))

    for dim in conv_filters[1:-1]:
        model.add(Conv2D(dim, (kernel, kernel), activation='relu', padding='same'))
        model.add(MaxPooling2D((stride, stride), padding='same'))

    model.add(Conv2D(conv_filters[-1], (kernel, kernel), activation='relu', padding='same'))
    model.add(MaxPooling2D((stride, stride), padding='same', name='embedding'))

    for dim in conv_filters[-2::-1]:
        # model.add(Conv2DTranspose(dim, (kernel, kernel), activation='relu', padding='same'))
        model.add(Conv2D(dim, (kernel, kernel), activation=None, padding='same'))
        model.add(UpSampling2D(size=(stride, stride), interpolation='bilinear'))

    model.add(Conv2D(input_shape[-1], (kernel, kernel), activation='sigmoid', padding='same'))
    model.add(UpSampling2D(size=(stride, stride), interpolation='bilinear'))
    model.summary()
    return model



class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.
    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight((self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DCEC(object):
    def __init__(self,
                 input_shape,
                 filters=[64, 128, 256, 512, 512],
                 n_clusters=10,
                 alpha=1.0, pretrained=False):

        super(DCEC, self).__init__()

        self.n_clusters = n_clusters
        self.input_shape = input_shape
        self.alpha = alpha
        self.pretrained = pretrained
        self.y_pred = []

        self.cae = CAE(input_shape, filters)
        hidden = self.cae.get_layer(name='embedding').output
        hidden = Flatten()(hidden)
        self.encoder = Model(inputs=self.cae.input, outputs=hidden)

        # Define DCEC model
        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(hidden)
        # Define prediction
        pred_label = Dense(512, activation="relu")(hidden)
        pred_label = Dense(1, activation="sigmoid")(pred_label)

        self.model = Model(inputs=self.cae.input,
                           outputs=[pred_label, clustering_layer, self.cae.output])

    def pretrain(self, x, batch_size=256, epochs=200, optimizer='adam', save_dir='results/temp'):
        self.cae.compile(optimizer=optimizer, loss='mse')

        # begin training
        self.cae.fit(x, x, batch_size=batch_size, epochs=epochs)
        self.cae.save(save_dir + '/pretrain_cae_model.h5')
        print('Pretrained weights are saved to %s/pretrain_cae_model.h5' % save_dir)
        self.pretrained = True

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def extract_feature(self, x):  # extract features from before clustering layer
        return self.encoder.predict(x)

    def predict(self, x):
        q, _ = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def compile(self, loss=['binary_crossentropy', 'kld', 'mse', ], loss_weights=[1, 1, 1], optimizer='adam'):
        self.model.compile(loss=loss, loss_weights=loss_weights, optimizer=optimizer)

    def fit(self, train_generator, valid_generator, batch_size=256, maxiter=2e4, tol=1e-3,
            update_interval=140, cae_weights=None, save_dir='./results/temp'):

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print('Update interval', update_interval)
        save_interval = train_generator.x.shape[0] / batch_size * 5
        print('Save interval', save_interval)

        # Step 1: pretrain if necessary
        t0 = time()
        if not self.pretrained and cae_weights is None:
            print('...pretraining CAE using default hyper-parameters:')
            print('   optimizer=\'adam\';   epochs=200')
            self.pretrain(train_generator.x * 1.0/255, batch_size, save_dir=save_dir)
            self.pretrained = True
        elif self.pretrained:
            self.cae = load_model(save_dir + '/pretrain_cae_model.h5')
        elif cae_weights is not None:
            self.cae.load_weights(cae_weights)
            print('cae_weights is loaded successfully.')

        # Step 2: initialize cluster centers using k-means
        t1 = time()
        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        self.y_pred = kmeans.fit_predict(self.encoder.predict(train_generator.x * 1.0/255))
        y_pred_last = np.copy(self.y_pred)
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

        # Step 3: deep clustering
        # logging file
        logfile = open(save_dir + '/dcec_log.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'L', 'Lc', 'Lr'])
        logwriter.writeheader()

        t2 = time()
        loss = [0, 0, 0]
        index = 0
        best_acc = 0.0
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                pred_labels, q, _ = self.model.predict(train_generator.x * 1.0/255, verbose=0)
                p = self.target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                # self.y_pred = q.argmax(1)
                self.y_pred = np.array(np.reshape(np.round(pred_labels), [-1]), np.int)

                acc = np.round(metrics.acc(train_generator.y, self.y_pred), 5)
                nmi = np.round(metrics.nmi(train_generator.y, self.y_pred), 5)
                ari = np.round(metrics.ari(train_generator.y, self.y_pred), 5)
                loss = np.round(loss, 5)
                logdict = dict(iter=ite, acc=acc, nmi=nmi, ari=ari, L=loss[0], Lc=loss[1], Lr=loss[2])
                logwriter.writerow(logdict)
                print('Iter', ite, ': Acc', acc, ', nmi', nmi, ', ari', ari, '; loss=', loss)

                # check stop criterion
                delta_label = np.sum(self.y_pred != y_pred_last).astype(np.float32) / self.y_pred.shape[0]
                y_pred_last = np.copy(self.y_pred)
                if ite > 0 and best_acc < acc:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    # logfile.close()
                    best_acc = acc
                    # break

            # train on batch
            batch_x, batch_y = train_generator.next()
            loss = self.model.train_on_batch(x=batch_x,
                                             y=[batch_y, batch_x])

            # save intermediate model
            if ite % save_interval == 0:
                # save DCEC model checkpoints
                print('saving model to:', save_dir + '/dcec_model_' + str(ite) + '.h5')
                self.model.save_weights(save_dir + '/dcec_model_' + str(ite) + '.h5')

            # ite += 1

        # save the trained model
        logfile.close()
        print('saving model to:', save_dir + '/dcec_model_final.h5')
        self.model.save_weights(save_dir + '/dcec_model_final.h5')
        t3 = time()
        print('Pretrain time:  ', t1 - t0)
        print('Clustering time:', t3 - t1)
        print('Total time:     ', t3 - t0)


def cluster_claffication():
    img_input = Input(shape=(64, 64, 3))

    x = img_input
    conv_dims = [64, 128, 256, 512, 512]
    multi_scale_results = list()
    for dim in conv_dims:
        x = Conv2D(dim, (kernel, kernel), activation='relu', padding='same')(x)
        x = Conv2D(dim, (kernel, kernel), activation='relu', padding='same')(x)
        multi_scale_results.append(x)
        x = MaxPooling2D((stride, stride), padding='same')(x)

    model = Flatten()(x)
    model = Dense(512, activation="relu")(model)
    o = Dense(1, activation="sigmoid")(model)
    model = module.model_compile(img_input, o)
    return model

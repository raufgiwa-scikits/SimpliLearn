import os
import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# from .layers import *
# from .utils import get_interaction_list

import os
import struct
import numpy as np
import ctypes as ct
from sys import platform
from numpy.ctypeslib import ndpointer
from collections import OrderedDict
from pandas.core.generic import NDFrame
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
try:
    from pandas.api.types import is_numeric_dtype, is_string_dtype
except ImportError:  # pragma: no cover
    from pandas.core.dtypes.common import is_numeric_dtype, is_string_dtype


import os
import numpy as np
import pandas as pd
from contextlib import closing
from itertools import combinations

import matplotlib
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from joblib import Parallel, delayed

# from .interpret import *

import numpy as np
import tensorflow as tf
import tensorflow_lattice as tfl
from tensorflow.keras import layers



def fprint(n,ls,s):
    if n in ls:
        print(s)

class GAMINet(tf.keras.Model):

    def __init__(self, meta_info,
                 interact_num=20,
                 subnet_arch=[40] * 5,
                 interact_arch=[40] * 5,
                 lr_bp=[1e-4, 1e-4, 1e-4],
                 epochs = [50, 50, 50],
                 early_stop_thres=[50, 50, 50],
                 training=[True,False,False],
                 batch_size=200,
                 task_type="Regression",
                 activation_func=tf.nn.relu,
                 heredity=True,
                 reg_clarity=0.1,
                 loss_threshold=0.01,
                 val_ratio=0.2,
                 mono_increasing_list=None,
                 mono_decreasing_list=None,
                 convex_list=None,
                 concave_list=None,
                 lattice_size=2,
                 include_interaction_list=[],
                 verbose=False,
                 random_state=0,
                 message_list=[0,1]):

        super(GAMINet, self).__init__()


        self.message_list=message_list
        self.training_main = training[0]
        self.training_interaction = training[1]
        self.training_fine = training[2]

        self.meta_info = meta_info
        self.subnet_arch = subnet_arch
        self.interact_arch = interact_arch
        
        
        self.class_weight = None
        self.sample_weight = None


        self.lr_bp = lr_bp
        self.batch_size = batch_size
        self.task_type = task_type
        self.activation_func = activation_func
        self.tuning_epochs = epochs[0]
        self.main_effect_epochs = epochs[1]
        self.interaction_epochs = epochs[2]
        self.early_stop_thres = early_stop_thres
        self.early_stop_thres1 = early_stop_thres[0]
        self.early_stop_thres2 = early_stop_thres[1]
        self.early_stop_thres3 = early_stop_thres[2]

        self.heredity = heredity
        self.reg_clarity = reg_clarity
        self.loss_threshold = loss_threshold

        self.mono_increasing_list = [] if mono_increasing_list is None else mono_increasing_list
        self.mono_decreasing_list = [] if mono_decreasing_list is None else mono_decreasing_list
        self.mono_list = self.mono_increasing_list + self.mono_decreasing_list
        self.convex_list = [] if convex_list is None else convex_list
        self.concave_list = [] if concave_list is None else concave_list
        self.con_list = self.convex_list + self.concave_list
        self.lattice_size = lattice_size
        
        self.verbose = verbose
        self.val_ratio = val_ratio
        self.random_state = random_state

        np.random.seed(random_state)
        tf.random.set_seed(random_state)

        self.dummy_values_ = {}
        self.nfeature_scaler_ = {}
        self.cfeature_num_ = 0
        self.nfeature_num_ = 0
        self.cfeature_list_ = []
        self.nfeature_list_ = []
        self.cfeature_index_list_ = []
        self.nfeature_index_list_ = []

        self.feature_list_ = []
        self.feature_type_list_ = []
        for idx, (feature_name, feature_info) in enumerate(meta_info.items()):
            if feature_info["type"] == "target":
                continue
            if feature_info["type"] == "categorical":
                self.cfeature_num_ += 1
                self.cfeature_list_.append(feature_name)
                self.cfeature_index_list_.append(idx)
                self.feature_type_list_.append("categorical")
                self.dummy_values_.update({feature_name: meta_info[feature_name]["values"]})
            else:
                self.nfeature_num_ += 1
                self.nfeature_list_.append(feature_name)
                self.nfeature_index_list_.append(idx)
                self.feature_type_list_.append("continuous")
                self.nfeature_scaler_.update({feature_name: meta_info[feature_name]["scaler"]})
            self.feature_list_.append(feature_name)

        # build
        self.interaction_list = []
        self.interact_num_added = 0
        self.interaction_status = False
        self.input_num = self.nfeature_num_ + self.cfeature_num_
        self.max_interact_num = int(round(self.input_num * (self.input_num - 1) / 2))
        self.include_interaction_list = list(set(include_interaction_list))
        self.interact_num = min(interact_num + len(self.include_interaction_list), self.max_interact_num)

        if self.training_main:
            self.MainEffectBlock = MainEffectBlock(feature_list=self.feature_list_,
                                    dummy_values=self.dummy_values_,
                                    nfeature_index_list=self.nfeature_index_list_,
                                    cfeature_index_list=self.cfeature_index_list_,
                                    subnet_arch=self.subnet_arch,
                                    activation_func=self.activation_func,
                                    mono_increasing_list=self.mono_increasing_list,
                                    mono_decreasing_list=self.mono_decreasing_list,
                                    convex_list=self.convex_list,
                                    concave_list=self.concave_list,
                                    lattice_size=self.lattice_size)
            
        if self.training_interaction:
            self.InteractionBlock = InteractionBlock(interact_num=self.interact_num,
                                    feature_list=self.feature_list_,
                                    cfeature_index_list=self.cfeature_index_list_,
                                    dummy_values=self.dummy_values_,
                                    interact_arch=self.interact_arch,
                                    activation_func=self.activation_func,
                                    mono_increasing_list=self.mono_increasing_list,
                                    mono_decreasing_list=self.mono_decreasing_list,
                                    convex_list=self.convex_list,
                                    concave_list=self.concave_list,
                                    lattice_size=self.lattice_size)
        
        self.OutputLayer = OutputLayer(input_num=self.input_num,
                            interact_num=self.interact_num,
                            mono_increasing_list=self.mono_increasing_list,
                            mono_decreasing_list=self.mono_decreasing_list,
                            convex_list=self.convex_list,
                            concave_list=self.concave_list)

        self.optimizer = tf.keras.optimizers.Adam()
        if self.task_type == "Regression":
            self.loss_fn = tf.keras.losses.MeanSquaredError()
        elif self.task_type == "Classification":
            self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        else:
            raise ValueError("The task type is not supported")

    def call(self, inputs, sample_weight=None, main_effect_training=False, interaction_training=False):

        self.clarity_loss = tf.constant(0.0)
        self.maineffect_outputs = self.MainEffectBlock(inputs, sample_weight, training=main_effect_training)
        if self.interaction_status:
            self.interact_outputs = self.InteractionBlock(inputs, sample_weight, training=interaction_training)
            main_weights = tf.multiply(self.OutputLayer.main_effect_switcher, self.OutputLayer.main_effect_weights)
            interaction_weights = tf.multiply(self.OutputLayer.interaction_switcher, self.OutputLayer.interaction_weights)
            for i, (k1, k2) in enumerate(self.interaction_list):
                a1 = tf.multiply(tf.gather(self.maineffect_outputs, [k1], axis=1), tf.gather(main_weights, [k1], axis=0))
                a2 = tf.multiply(tf.gather(self.maineffect_outputs, [k2], axis=1), tf.gather(main_weights, [k2], axis=0))
                b = tf.multiply(tf.gather(self.interact_outputs, [i], axis=1), tf.gather(interaction_weights, [i], axis=0))
                if sample_weight is not None:
                    self.clarity_loss += tf.abs(tf.reduce_mean(tf.multiply(tf.multiply(a1, b), tf.reshape(sample_weight, (-1, 1)))))
                    self.clarity_loss += tf.abs(tf.reduce_mean(tf.multiply(tf.multiply(a2, b), tf.reshape(sample_weight, (-1, 1)))))
                else:
                    self.clarity_loss += tf.abs(tf.reduce_mean(tf.multiply(a1, b)))
                    self.clarity_loss += tf.abs(tf.reduce_mean(tf.multiply(a2, b)))
        else:
            self.interact_outputs = tf.zeros([inputs.shape[0], self.interact_num])

        concat_list = [self.maineffect_outputs]
        if self.interact_num > 0:
            concat_list.append(self.interact_outputs)

        if self.task_type == "Regression":
            output = self.OutputLayer(tf.concat(concat_list, 1))
        elif self.task_type == "Classification":
            output = tf.nn.sigmoid(self.OutputLayer(tf.concat(concat_list, 1)))
        else:
            raise ValueError("The task type is not supported")

        return output

    @tf.function
    def predict_graph(self, x, main_effect_training=False, interaction_training=False):
        return self.__call__(x, sample_weight=None,
                      main_effect_training=main_effect_training,
                      interaction_training=interaction_training)

    def predict(self, x):
        return self.predict_graph(tf.cast(x, tf.float32)).numpy()

    @tf.function
    def evaluate_graph_init(self, x, y, sample_weight=None, main_effect_training=False, interaction_training=False):
        return self.loss_fn(y, self.__call__(x, sample_weight,
                               main_effect_training=main_effect_training,
                               interaction_training=interaction_training), sample_weight=sample_weight)

    @tf.function
    def evaluate_graph_inter(self, x, y, sample_weight=None, main_effect_training=False, interaction_training=False):
        return self.loss_fn(y, self.__call__(x, sample_weight,
                               main_effect_training=main_effect_training,
                               interaction_training=interaction_training), sample_weight=sample_weight)

    def evaluate(self, x, y, sample_weight=None, main_effect_training=False, interaction_training=False):
        if self.interaction_status:
            return self.evaluate_graph_inter(tf.cast(x, tf.float32), tf.cast(y, tf.float32),
                                  tf.cast(sample_weight, tf.float32) if sample_weight is not None else None,
                                  main_effect_training=main_effect_training,
                                  interaction_training=interaction_training).numpy()
        else:
            return self.evaluate_graph_init(tf.cast(x, tf.float32), tf.cast(y, tf.float32),
                                  tf.cast(sample_weight, tf.float32) if sample_weight is not None else None,
                                  main_effect_training=main_effect_training,
                                  interaction_training=interaction_training).numpy()

    @tf.function
    def train_main_effect(self, inputs, labels, sample_weight=None):

        with tf.GradientTape() as tape:
            pred = self.__call__(inputs, sample_weight, main_effect_training=True, interaction_training=False)
            total_loss = self.loss_fn(labels, pred, sample_weight=sample_weight)

        train_weights_list = []
        train_weights = self.MainEffectBlock.weights
        train_weights.append(self.OutputLayer.main_effect_weights)
        train_weights.append(self.OutputLayer.output_bias)
        trainable_weights_names = [w.name for w in self.trainable_weights]
        for i in range(len(train_weights)):
            if train_weights[i].name in trainable_weights_names:
                train_weights_list.append(train_weights[i])
        grads = tape.gradient(total_loss, train_weights_list)
        self.optimizer.apply_gradients(zip(grads, train_weights_list))

    @tf.function
    def train_interaction(self, inputs, labels, sample_weight=None):

        with tf.GradientTape() as tape:
            pred = self.__call__(inputs, sample_weight, main_effect_training=False, interaction_training=True)
            total_loss = self.loss_fn(labels, pred, sample_weight=sample_weight) + self.reg_clarity * self.clarity_loss

        train_weights_list = []
        train_weights = self.InteractionBlock.weights
        train_weights.append(self.OutputLayer.interaction_weights)
        train_weights.append(self.OutputLayer.output_bias)
        trainable_weights_names = [w.name for w in self.trainable_weights]
        for i in range(len(train_weights)):
            if train_weights[i].name in trainable_weights_names:
                train_weights_list.append(train_weights[i])
        grads = tape.gradient(total_loss, train_weights_list)
        self.optimizer.apply_gradients(zip(grads, train_weights_list))

    @tf.function
    def train_all(self, inputs, labels, sample_weight=None):

        with tf.GradientTape() as tape_maineffects:
            with tf.GradientTape() as tape_intearction:
                pred = self.__call__(inputs, sample_weight, main_effect_training=True, interaction_training=False)
                total_loss_maineffects = self.loss_fn(labels, pred, sample_weight=sample_weight)
                total_loss_interactions = self.loss_fn(labels, pred, sample_weight=sample_weight) + self.reg_clarity * self.clarity_loss

        train_weights_list = []
        train_weights = self.MainEffectBlock.weights
        train_weights.append(self.OutputLayer.main_effect_weights)
        train_weights.append(self.OutputLayer.output_bias)
        trainable_weights_names = [w.name for w in self.trainable_weights]
        for i in range(len(train_weights)):
            if train_weights[i].name in trainable_weights_names:
                train_weights_list.append(train_weights[i])
        grads = tape_maineffects.gradient(total_loss_maineffects, train_weights_list)
        self.optimizer.apply_gradients(zip(grads, train_weights_list))

        train_weights_list = []
        train_weights = self.InteractionBlock.weights
        train_weights.append(self.OutputLayer.interaction_weights)
        train_weights.append(self.OutputLayer.output_bias)
        for i in range(len(train_weights)):
            if train_weights[i].name in trainable_weights_names:
                train_weights_list.append(train_weights[i])
        grads = tape_intearction.gradient(total_loss_interactions, train_weights_list)
        self.optimizer.apply_gradients(zip(grads, train_weights_list))

    def get_main_effect_rank(self):

        sorted_index = np.array([])
        componment_scales = [0 for i in range(self.input_num)]
        main_effect_norm = [self.MainEffectBlock.subnets[i].moving_norm.numpy()[0] for i in range(self.input_num)]
        beta = (self.OutputLayer.main_effect_weights.numpy() ** 2 * np.array([main_effect_norm]).reshape([-1, 1]))
        componment_scales = (np.abs(beta) / np.sum(np.abs(beta))).reshape([-1])
        sorted_index = np.argsort(componment_scales)[::-1]
        return sorted_index, componment_scales

    def get_interaction_rank(self):

        sorted_index = np.array([])
        componment_scales = [0 for i in range(self.interact_num_added)]
        if self.interact_num_added > 0:
            interaction_norm = [self.InteractionBlock.interacts[i].moving_norm.numpy()[0] for i in range(self.interact_num_added)]
            gamma = (self.OutputLayer.interaction_weights.numpy()[:self.interact_num_added] ** 2
                  * np.array([interaction_norm]).reshape([-1, 1]))
            componment_scales = (np.abs(gamma) / np.sum(np.abs(gamma))).reshape([-1])
            sorted_index = np.argsort(componment_scales)[::-1]
        return sorted_index, componment_scales

    def get_all_active_rank(self):

        componment_scales = [0 for i in range(self.input_num + self.interact_num_added)]
        main_effect_norm = [self.MainEffectBlock.subnets[i].moving_norm.numpy()[0] for i in range(self.input_num)]
        beta = (self.OutputLayer.main_effect_weights.numpy() ** 2 * np.array([main_effect_norm]).reshape([-1, 1])
             * self.OutputLayer.main_effect_switcher.numpy())

        interaction_norm = [self.InteractionBlock.interacts[i].moving_norm.numpy()[0] for i in range(self.interact_num_added)]
        gamma = (self.OutputLayer.interaction_weights.numpy()[:self.interact_num_added] ** 2
              * np.array([interaction_norm]).reshape([-1, 1])
              * self.OutputLayer.interaction_switcher.numpy()[:self.interact_num_added])
        gamma = np.vstack([gamma, np.zeros((self.interact_num - self.interact_num_added, 1))])

        componment_coefs = np.vstack([beta, gamma])
        componment_scales = (np.abs(componment_coefs) / np.sum(np.abs(componment_coefs))).reshape([-1])
        sorted_index = np.argsort(componment_scales)[::-1]
        return sorted_index, componment_scales

    def estimate_density(self, x, sample_weight):

        n_samples = x.shape[0]
        self.data_dict_density = {}
        for indice in range(self.input_num):
            feature_name = self.feature_list_[indice]
            if indice in self.nfeature_index_list_:
                sx = self.nfeature_scaler_[feature_name]
                density, bins = np.histogram(sx.inverse_transform(x[:,[indice]]), bins=10, weights=sample_weight.reshape(-1, 1), density=True)
                self.data_dict_density.update({feature_name: {"density": {"names": bins,"scores": density}}})
            elif indice in self.cfeature_index_list_:
                unique, counts = np.unique(x[:, indice], return_counts=True)
                density = np.zeros((len(self.dummy_values_[feature_name])))
                for val in unique:
                    density[val.round().astype(int)] = np.sum((x[:, indice] == val).astype(int) * sample_weight) / sample_weight.sum()
                self.data_dict_density.update({feature_name: {"density": {"names": np.arange(len(self.dummy_values_[feature_name])),
                                                     "scores": density}}})

    def center_main_effects(self):

        output_bias = self.OutputLayer.output_bias
        main_weights = tf.multiply(self.OutputLayer.main_effect_switcher, self.OutputLayer.main_effect_weights)
        for idx, subnet in enumerate(self.MainEffectBlock.subnets):
            if idx in self.nfeature_index_list_:
                if idx in self.mono_list + self.con_list:
                    subnet_bias = subnet.lattice_layer_bias - subnet.moving_mean
                    subnet.lattice_layer_bias.assign(subnet_bias)
                else:
                    subnet_bias = subnet.output_layer.bias - subnet.moving_mean
                    subnet.output_layer.bias.assign(subnet_bias)
            elif idx in self.cfeature_index_list_:
                subnet_bias = subnet.output_layer_bias - subnet.moving_mean
                subnet.output_layer_bias.assign(subnet_bias)

            output_bias = output_bias + tf.multiply(subnet.moving_mean, tf.gather(main_weights, idx, axis=0))
        self.OutputLayer.output_bias.assign(output_bias)

    def center_interactions(self):

        output_bias = self.OutputLayer.output_bias
        interaction_weights = tf.multiply(self.OutputLayer.interaction_switcher, self.OutputLayer.interaction_weights)
        for idx, interact in enumerate(self.InteractionBlock.interacts):
            if idx >= len(self.interaction_list):
                break

            if (interact.interaction[0] in self.mono_list + self.con_list) or (interact.interaction[1] in self.mono_list + self.con_list):
                interact_bias = interact.lattice_layer_bias - interact.moving_mean
                interact.lattice_layer_bias.assign(interact_bias)
            else:
                interact_bias = interact.output_layer.bias - interact.moving_mean
                interact.output_layer.bias.assign(interact_bias)
            output_bias = output_bias + tf.multiply(interact.moving_mean, tf.gather(interaction_weights, idx, axis=0))
        self.OutputLayer.output_bias.assign(output_bias)

    def fit_main_effect(self, tr_x, tr_y, val_x, val_y, sample_weight=None):

        last_improvement = 0
        best_validation = np.inf
        train_size = tr_x.shape[0]
        tr_sw = sample_weight[self.tr_idx]
        for epoch in range(self.main_effect_epochs):
            shuffle_index = np.arange(tr_x.shape[0])
            np.random.shuffle(shuffle_index)
            tr_x = tr_x[shuffle_index]
            tr_y = tr_y[shuffle_index]
            tr_sw = tr_sw[shuffle_index]
            for iterations in range(train_size // self.batch_size):
                offset = (iterations * self.batch_size) % train_size
                batch_xx = tr_x[offset:(offset + self.batch_size), :]
                batch_yy = tr_y[offset:(offset + self.batch_size)]
                batch_sw = tr_sw[offset:(offset + self.batch_size)]
                self.train_main_effect(tf.cast(batch_xx, tf.float32), tf.cast(batch_yy, tf.float32), tf.cast(batch_sw, tf.float32))

            self.err_train_main_effect_training.append(self.evaluate(tr_x, tr_y, tr_sw,
                                                 main_effect_training=False, interaction_training=False))
            self.err_val_main_effect_training.append(self.evaluate(val_x, val_y, sample_weight[self.val_idx],
                                                 main_effect_training=False, interaction_training=False))
            if self.verbose & (epoch % 1 == 0):
                fprint(2,self.message_list, "Main effects training epoch: %d, train loss: %0.5f, val loss: %0.5f" %
                      (epoch + 1, self.err_train_main_effect_training[-1], self.err_val_main_effect_training[-1]))

            if self.err_val_main_effect_training[-1] < best_validation:
                best_validation = self.err_val_main_effect_training[-1]
                last_improvement = epoch
            if epoch - last_improvement > self.early_stop_thres1:
                if self.verbose:
                    fprint(1,self.message_list, "Early stop at epoch %d, with validation loss: %0.5f" % (epoch + 1, self.err_val_main_effect_training[-1]))
                break
        self.evaluate(tr_x, tr_y, sample_weight[self.tr_idx], main_effect_training=True, interaction_training=False)
        self.center_main_effects()

    def prune_main_effect(self, val_x, val_y, sample_weight=None):

        self.main_effect_val_loss = []
        sorted_index, componment_scales = self.get_main_effect_rank()
        self.OutputLayer.main_effect_switcher.assign(tf.constant(np.zeros((self.input_num, 1)), dtype=tf.float32))
        self.main_effect_val_loss.append(self.evaluate(val_x, val_y, sample_weight[self.val_idx],
                                        main_effect_training=False, interaction_training=False))
        for idx in range(self.input_num):
            selected_index = sorted_index[:(idx + 1)]
            main_effect_switcher = np.zeros((self.input_num, 1))
            main_effect_switcher[selected_index] = 1
            self.OutputLayer.main_effect_switcher.assign(tf.constant(main_effect_switcher, dtype=tf.float32))
            val_loss = self.evaluate(val_x, val_y, sample_weight[self.val_idx], main_effect_training=False, interaction_training=False)
            self.main_effect_val_loss.append(val_loss)

        best_idx = np.argmin(self.main_effect_val_loss)
        loss_best = np.min(self.main_effect_val_loss)
        loss_range = np.max(self.main_effect_val_loss) - np.min(self.main_effect_val_loss)
        if loss_range > 0:
            if np.sum(((self.main_effect_val_loss - loss_best) / loss_range) < self.loss_threshold) > 0:
                best_idx = np.where(((self.main_effect_val_loss - loss_best) / loss_range) < self.loss_threshold)[0][0]

        self.active_main_effect_index = sorted_index[:best_idx]
        main_effect_switcher = np.zeros((self.input_num, 1))
        main_effect_switcher[self.active_main_effect_index] = 1
        self.OutputLayer.main_effect_switcher.assign(tf.constant(main_effect_switcher, dtype=tf.float32))

    def add_interaction(self, tr_x, tr_y, val_x, val_y, sample_weight=None):

        if sample_weight is not None:
            tr_resample = np.random.choice(tr_x.shape[0], size=(tr_x.shape[0], ),
                                  p=sample_weight[self.tr_idx] / sample_weight[self.tr_idx].sum())
            tr_x = tr_x[tr_resample]
            tr_y = tr_y[tr_resample]
            val_resample = np.random.choice(val_x.shape[0], size=(val_x.shape[0], ),
                                  p=sample_weight[self.val_idx] / sample_weight[self.val_idx].sum())
            val_x = val_x[val_resample]
            val_y = val_y[val_resample]

        tr_pred = self.__call__(tf.cast(tr_x, tf.float32), sample_weight[self.tr_idx],
                        main_effect_training=False, interaction_training=False).numpy().astype(np.float64)
        val_pred = self.__call__(tf.cast(val_x, tf.float32), sample_weight[self.val_idx],
                         main_effect_training=False, interaction_training=False).numpy().astype(np.float64)
        if self.heredity:
            interaction_list_all = get_interaction_list(tr_x, val_x, tr_y.ravel(), val_y.ravel(),
                                      tr_pred.ravel(), val_pred.ravel(),
                                      self.feature_list_,
                                      self.feature_type_list_,
                                      task_type=self.task_type,
                                      active_main_effect_index=self.active_main_effect_index)
        else:
            interaction_list_all = get_interaction_list(tr_x, val_x, tr_y.ravel(), val_y.ravel(),
                          tr_pred.ravel(), val_pred.ravel(),
                          self.feature_list_,
                          self.feature_type_list_,
                          task_type=self.task_type,
                          active_main_effect_index=np.arange(self.input_num))

        self.interaction_list = list(set(self.include_interaction_list +
                             interaction_list_all[:self.interact_num - len(self.include_interaction_list)]))
        if self.interact_num - len(self.interaction_list) > 0:

            self.interaction_list = list(set(self.interaction_list +
                        interaction_list_all[self.interact_num - len(self.include_interaction_list):
                        self.interact_num - len(self.include_interaction_list) +
                        self.interact_num - len(self.interaction_list)]))

            self.interaction_list = self.interaction_list + interaction_list_all[self.interact_num:
                                     self.interact_num + self.interact_num - len(self.interaction_list)]
        self.interact_num_added = len(self.interaction_list)
        self.InteractionBlock.set_interaction_list(self.interaction_list)
        self.OutputLayer.set_interaction_list(self.interaction_list)

    def fit_interaction(self, tr_x, tr_y, val_x, val_y, sample_weight=None):

        last_improvement = 0
        best_validation = np.inf
        train_size = tr_x.shape[0]
        self.interaction_status = True
        tr_sw = sample_weight[self.tr_idx]
        for epoch in range(self.interaction_epochs):
            shuffle_index = np.arange(tr_x.shape[0])
            np.random.shuffle(shuffle_index)
            tr_x = tr_x[shuffle_index]
            tr_y = tr_y[shuffle_index]
            tr_sw = tr_sw[shuffle_index]
            for iterations in range(train_size // self.batch_size):
                offset = (iterations * self.batch_size) % train_size
                batch_xx = tr_x[offset:(offset + self.batch_size), :]
                batch_yy = tr_y[offset:(offset + self.batch_size)]
                batch_sw = tr_sw[offset:(offset + self.batch_size)]
                self.train_interaction(tf.cast(batch_xx, tf.float32), tf.cast(batch_yy, tf.float32), tf.cast(batch_sw, tf.float32))

            self.err_train_interaction_training.append(self.evaluate(tr_x, tr_y, tr_sw,
                                                 main_effect_training=False, interaction_training=False))
            self.err_val_interaction_training.append(self.evaluate(val_x, val_y, sample_weight[self.val_idx],
                                                 main_effect_training=False, interaction_training=False))
            if self.verbose & (epoch % 1 == 0):
                print("Interaction training epoch: %d, train loss: %0.5f, val loss: %0.5f" %
                      (epoch + 1, self.err_train_interaction_training[-1], self.err_val_interaction_training[-1]))

            if self.err_val_interaction_training[-1] < best_validation:
                best_validation = self.err_val_interaction_training[-1]
                last_improvement = epoch
            if epoch - last_improvement > self.early_stop_thres2:
                if self.verbose:
                    print("Early stop at epoch %d, with validation loss: %0.5f" % (epoch + 1, self.err_val_interaction_training[-1]))
                break
        self.evaluate(tr_x, tr_y, sample_weight[self.tr_idx], main_effect_training=False, interaction_training=True)
        self.center_interactions()

    def prune_interaction(self, val_x, val_y, sample_weight=None):

        self.interaction_val_loss = []
        sorted_index, componment_scales = self.get_interaction_rank()
        self.OutputLayer.interaction_switcher.assign(tf.constant(np.zeros((self.interact_num, 1)), dtype=tf.float32))
        self.interaction_val_loss.append(self.evaluate(val_x, val_y, sample_weight[self.val_idx],
                                        main_effect_training=False, interaction_training=False))
        for idx in range(self.interact_num_added):
            selected_index = sorted_index[:(idx + 1)]
            interaction_switcher = np.zeros((self.interact_num, 1))
            interaction_switcher[selected_index] = 1
            self.OutputLayer.interaction_switcher.assign(tf.constant(interaction_switcher, dtype=tf.float32))
            val_loss = self.evaluate(val_x, val_y, sample_weight[self.val_idx], main_effect_training=False, interaction_training=False)
            self.interaction_val_loss.append(val_loss)

        best_idx = np.argmin(self.interaction_val_loss)
        loss_best = np.min(self.interaction_val_loss)
        loss_range = np.max(self.interaction_val_loss) - np.min(self.interaction_val_loss)
        if loss_range > 0:
            if np.sum(((self.interaction_val_loss - loss_best) / loss_range) < self.loss_threshold) > 0:
                best_idx = np.where(((self.interaction_val_loss - loss_best) / loss_range) < self.loss_threshold)[0][0]

        self.active_interaction_index = sorted_index[:best_idx]
        interaction_switcher = np.zeros((self.interact_num, 1))
        interaction_switcher[self.active_interaction_index] = 1
        self.OutputLayer.interaction_switcher.assign(tf.constant(interaction_switcher, dtype=tf.float32))

    def fine_tune_all(self, tr_x, tr_y, val_x, val_y, sample_weight=None):

        last_improvement = 0
        best_validation = np.inf
        train_size = tr_x.shape[0]
        tr_sw = sample_weight[self.tr_idx]
        for epoch in range(self.tuning_epochs):
            shuffle_index = np.arange(train_size)
            np.random.shuffle(shuffle_index)
            tr_x = tr_x[shuffle_index]
            tr_y = tr_y[shuffle_index]
            tr_sw = tr_sw[shuffle_index]
            for iterations in range(train_size // self.batch_size):
                offset = (iterations * self.batch_size) % train_size
                batch_xx = tr_x[offset:(offset + self.batch_size), :]
                batch_yy = tr_y[offset:(offset + self.batch_size)]
                batch_sw = tr_sw[offset:(offset + self.batch_size)]
                self.train_all(tf.cast(batch_xx, tf.float32), tf.cast(batch_yy, tf.float32), tf.cast(batch_sw, tf.float32))

            self.err_train_tuning.append(self.evaluate(tr_x, tr_y, tr_sw,
                                         main_effect_training=False, interaction_training=False))
            self.err_val_tuning.append(self.evaluate(val_x, val_y, sample_weight[self.val_idx],
                                        main_effect_training=False, interaction_training=False))
            if self.verbose & (epoch % 1 == 0):
                print("Fine tuning epoch: %d, train loss: %0.5f, val loss: %0.5f" %
                      (epoch + 1, self.err_train_tuning[-1], self.err_val_tuning[-1]))

            if self.err_val_tuning[-1] < best_validation:
                best_validation = self.err_val_tuning[-1]
                last_improvement = epoch
            if epoch - last_improvement > self.early_stop_thres3:
                if self.verbose:
                    print("Early stop at epoch %d, with validation loss: %0.5f" % (epoch + 1, self.err_val_tuning[-1]))
                break
        self.evaluate(tr_x, tr_y, sample_weight[self.tr_idx], main_effect_training=True, interaction_training=True)
        self.center_main_effects()
        self.center_interactions()

    def init_fit(self, train_x, train_y, sample_weight=None):

        # initialization
        self.data_dict_density = {}
        self.err_train_main_effect_training = []
        self.err_val_main_effect_training = []
        self.err_train_interaction_training = []
        self.err_val_interaction_training = []
        self.err_train_tuning = []
        self.err_val_tuning = []

        self.interaction_list = []
        self.active_main_effect_index = []
        self.active_interaction_index = []
        self.main_effect_val_loss = []
        self.interaction_val_loss = []

        # data loading
        n_samples = train_x.shape[0]
        indices = np.arange(n_samples)
        if self.task_type == "Regression":
            tr_x, val_x, tr_y, val_y, tr_idx, val_idx = train_test_split(train_x, train_y, indices, test_size=self.val_ratio,
                                          random_state=self.random_state)
        elif self.task_type == "Classification":
            tr_x, val_x, tr_y, val_y, tr_idx, val_idx = train_test_split(train_x, train_y, indices, test_size=self.val_ratio,
                                      stratify=train_y, random_state=self.random_state)
        self.tr_idx = tr_idx
        self.val_idx = val_idx
        self.estimate_density(tr_x, sample_weight[self.tr_idx])
        return tr_x, val_x, tr_y, val_y

    def fit(self, train_x, train_y, sample_weight=None):

        n_samples = train_x.shape[0]
        if sample_weight is None:
            sample_weight = np.ones(n_samples)
        else:
            sample_weight = n_samples * sample_weight.ravel() / np.sum(sample_weight)

        tr_x, val_x, tr_y, val_y = self.init_fit(train_x, train_y, sample_weight)
        
        fprint(1,self.message_list, "#" * 20 + "GAMI-Net training start." + "#" * 20)
            
            
        # step 1: main effects
        if not self.training_main:
            return
        else:
            
            fprint(1,self.message_list, "#" * 10 + "Stage 1: main effect training start." + "#" * 10)        
            self.optimizer.learning_rate.assign(self.lr_bp[0])
            self.fit_main_effect(tr_x, tr_y, val_x, val_y, sample_weight)
            
            fprint(1,self.message_list, "#" * 10 + "Stage 1: main effect training stop." + "#" * 10)
            self.prune_main_effect(val_x, val_y, sample_weight)
            if (len(self.active_main_effect_index) == 0) and self.heredity:
                if self.verbose:
                    fprint(1,self.message_list, "#" * 10 + "No main effect is selected, training stop." + "#" * 10)
                return

        # step2: interaction
        if not self.training_interaction:
            return  
        else:        
        
            if (self.interact_num == 0) and len(self.include_interaction_list) == 0:
                if self.verbose:
                    fprint(1,self.message_list, "#" * 10 + "Max interaction is specified to zero, training stop." + "#" * 10)
                return
            
            fprint(1,self.message_list, "#" * 10 + "Stage 2: interaction training start." + "#" * 10)
            self.add_interaction(tr_x, tr_y, val_x, val_y, sample_weight)
            self.optimizer.lr.assign(self.lr_bp[1])
            self.fit_interaction(tr_x, tr_y, val_x, val_y, sample_weight)
            
            fprint(1,self.message_list, "#" * 10 + "Stage 2: interaction training stop." + "#" * 10)
            self.prune_interaction(val_x, val_y, sample_weight)

            self.optimizer.lr.assign(self.lr_bp[2])
        
        # step3: fine tune
        if not self.training_fine:
            return    
        
        else:      
            self.fine_tune_all(tr_x, tr_y, val_x, val_y, sample_weight)
            self.active_indice = 1 + np.hstack([-1, self.active_main_effect_index, self.input_num + self.active_interaction_index]).astype(int)
            self.effect_names = np.hstack(["Intercept", np.array(self.feature_list_), [self.feature_list_[self.interaction_list[i][0]] + " x "
                            + self.feature_list_[self.interaction_list[i][1]] for i in range(len(self.interaction_list))]])
            
            fprint(1,self.message_list, "#" * 20 + "GAMI-Net training finished." + "#" * 20)
       
    def summary_logs(self, save_dict=False, folder="./", name="summary_logs"):

        data_dict_log = {}
        data_dict_log.update({"err_train_main_effect_training": self.err_train_main_effect_training,
                       "err_val_main_effect_training": self.err_val_main_effect_training,
                       "err_train_interaction_training": self.err_train_interaction_training,
                       "err_val_interaction_training": self.err_val_interaction_training,
                       "err_train_tuning": self.err_train_tuning,
                       "err_val_tuning": self.err_val_tuning,
                       "interaction_list": self.interaction_list,
                       "active_main_effect_index": self.active_main_effect_index,
                       "active_interaction_index": self.active_interaction_index,
                       "main_effect_val_loss": self.main_effect_val_loss,
                       "interaction_val_loss": self.interaction_val_loss})
        
        if save_dict:
            if not os.path.exists(folder):
                os.makedirs(folder)
            save_path = folder + name
            np.save("%s.npy" % save_path, data_dict_log)

        return data_dict_log

    def global_explain(self, main_grid_size=100, interact_grid_size=100, save_dict=False, folder="./", name="global_explain"):

        # By default, we use the same main_grid_size and interact_grid_size as that of the zero mean constraint
        # Alternatively, we can also specify it manually, e.g., when we want to have the same grid size as EBM (256).        
        data_dict_global = self.data_dict_density
        sorted_index, componment_scales = self.get_all_active_rank()
        for indice in range(self.input_num):
            feature_name = self.feature_list_[indice]
            subnet = self.MainEffectBlock.subnets[indice]
            if indice in self.nfeature_index_list_:
                sx = self.nfeature_scaler_[feature_name]
                main_effect_inputs = np.linspace(0, 1, main_grid_size).reshape([-1, 1])
                main_effect_inputs_original = sx.inverse_transform(main_effect_inputs)
                main_effect_outputs = (self.OutputLayer.main_effect_weights.numpy()[indice]
                            * self.OutputLayer.main_effect_switcher.numpy()[indice]
                            * subnet.__call__(tf.cast(tf.constant(main_effect_inputs), tf.float32)).numpy())
                data_dict_global[feature_name].update({"type":"continuous",
                                      "importance":componment_scales[indice],
                                      "inputs":main_effect_inputs_original.ravel(),
                                      "outputs":main_effect_outputs.ravel()})

            elif indice in self.cfeature_index_list_:
                main_effect_inputs_original = self.dummy_values_[feature_name]
                main_effect_inputs = np.arange(len(main_effect_inputs_original)).reshape([-1, 1])
                main_effect_outputs = (self.OutputLayer.main_effect_weights.numpy()[indice]
                            * self.OutputLayer.main_effect_switcher.numpy()[indice]
                            * subnet.__call__(tf.cast(main_effect_inputs, tf.float32)).numpy())

                main_effect_input_ticks = (main_effect_inputs.ravel().astype(int) if len(main_effect_inputs_original) <= 6 else
                              np.linspace(0.1 * len(main_effect_inputs_original), len(main_effect_inputs_original) * 0.9, 4).astype(int))
                main_effect_input_labels = [main_effect_inputs_original[i] for i in main_effect_input_ticks]
                if len("".join(list(map(str, main_effect_input_labels)))) > 30:
                    main_effect_input_labels = [str(main_effect_inputs_original[i])[:4] for i in main_effect_input_ticks]

                data_dict_global[feature_name].update({"type": "categorical",
                                      "importance": componment_scales[indice],
                                      "inputs": main_effect_inputs_original,
                                      "outputs": main_effect_outputs.ravel(),
                                      "input_ticks": main_effect_input_ticks,
                                      "input_labels": main_effect_input_labels})

        for indice in range(self.interact_num_added):

            inter_net = self.InteractionBlock.interacts[indice]
            feature_name1 = self.feature_list_[self.interaction_list[indice][0]]
            feature_name2 = self.feature_list_[self.interaction_list[indice][1]]
            feature_type1 = "categorical" if feature_name1 in self.cfeature_list_ else "continuous"
            feature_type2 = "categorical" if feature_name2 in self.cfeature_list_ else "continuous"
            
            axis_extent = []
            interact_input_list = []
            if feature_name1 in self.cfeature_list_:
                interact_input1_original = self.dummy_values_[feature_name1]
                interact_input1 = np.arange(len(interact_input1_original), dtype=np.float32)
                interact_input1_ticks = (interact_input1.astype(int) if len(interact_input1) <= 6 else 
                                 np.linspace(0.1 * len(interact_input1), len(interact_input1) * 0.9, 4).astype(int))
                interact_input1_labels = [interact_input1_original[i] for i in interact_input1_ticks]
                if len("".join(list(map(str, interact_input1_labels)))) > 30:
                    interact_input1_labels = [str(interact_input1_original[i])[:4] for i in interact_input1_ticks]
                interact_input_list.append(interact_input1)
                axis_extent.extend([-0.5, len(interact_input1_original) - 0.5])
            else:
                sx1 = self.nfeature_scaler_[feature_name1]
                interact_input1 = np.array(np.linspace(0, 1, interact_grid_size), dtype=np.float32)
                interact_input1_original = sx1.inverse_transform(interact_input1.reshape([-1, 1])).ravel()
                interact_input1_ticks = []
                interact_input1_labels = []
                interact_input_list.append(interact_input1)
                axis_extent.extend([interact_input1_original.min(), interact_input1_original.max()])
            if feature_name2 in self.cfeature_list_:
                interact_input2_original = self.dummy_values_[feature_name2]
                interact_input2 = np.arange(len(interact_input2_original), dtype=np.float32)
                interact_input2_ticks = (interact_input2.astype(int) if len(interact_input2) <= 6 else
                                 np.linspace(0.1 * len(interact_input2), len(interact_input2) * 0.9, 4).astype(int))
                interact_input2_labels = [interact_input2_original[i] for i in interact_input2_ticks]
                if len("".join(list(map(str, interact_input2_labels)))) > 30:
                    interact_input2_labels = [str(interact_input2_original[i])[:4] for i in interact_input2_ticks]
                interact_input_list.append(interact_input2)
                axis_extent.extend([-0.5, len(interact_input2_original) - 0.5])
            else:
                sx2 = self.nfeature_scaler_[feature_name2]
                interact_input2 = np.array(np.linspace(0, 1, interact_grid_size), dtype=np.float32)
                interact_input2_original = sx2.inverse_transform(interact_input2.reshape([-1, 1])).ravel()
                interact_input2_ticks = []
                interact_input2_labels = []
                interact_input_list.append(interact_input2)
                axis_extent.extend([interact_input2_original.min(), interact_input2_original.max()])

            x1, x2 = np.meshgrid(interact_input_list[0], interact_input_list[1][::-1])
            input_grid = np.hstack([np.reshape(x1, [-1, 1]), np.reshape(x2, [-1, 1])])

            interact_outputs = (self.OutputLayer.interaction_weights.numpy()[indice]
                        * self.OutputLayer.interaction_switcher.numpy()[indice]
                        * inter_net.__call__(input_grid, training=False).numpy().reshape(x1.shape))
            data_dict_global.update({feature_name1 + " vs. " + feature_name2:{"type": "pairwise",
                                                       "xtype": feature_type1,
                                                       "ytype": feature_type2,
                                                       "importance": componment_scales[self.input_num + indice],
                                                       "input1": interact_input1_original,
                                                       "input2": interact_input2_original,
                                                       "outputs": interact_outputs,
                                                       "input1_ticks": interact_input1_ticks,
                                                       "input2_ticks": interact_input2_ticks,
                                                       "input1_labels": interact_input1_labels,
                                                       "input2_labels": interact_input2_labels,
                                                       "axis_extent": axis_extent}})

        if save_dict:
            if not os.path.exists(folder):
                os.makedirs(folder)
            save_path = folder + name
            np.save("%s.npy" % save_path, data_dict_global)
            
        return data_dict_global
        
    def local_explain(self, x, y=None, save_dict=False, folder="./", name="local_explain"):

        predicted = self.predict(x)
        intercept = self.OutputLayer.output_bias.numpy()

        main_effect_output = self.MainEffectBlock.__call__(tf.cast(tf.constant(x), tf.float32)).numpy()
        if self.interact_num > 0:
            interaction_output = self.InteractionBlock.__call__(tf.cast(tf.constant(x), tf.float32)).numpy()
        else:
            interaction_output = np.empty(shape=(x.shape[0], 0))

        main_effect_weights = ((self.OutputLayer.main_effect_weights.numpy()) * self.OutputLayer.main_effect_switcher.numpy()).ravel()
        interaction_weights = ((self.OutputLayer.interaction_weights.numpy()[:self.interact_num_added])
                              * self.OutputLayer.interaction_switcher.numpy()[:self.interact_num_added]).ravel()
        interaction_weights = np.hstack([interaction_weights, np.zeros((self.interact_num - self.interact_num_added))])
        scores = np.hstack([np.repeat(intercept[0], x.shape[0]).reshape(-1, 1), np.hstack([main_effect_weights, interaction_weights])
                                  * np.hstack([main_effect_output, interaction_output])])

        data_dict_local = [{"active_indice": self.active_indice,
                    "scores": scores[i],
                    "effect_names": self.effect_names,
                    "predicted": predicted[i],
                    "actual": y[i]} for i in range(x.shape[0])]

        if save_dict:
            if not os.path.exists(folder):
                os.makedirs(folder)
            save_path = folder + name
            np.save("%s.npy" % save_path, data_dict_local)

        return data_dict_local
    
    def load(self, folder="./", name="model"):
        
        save_path = folder + name + ".pickle"
        if not os.path.exists(save_path):
            raise "file not found!"

        with open(save_path, "rb") as input_file:
            model_dict = pickle.load(input_file)
        for key, item in model_dict.items():
            setattr(self, key, item)
        self.optimizer.lr = model_dict["lr_bp"][0]

    def save(self, folder="./", name="model"):

        self.__call__(np.random.uniform(0, 1, size=(1, len(self.meta_info) - 1)))

        model_dict = {}
        model_dict["meta_info"] = self.meta_info
        model_dict["subnet_arch"] = self.subnet_arch
        model_dict["interact_arch"] = self.interact_arch

        model_dict["lr_bp"] = self.lr_bp
        model_dict["batch_size"] = self.batch_size
        model_dict["task_type"] = self.task_type
        model_dict["activation_func"] = self.activation_func
        model_dict["tuning_epochs"] = self.tuning_epochs
        model_dict["main_effect_epochs"] = self.main_effect_epochs
        model_dict["interaction_epochs"] = self.interaction_epochs
        model_dict["early_stop_thres"] = self.early_stop_thres

        model_dict["heredity"] = self.heredity
        model_dict["reg_clarity"] = self.reg_clarity
        model_dict["loss_threshold"] = self.loss_threshold

        model_dict["mono_increasing_list"] = self.mono_increasing_list
        model_dict["mono_decreasing_list"] = self.mono_decreasing_list
        model_dict["lattice_size"] = self.lattice_size

        model_dict["verbose"] = self.verbose
        model_dict["val_ratio"]= self.val_ratio
        model_dict["random_state"] = self.random_state

        model_dict["dummy_values_"] = self.dummy_values_ 
        model_dict["nfeature_scaler_"] = self.nfeature_scaler_
        model_dict["cfeature_num_"] = self.cfeature_num_
        model_dict["nfeature_num_"] = self.nfeature_num_
        model_dict["feature_list_"] = self.feature_list_
        model_dict["cfeature_list_"] = self.cfeature_list_
        model_dict["nfeature_list_"] = self.nfeature_list_
        model_dict["feature_type_list_"] = self.feature_type_list_
        model_dict["cfeature_index_list_"] = self.cfeature_index_list_
        model_dict["nfeature_index_list_"] = self.nfeature_index_list_

        model_dict["interaction_list"] = self.interaction_list
        model_dict["interact_num_added"] = self.interact_num_added 
        model_dict["interaction_status"] = self.interaction_status
        model_dict["input_num"] = self.input_num
        model_dict["max_interact_num"] = self.max_interact_num
        model_dict["interact_num"] = self.interact_num

        model_dict["maineffect_blocks"] = self.MainEffectBlock
        model_dict["interact_blocks"] = self.InteractionBlock
        model_dict["output_layer"] = self.OutputLayer
        model_dict["loss_fn"] = self.loss_fn

        model_dict["clarity_loss"] = self.clarity_loss
        model_dict["data_dict_density"] = self.data_dict_density

        model_dict["err_train_main_effect_training"] = self.err_train_main_effect_training
        model_dict["err_val_main_effect_training"] = self.err_val_main_effect_training
        model_dict["err_train_interaction_training"] = self.err_train_interaction_training
        model_dict["err_val_interaction_training"] = self.err_val_interaction_training
        model_dict["err_train_tuning"] = self.err_train_tuning
        model_dict["err_val_tuning"] = self.err_val_tuning
        model_dict["interaction_list"] = self.interaction_list
        model_dict["main_effect_val_loss"] = self.main_effect_val_loss
        model_dict["interaction_val_loss"] = self.interaction_val_loss

        model_dict["active_indice"] = self.active_indice
        model_dict["effect_names"] = self.effect_names
        model_dict["active_main_effect_index"] = self.active_main_effect_index
        model_dict["active_interaction_index"] = self.active_interaction_index

        model_dict["tr_idx"] = self.tr_idx
        model_dict["val_idx"] = self.val_idx
        
        if not os.path.exists(folder):
            os.makedirs(folder)
        save_path = folder + name + ".pickle"
        with open(save_path, 'wb') as handle:
            pickle.dump(model_dict, handle)

# All the codes in this file are derived from interpret package by Microsoft Corporation
# Distributed under the MIT software license

def gen_attributes(col_types, col_n_bins):
    # Create Python form of attributes
    # Undocumented.
    attributes = [None] * len(col_types)
    for col_idx, _ in enumerate(attributes):
        attributes[col_idx] = {
            # NOTE: Ordinal only handled at native, override.
            # 'type': col_types[col_idx],
            "type": "continuous",
            # NOTE: Missing not implemented at native, always set to false.
            "has_missing": False,
            "n_bins": col_n_bins[col_idx],
        }
    return attributes

def gen_attribute_sets(attribute_indices):
    attribute_sets = [None] * len(attribute_indices)
    for i, indices in enumerate(attribute_indices):
        attribute_set = {"n_attributes": len(indices), "attributes": indices}
        attribute_sets[i] = attribute_set
    return attribute_sets

def autogen_schema(X, ordinal_max_items=2, feature_names=None, feature_types=None):
    """ Generates data schema for a given dataset as JSON representable.
    Args:
        X: Dataframe/ndarray to build schema from.
        ordinal_max_items: If a numeric column's cardinality
            is at most this integer,
            consider it as ordinal instead of continuous.
        feature_names: Feature names
        feature_types: Feature types
    Returns:
        A dictionary - schema that encapsulates column information,
        such as type and domain.
    """
    col_number = 0
    schema = OrderedDict()
    for idx, (name, col_dtype) in enumerate(zip(X.dtypes.index, X.dtypes)):
        schema[name] = {}
        if feature_types is not None:
            schema[name]["type"] = feature_types[idx]
        else:
            if is_numeric_dtype(col_dtype):
                if len(set(X[name])) > ordinal_max_items:
                    schema[name]["type"] = "continuous"
                else:
                    # TODO: Work with ordinal later.
                    schema[name]["type"] = "categorical"
                    # schema[name]['type'] = 'ordinal'
                    # schema[name]['order'] = list(set(X[name]))
            elif is_string_dtype(col_dtype):
                schema[name]["type"] = "categorical"
            else:  # pragma: no cover
                warnings.warn("Unknown column: " + name, RuntimeWarning)
                schema[name]["type"] = "unknown"
        schema[name]["column_number"] = col_number
        col_number += 1
    return schema

class EBMPreprocessor(BaseEstimator, TransformerMixin):
    """ Transformer that preprocesses data to be ready before EBM. """

    def __init__(
        self,
        schema=None,
        max_n_bins=255,
        missing_constant=0,
        unknown_constant=0,
        feature_names=None,
        binning_strategy="uniform",
    ):
        """ Initializes EBM preprocessor.
        Args:
            schema: A dictionary that encapsulates column information,
                    such as type and domain.
            max_n_bins: Max number of bins to process numeric features.
            missing_constant: Missing encoded as this constant.
            unknown_constant: Unknown encoded as this constant.
            feature_names: Feature names as list.
            binning_strategy: Strategy to compute bins according to density if "quantile" or equidistant if "uniform".
        """
        self.schema = schema
        self.max_n_bins = max_n_bins
        self.missing_constant = missing_constant
        self.unknown_constant = unknown_constant
        self.feature_names = feature_names
        self.binning_strategy = binning_strategy

    def fit(self, X):
        """ Fits transformer to provided instances.
        Args:
            X: Numpy array for training instances.
        Returns:
            Itself.
        """
        # self.col_bin_counts_ = {}
        self.col_bin_edges_ = {}

        self.hist_counts_ = {}
        self.hist_edges_ = {}

        self.col_mapping_ = {}
        self.col_mapping_counts_ = {}

        self.col_n_bins_ = {}

        self.col_names_ = []
        self.col_types_ = []
        self.has_fitted_ = False

        self.schema_ = (
            self.schema
            if self.schema is not None
            else autogen_schema(X, feature_names=self.feature_names)
        )
        schema = self.schema_

        for col_idx in range(X.shape[1]):
            col_name = list(schema.keys())[col_idx]
            self.col_names_.append(col_name)

            col_info = schema[col_name]
            assert col_info["column_number"] == col_idx
            col_data = X[:, col_idx]

            self.col_types_.append(col_info["type"])
            if col_info["type"] == "continuous":
                col_data = col_data.astype(float)
                col_data = col_data[~np.isnan(col_data)]

                iteration = 0
                uniq_vals = set()
                batch_size = 1000
                small_unival = True
                while True:
                    start = iteration * batch_size
                    end = (iteration + 1) * batch_size
                    uniq_vals.update(set(col_data[start:end]))
                    iteration += 1
                    if len(uniq_vals) >= self.max_n_bins:
                        small_unival = False
                        break
                    if end >= col_data.shape[0]:
                        break

                if small_unival:
                    bins = list(sorted(uniq_vals))
                else:
                    if self.binning_strategy == "uniform":
                        bins = self.max_n_bins
                    elif self.binning_strategy == "quantile":
                        bins = np.unique(
                            np.quantile(
                                col_data, q=np.linspace(0, 1, self.max_n_bins + 1)
                            )
                        )
                    else:
                        raise ValueError(
                            "Unknown binning_strategy: '{}'.".format(
                                self.binning_strategy
                            )
                        )

                _, bin_edges = np.histogram(col_data, bins=bins)

                # hist_counts, hist_edges = np.histogram(col_data, bins="doane")
                self.col_bin_edges_[col_idx] = bin_edges

                # self.hist_edges_[col_idx] = hist_edges
                # self.hist_counts_[col_idx] = hist_counts
                self.col_n_bins_[col_idx] = len(bin_edges)
            elif col_info["type"] == "ordinal":
                mapping = {val: indx for indx, val in enumerate(col_info["order"])}
                self.col_mapping_[col_idx] = mapping
                self.col_n_bins_[col_idx] = len(col_info["order"])
            elif col_info["type"] == "categorical":
                uniq_vals, counts = np.unique(col_data, return_counts=True)

                non_nan_index = ~np.isnan(counts)
                uniq_vals = uniq_vals[non_nan_index]
                counts = counts[non_nan_index]

                mapping = {val: indx for indx, val in enumerate(uniq_vals)}
                self.col_mapping_counts_[col_idx] = counts
                self.col_mapping_[col_idx] = mapping

                # TODO: Review NA as we don't support it yet.
                self.col_n_bins_[col_idx] = len(uniq_vals)

        self.has_fitted_ = True
        return self

    def transform(self, X):
        """ Transform on provided instances.
        Args:
            X: Numpy array for instances.
        Returns:
            Transformed numpy array.
        """
        check_is_fitted(self, "has_fitted_")

        schema = self.schema
        X_new = np.copy(X)
        for col_idx in range(X.shape[1]):
            col_info = schema[list(schema.keys())[col_idx]]
            assert col_info["column_number"] == col_idx
            col_data = X[:, col_idx]
            if col_info["type"] == "continuous":
                col_data = col_data.astype(float)
                bin_edges = self.col_bin_edges_[col_idx].copy()

                digitized = np.digitize(col_data, bin_edges, right=False)
                digitized[digitized == 0] = 1
                digitized -= 1

                # NOTE: NA handling done later.
                # digitized[np.isnan(col_data)] = self.missing_constant
                X_new[:, col_idx] = digitized
            elif col_info["type"] == "ordinal":
                mapping = self.col_mapping_[col_idx]
                mapping[np.nan] = self.missing_constant
                vec_map = np.vectorize(
                    lambda x: mapping[x] if x in mapping else self.unknown_constant
                )
                X_new[:, col_idx] = vec_map(col_data)
            elif col_info["type"] == "categorical":
                mapping = self.col_mapping_[col_idx]
                mapping[np.nan] = self.missing_constant
                vec_map = np.vectorize(
                    lambda x: mapping[x] if x in mapping else self.unknown_constant
                )
                X_new[:, col_idx] = vec_map(col_data)

        return X_new.astype(np.int64)
    
class Native:
    """Layer/Class responsible for native function calls."""

    # enum FeatureType : int64_t
    # Ordinal = 0
    FeatureTypeOrdinal = 0
    # Nominal = 1
    FeatureTypeNominal = 1

    class EbmCoreFeature(ct.Structure):
        _fields_ = [
            # FeatureType featureType;
            ("featureType", ct.c_longlong),
            # int64_t hasMissing;
            ("hasMissing", ct.c_longlong),
            # int64_t countBins;
            ("countBins", ct.c_longlong),
        ]

    class EbmCoreFeatureCombination(ct.Structure):
        _fields_ = [
            # int64_t countFeaturesInCombination;
            ("countFeaturesInCombination", ct.c_longlong)
        ]

    LogFuncType = ct.CFUNCTYPE(None, ct.c_char, ct.c_char_p)

    # const signed char TraceLevelOff = 0;
    TraceLevelOff = 0
    # const signed char TraceLevelError = 1;
    TraceLevelError = 1
    # const signed char TraceLevelWarning = 2;
    TraceLevelWarning = 2
    # const signed char TraceLevelInfo = 3;
    TraceLevelInfo = 3
    # const signed char TraceLevelVerbose = 4;
    TraceLevelVerbose = 4

    def __init__(self):

        self.lib = ct.cdll.LoadLibrary(self.get_ebm_lib_path())
        self.harden_function_signatures()

    def harden_function_signatures(self):
        """ Adds types to function signatures. """

        self.lib.InitializeInteractionClassification.argtypes = [
            # int64_t countFeatures
            ct.c_longlong,
            # EbmCoreFeature * features
            ct.POINTER(self.EbmCoreFeature),
            # int64_t countTargetClasses
            ct.c_longlong,
            # int64_t countInstances
            ct.c_longlong,
            # int64_t * targets
            ndpointer(dtype=ct.c_longlong, flags="F_CONTIGUOUS", ndim=1),
            # int64_t * binnedData
            ndpointer(dtype=ct.c_longlong, flags="F_CONTIGUOUS", ndim=2),
            # double * predictorScores
            ndpointer(dtype=ct.c_double, flags="F_CONTIGUOUS", ndim=1),
        ]
        self.lib.InitializeInteractionClassification.restype = ct.c_void_p

        self.lib.InitializeInteractionRegression.argtypes = [
            # int64_t countFeatures
            ct.c_longlong,
            # EbmCoreFeature * features
            ct.POINTER(self.EbmCoreFeature),
            # int64_t countInstances
            ct.c_longlong,
            # double * targets
            ndpointer(dtype=ct.c_double, flags="F_CONTIGUOUS", ndim=1),
            # int64_t * binnedData
            ndpointer(dtype=ct.c_longlong, flags="F_CONTIGUOUS", ndim=2),
            # double * predictorScores
            ndpointer(dtype=ct.c_double, flags="F_CONTIGUOUS", ndim=1),
        ]
        self.lib.InitializeInteractionRegression.restype = ct.c_void_p

        self.lib.GetInteractionScore.argtypes = [
            # void * ebmInteraction
            ct.c_void_p,
            # int64_t countFeaturesInCombination
            ct.c_longlong,
            # int64_t * featureIndexes
            ndpointer(dtype=ct.c_longlong, flags="F_CONTIGUOUS", ndim=1),
            # double * interactionScoreReturn
            ct.POINTER(ct.c_double),
        ]
        self.lib.GetInteractionScore.restype = ct.c_longlong

        self.lib.FreeInteraction.argtypes = [
            # void * ebmInteraction
            ct.c_void_p
        ]

    def get_ebm_lib_path(self):
        """ Returns filepath of core EBM library.
        Returns:
            A string representing filepath.
        """
        bitsize = struct.calcsize("P") * 8
        is_64_bit = bitsize == 64

        script_path = os.path.dirname(os.path.abspath(__file__))
        package_path = script_path # os.path.join(script_path, "..", "..")

        debug_str = ""
        if platform == "linux" or platform == "linux2" and is_64_bit:
            return os.path.join(
                package_path, "lib", "lib_ebmcore_linux_x64{0}.so".format(debug_str)
            )
        elif platform == "win32" and is_64_bit:
            return os.path.join(
                package_path, "lib", "lib_ebmcore_win_x64{0}.dll".format(debug_str)
            )
        elif platform == "darwin" and is_64_bit:
            return os.path.join(
                package_path, "lib", "lib_ebmcore_mac_x64{0}.dylib".format(debug_str)
            )
        else:
            msg = "Platform {0} at {1} bit not supported for EBM".format(
                platform, bitsize
            )
            raise Exception(msg)

class NativeEBM:
    """Lightweight wrapper for EBM C code.
    """

    def __init__(
        self,
        attributes,
        attribute_sets,
        X_train,
        y_train,
        X_val,
        y_val,
        model_type="regression",
        num_inner_bags=0,
        num_classification_states=2,
        training_scores=None,
        validation_scores=None,
        random_state=1337,
    ):

        # TODO: Update documentation for training/val scores args.
        """ Initializes internal wrapper for EBM C code.
        Args:
            attributes: List of attributes represented individually as
                dictionary of keys ('type', 'has_missing', 'n_bins').
            attribute_sets: List of attribute sets represented as
                a dictionary of keys ('n_attributes', 'attributes')
            X_train: Training design matrix as 2-D ndarray.
            y_train: Training response as 1-D ndarray.
            X_val: Validation design matrix as 2-D ndarray.
            y_val: Validation response as 1-D ndarray.
            model_type: 'regression'/'classification'.
            num_inner_bags: Per feature training step, number of inner bags.
            num_classification_states: Specific to classification,
                number of unique classes.
            training_scores: Undocumented.
            validation_scores: Undocumented.
            random_state: Random seed as integer.
        """
        if not hasattr(self, "native"):
            self.native = Native()

        # Store args
        self.attributes = attributes
        self.attribute_sets = attribute_sets
        self.attribute_array, self.attribute_sets_array, self.attribute_set_indexes = self._convert_attribute_info_to_c(
            attributes, attribute_sets
        )

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.model_type = model_type
        self.num_inner_bags = num_inner_bags
        self.num_classification_states = num_classification_states

        # # Set train/val scores to zeros if not passed.
        # if isinstance(intercept, numbers.Number) or len(intercept) == 1:
        #     score_vector = np.zeros(X.shape[0])
        #     else:
        # score_vector = np.zeros((X.shape[0], len(intercept)))

        self.training_scores = training_scores
        self.validation_scores = validation_scores
        if self.training_scores is None:
            if self.num_classification_states > 2:
                self.training_scores = np.zeros(
                    (X_train.shape[0], self.num_classification_states)
                ).reshape(-1)
            else:
                self.training_scores = np.zeros(X_train.shape[0])
        if self.validation_scores is None:
            if self.num_classification_states > 2:
                self.validation_scores = np.zeros(
                    (X_train.shape[0], self.num_classification_states)
                ).reshape(-1)
            else:
                self.validation_scores = np.zeros(X_train.shape[0])
        self.random_state = random_state

        # Convert n-dim arrays ready for C.
        self.X_train_f = np.asfortranarray(self.X_train)
        self.X_val_f = np.asfortranarray(self.X_val)

        # Define extra properties
        self.model_pointer = None
        self.interaction_pointer = None

        # Allocate external resources
        if self.model_type == "regression":
            self.y_train = self.y_train.astype("float64")
            self.y_val = self.y_val.astype("float64")
            self._initialize_interaction_regression()
        elif self.model_type == "classification":
            self.y_train = self.y_train.astype("int64")
            self.y_val = self.y_val.astype("int64")
            self._initialize_interaction_classification()

    def _convert_attribute_info_to_c(self, attributes, attribute_sets):
        # Create C form of attributes
        attribute_ar = (self.native.EbmCoreFeature * len(attributes))()
        for idx, attribute in enumerate(attributes):
            if attribute["type"] == "categorical":
                attribute_ar[idx].featureType = self.native.FeatureTypeNominal
            else:
                attribute_ar[idx].featureType = self.native.FeatureTypeOrdinal
            attribute_ar[idx].hasMissing = 1 * attribute["has_missing"]
            attribute_ar[idx].countBins = attribute["n_bins"]

        attribute_set_indexes = []
        attribute_sets_ar = (
            self.native.EbmCoreFeatureCombination * len(attribute_sets)
        )()
        for idx, attribute_set in enumerate(attribute_sets):
            attribute_sets_ar[idx].countFeaturesInCombination = attribute_set[
                "n_attributes"
            ]

            for attr_idx in attribute_set["attributes"]:
                attribute_set_indexes.append(attr_idx)

        attribute_set_indexes = np.array(attribute_set_indexes, dtype="int64")

        return attribute_ar, attribute_sets_ar, attribute_set_indexes

    def _initialize_interaction_regression(self):
        self.interaction_pointer = self.native.lib.InitializeInteractionRegression(
            len(self.attribute_array),
            self.attribute_array,
            self.X_train.shape[0],
            self.y_train,
            self.X_train_f,
            self.training_scores,
        )

    def _initialize_interaction_classification(self):
        self.interaction_pointer = self.native.lib.InitializeInteractionClassification(
            len(self.attribute_array),
            self.attribute_array,
            self.num_classification_states,
            self.X_train.shape[0],
            self.y_train,
            self.X_train_f,
            self.training_scores,
        )

    def close(self):
        """ Deallocates C objects used to train EBM. """
        self.native.lib.FreeInteraction(self.interaction_pointer)

    def fast_interaction_score(self, attribute_index_tuple):
        """ Provides score for an attribute interaction. Higher is better."""
        score = ct.c_double(0.0)
        self.native.lib.GetInteractionScore(
            self.interaction_pointer,
            len(attribute_index_tuple),
            np.array(attribute_index_tuple, dtype=np.int64),
            ct.byref(score),
        )
        return score.value

def get_interaction_list(tr_x, val_x, tr_y, val_y, pred_tr, pred_val, feature_list, feature_type_list, 
                 active_main_effect_index, task_type="Regression", n_jobs=1):

    if task_type == "Regression":
        num_classes_ = -1
        model_type = "regression"
    elif task_type == "Classification":
        num_classes_ = 2
        model_type = "classification"
        pred_tr = np.minimum(np.maximum(pred_tr, 0.0000001), 0.9999999)
        pred_val = np.minimum(np.maximum(pred_val, 0.0000001), 0.9999999)
        pred_tr = np.log(pred_tr / (1 - pred_tr))
        pred_val = np.log(pred_val / (1 - pred_val))

    train_num = tr_x.shape[0]
    x = np.vstack([tr_x, val_x])
    schema_ = autogen_schema(pd.DataFrame(x), feature_names=feature_list, feature_types=feature_type_list)
    preprocessor_ = EBMPreprocessor(schema=schema_)
    preprocessor_.fit(x)
    xt = preprocessor_.transform(x)

    tr_x, val_x = xt[:train_num, :], xt[train_num:, :]
    attributes_ = gen_attributes(preprocessor_.col_types_, preprocessor_.col_n_bins_)
    main_attr_sets = gen_attribute_sets([[item] for item in range(len(attributes_))])

    with closing(
        NativeEBM(
            attributes_,
            main_attr_sets,
            tr_x,
            tr_y,
            val_x,
            val_y,
            num_inner_bags=0,
            num_classification_states=num_classes_,
            model_type=model_type,
            training_scores=pred_tr,
            validation_scores=pred_val,
        )
    ) as native_ebm:

        def evaluate_parallel(pair):
            return pair, native_ebm.fast_interaction_score(pair)

        all_pairs = [pair for pair in combinations(range(len(preprocessor_.col_types_)), 2)
                   if (pair[0] in active_main_effect_index) or (pair[1] in active_main_effect_index)]
        interaction_scores = Parallel(n_jobs=n_jobs, backend="threading")(delayed(evaluate_parallel)(pair) for pair in all_pairs)
    
    ranked_scores = list(sorted(interaction_scores, key=lambda item: item[1], reverse=True))
    interaction_list = [ranked_scores[i][0] for i in range(len(ranked_scores))]
    return interaction_list

def plot_regularization(data_dict_logs, log_scale=True, save_eps=False, save_png=False, folder="./results/", name="demo"):

    main_loss = data_dict_logs["main_effect_val_loss"]
    inter_loss = data_dict_logs["interaction_val_loss"]
    active_main_effect_index = data_dict_logs["active_main_effect_index"]
    active_interaction_index = data_dict_logs["active_interaction_index"]

    fig = plt.figure(figsize=(14, 4))
    if len(main_loss) > 0:
        ax1 = plt.subplot(1, 2, 1)
        ax1.plot(np.arange(0, len(main_loss), 1), main_loss)
        ax1.axvline(np.argmin(main_loss), linestyle="dotted", color="red")
        ax1.axvline(len(active_main_effect_index), linestyle="dotted", color="red")
        ax1.plot(np.argmin(main_loss), np.min(main_loss), "*", markersize=12, color="red")
        ax1.plot(len(active_main_effect_index), main_loss[len(active_main_effect_index)], "o", markersize=8, color="red")
        ax1.set_xlabel("Number of Main Effects", fontsize=12)
        ax1.set_xlim(-0.5, len(main_loss) - 0.5)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        if log_scale:
            ax1.set_yscale("log")
            ax1.set_yticks((10 ** np.linspace(np.log10(np.nanmin(main_loss)), np.log10(np.nanmax(main_loss)), 5)).round(5))
            ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax1.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax1.set_ylabel("Validation Loss (Log Scale)", fontsize=12)
        else:
            ax1.set_yticks((np.linspace(np.nanmin(main_loss), np.nanmax(main_loss), 5)).round(5))
            ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax1.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax1.set_ylabel("Validation Loss", fontsize=12)

    if len(inter_loss) > 0:
        ax2 = plt.subplot(1, 2, 2)
        ax2.plot(np.arange(0, len(inter_loss), 1), inter_loss)
        ax2.axvline(np.argmin(inter_loss), linestyle="dotted", color="red")
        ax2.axvline(len(active_interaction_index), linestyle="dotted", color="red")
        ax2.plot(np.argmin(inter_loss), np.min(inter_loss), "*", markersize=12, color="red")
        ax2.plot(len(active_interaction_index), inter_loss[len(active_interaction_index)], "o", markersize=8, color="red")
        ax2.set_xlabel("Number of Interactions", fontsize=12)
        ax2.set_xlim(-0.5, len(inter_loss) - 0.5)
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        if log_scale:
            ax2.set_yscale("log")
            ax2.set_yticks((10 ** np.linspace(np.log10(np.nanmin(inter_loss)), np.log10(np.nanmax(inter_loss)), 5)).round(5))
            ax2.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax2.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax2.set_ylabel("Validation Loss (Log Scale)", fontsize=12)
        else:
            ax2.set_yticks((np.linspace(np.nanmin(inter_loss), np.nanmax(inter_loss), 5)).round(5))
            ax2.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax2.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax2.set_ylabel("Validation Loss", fontsize=12)
    plt.show()

    save_path = folder + name
    if save_eps:
        if not os.path.exists(folder):
            os.makedirs(folder)
        fig.savefig("%s.eps" % save_path, bbox_inches="tight", dpi=100)
    if save_png:
        if not os.path.exists(folder):
            os.makedirs(folder)
        fig.savefig("%s.png" % save_path, bbox_inches="tight", dpi=100)

def plot_trajectory(data_dict_logs, log_scale=True, save_eps=False, save_png=False, folder="./results/", name="demo"):

    t1, t2, t3 = [data_dict_logs["err_train_main_effect_training"],
              data_dict_logs["err_train_interaction_training"], data_dict_logs["err_train_tuning"]]
    v1, v2, v3= [data_dict_logs["err_val_main_effect_training"],
              data_dict_logs["err_val_interaction_training"], data_dict_logs["err_val_tuning"]]

    fig = plt.figure(figsize=(14, 4))
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(np.arange(1, len(t1) + 1, 1), t1, color="r")
    ax1.plot(np.arange(len(t1) + 1, len(t1 + t2) + 1, 1), t2, color="b")
    ax1.plot(np.arange(len(t1 + t2) + 1, len(t1 + t2 + t3) + 1, 1), t3, color="y")
    if log_scale:
        ax1.set_yscale("log")
        ax1.set_yticks((10 ** np.linspace(np.log10(np.nanmin(t1 + t2 + t3)), np.log10(np.nanmax(t1 + t2 + t3)), 5)).round(5))
        ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax1.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax1.set_xlabel("Number of Epochs", fontsize=12)
        ax1.set_ylabel("Training Loss (Log Scale)", fontsize=12)
    else:
        ax1.set_yticks((np.linspace(np.nanmin(t1 + t2), np.nanmax(t1 + t2), 5)).round(5))
        ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax1.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax1.set_xlabel("Number of Epochs", fontsize=12)
        ax1.set_ylabel("Training Loss", fontsize=12)

    ax1.legend(["Stage 1: Training Main Effects", "Stage 2: Training Interactions", "Stage 3: Fine Tuning"])

    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(np.arange(1, len(v1) + 1, 1), v1, color="r")
    ax2.plot(np.arange(len(v1) + 1, len(v1 + v2) + 1, 1), v2, color="b")
    ax2.plot(np.arange(len(v1 + v2) + 1, len(v1 + v2 + v3) + 1, 1), v3, color="y")
    if log_scale:
        ax2.set_yscale("log")
        ax2.set_yticks((10 ** np.linspace(np.log10(np.nanmin(v1 + v2 + v3)), np.log10(np.nanmax(v1 + v2 + v3)), 5)).round(5))
        ax2.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax2.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax2.set_xlabel("Number of Epochs", fontsize=12)
        ax2.set_ylabel("Validation Loss (Log Scale)", fontsize=12)
    else:
        ax2.set_yticks((np.linspace(np.nanmin(v1 + v2 + v3), np.nanmax(v1 + v2 + v3), 5)).round(5))
        ax2.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax2.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax2.set_xlabel("Number of Epochs", fontsize=12)
        ax2.set_ylabel("Validation Loss", fontsize=12)
    ax2.legend(["Stage 1: Training Main Effects", "Stage 2: Training Interactions", "Stage 3: Fine Tuning"])
    plt.show()

    save_path = folder + name
    if save_eps:
        if not os.path.exists(folder):
            os.makedirs(folder)
        fig.savefig("%s.eps" % save_path, bbox_inches="tight", dpi=100)
    if save_png:
        if not os.path.exists(folder):
            os.makedirs(folder)
        fig.savefig("%s.png" % save_path, bbox_inches="tight", dpi=100)

def feature_importance_visualize(data_dict_global, folder="./results/", name="demo", save_png=False, save_eps=False):

    all_ir = []
    all_names = []
    for key, item in data_dict_global.items():
        if item["importance"] > 0:
            all_ir.append(item["importance"])
            all_names.append(key)

    max_ids = len(all_names)
    if max_ids > 0:
        fig = plt.figure(figsize=(0.4 + 0.6 * max_ids, 4))
        ax = plt.axes()
        ax.bar(np.arange(len(all_ir)), [ir for ir, _ in sorted(zip(all_ir, all_names))][::-1])
        ax.set_xticks(np.arange(len(all_ir)))
        ax.set_xticklabels([name for _, name in sorted(zip(all_ir, all_names))][::-1], rotation=60)
        plt.xlabel("Feature Name", fontsize=12)
        plt.ylim(0, np.max(all_ir) + 0.05)
        plt.xlim(-1, len(all_names))
        plt.title("Feature Importance")

        save_path = folder + name
        if save_eps:
            if not os.path.exists(folder):
                os.makedirs(folder)
            fig.savefig("%s.eps" % save_path, bbox_inches="tight", dpi=100)
        if save_png:
            if not os.path.exists(folder):
                os.makedirs(folder)
            fig.savefig("%s.png" % save_path, bbox_inches="tight", dpi=100)

def global_visualize_density(data_dict_global, main_effect_num=None, interaction_num=None, cols_per_row=4,
                        save_png=False, save_eps=False, folder="./results/", name="demo"):

    maineffect_count = 0
    componment_scales = []
    for key, item in data_dict_global.items():
        componment_scales.append(item["importance"])
        if item["type"] != "pairwise":
            maineffect_count += 1

    componment_scales = np.array(componment_scales)
    sorted_index = np.argsort(componment_scales)
    active_index = sorted_index[componment_scales[sorted_index].cumsum() > 0][::-1]
    active_univariate_index = active_index[active_index < maineffect_count][:main_effect_num]
    active_interaction_index = active_index[active_index >= maineffect_count][:interaction_num]
    max_ids = len(active_univariate_index) + len(active_interaction_index)

    idx = 0
    fig = plt.figure(figsize=(6 * cols_per_row, 4.6 * int(np.ceil(max_ids / cols_per_row))))
    outer = gridspec.GridSpec(int(np.ceil(max_ids / cols_per_row)), cols_per_row, wspace=0.25, hspace=0.35)
    for indice in active_univariate_index:

        feature_name = list(data_dict_global.keys())[indice]
        if data_dict_global[feature_name]["type"] == "continuous":

            inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[idx], wspace=0.1, hspace=0.1, height_ratios=[6, 1])
            ax1 = plt.Subplot(fig, inner[0])
            ax1.plot(data_dict_global[feature_name]["inputs"], data_dict_global[feature_name]["outputs"])
            ax1.set_xticklabels([])
            fig.add_subplot(ax1)

            ax2 = plt.Subplot(fig, inner[1])
            xint = ((np.array(data_dict_global[feature_name]["density"]["names"][1:])
                            + np.array(data_dict_global[feature_name]["density"]["names"][:-1])) / 2).reshape([-1, 1]).reshape([-1])
            ax2.bar(xint, data_dict_global[feature_name]["density"]["scores"], width=xint[1] - xint[0])
            ax2.get_shared_x_axes().join(ax1, ax2)
            ax2.set_yticklabels([])
            ax2.autoscale()
            fig.add_subplot(ax2)

        elif data_dict_global[feature_name]["type"] == "categorical":

            inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[idx],
                                        wspace=0.1, hspace=0.1, height_ratios=[6, 1])
            ax1 = plt.Subplot(fig, inner[0])
            ax1.bar(np.arange(len(data_dict_global[feature_name]["inputs"])),
                        data_dict_global[feature_name]["outputs"])
            ax1.set_xticklabels([])
            fig.add_subplot(ax1)

            ax2 = plt.Subplot(fig, inner[1])
            ax2.bar(np.arange(len(data_dict_global[feature_name]["density"]["names"])),
                    data_dict_global[feature_name]["density"]["scores"])
            ax2.get_shared_x_axes().join(ax1, ax2)
            ax2.autoscale()
            ax2.set_xticks(data_dict_global[feature_name]["input_ticks"])
            ax2.set_xticklabels(data_dict_global[feature_name]["input_labels"])
            ax2.set_yticklabels([])
            fig.add_subplot(ax2)

        idx = idx + 1
        if len(str(ax2.get_xticks())) > 60:
            ax2.xaxis.set_tick_params(rotation=20)
        ax1.set_title(feature_name + " (" + str(np.round(100 * data_dict_global[feature_name]["importance"], 1)) + "%)", fontsize=12)

    for indice in active_interaction_index:

        feature_name = list(data_dict_global.keys())[indice]
        feature_name1 = feature_name.split(" vs. ")[0]
        feature_name2 = feature_name.split(" vs. ")[1]
        axis_extent = data_dict_global[feature_name]["axis_extent"]

        inner = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=outer[idx],
                                wspace=0.1, hspace=0.1, height_ratios=[6, 1], width_ratios=[0.6, 3, 0.15, 0.2])
        ax_main = plt.Subplot(fig, inner[1])
        interact_plot = ax_main.imshow(data_dict_global[feature_name]["outputs"], interpolation="nearest",
                             aspect="auto", extent=axis_extent)
        ax_main.set_xticklabels([])
        ax_main.set_yticklabels([])
        ax_main.set_title(feature_name + " (" + str(np.round(100 * data_dict_global[feature_name]["importance"], 1)) + "%)", fontsize=12)
        fig.add_subplot(ax_main)

        ax_bottom = plt.Subplot(fig, inner[5])
        if data_dict_global[feature_name]["xtype"] == "categorical":
            xint = np.arange(len(data_dict_global[feature_name1]["density"]["names"]))
            ax_bottom.bar(xint, data_dict_global[feature_name1]["density"]["scores"])
            ax_bottom.set_xticks(data_dict_global[feature_name]["input1_ticks"])
            ax_bottom.set_xticklabels(data_dict_global[feature_name]["input1_labels"])
        else:
            xint = ((np.array(data_dict_global[feature_name1]["density"]["names"][1:])
                  + np.array(data_dict_global[feature_name1]["density"]["names"][:-1])) / 2).reshape([-1])
            ax_bottom.bar(xint, data_dict_global[feature_name1]["density"]["scores"], width=xint[1] - xint[0])
        ax_bottom.set_yticklabels([])
        ax_bottom.set_xlim([axis_extent[0], axis_extent[1]])
        ax_bottom.get_shared_x_axes().join(ax_bottom, ax_main)
        ax_bottom.autoscale()
        fig.add_subplot(ax_bottom)
        if len(str(ax_bottom.get_xticks())) > 60:
            ax_bottom.xaxis.set_tick_params(rotation=20)

        ax_left = plt.Subplot(fig, inner[0])
        if data_dict_global[feature_name]["ytype"] == "categorical":
            xint = np.arange(len(data_dict_global[feature_name2]["density"]["names"]))
            ax_left.barh(xint, data_dict_global[feature_name2]["density"]["scores"])
            ax_left.set_yticks(data_dict_global[feature_name]["input2_ticks"])
            ax_left.set_yticklabels(data_dict_global[feature_name]["input2_labels"])
        else:
            xint = ((np.array(data_dict_global[feature_name2]["density"]["names"][1:])
                  + np.array(data_dict_global[feature_name2]["density"]["names"][:-1])) / 2).reshape([-1])
            ax_left.barh(xint, data_dict_global[feature_name2]["density"]["scores"], height=xint[1] - xint[0])
        ax_left.set_xticklabels([])
        ax_left.set_ylim([axis_extent[2], axis_extent[3]])
        ax_left.get_shared_y_axes().join(ax_left, ax_main)
        ax_left.autoscale()
        fig.add_subplot(ax_left)

        ax_colorbar = plt.Subplot(fig, inner[2])
        response_precision = max(int(- np.log10(np.max(data_dict_global[feature_name]["outputs"])
                                   - np.min(data_dict_global[feature_name]["outputs"]))) + 2, 0)
        fig.colorbar(interact_plot, cax=ax_colorbar, orientation="vertical",
                     format="%0." + str(response_precision) + "f", use_gridspec=True)
        fig.add_subplot(ax_colorbar)
        idx = idx + 1

    if max_ids > 0:
        save_path = folder + name
        if save_eps:
            if not os.path.exists(folder):
                os.makedirs(folder)
            fig.savefig("%s.eps" % save_path, bbox_inches="tight", dpi=100)
        if save_png:
            if not os.path.exists(folder):
                os.makedirs(folder)
            fig.savefig("%s.png" % save_path, bbox_inches="tight", dpi=100)

def global_visualize_wo_density(data_dict_global, main_effect_num=None, interaction_num=None, cols_per_row=4,
                        save_png=False, save_eps=False, folder="./results/", name="demo"):

    maineffect_count = 0
    componment_scales = []
    for key, item in data_dict_global.items():
        componment_scales.append(item["importance"])
        if item["type"] != "pairwise":
            maineffect_count += 1

    componment_scales = np.array(componment_scales)
    sorted_index = np.argsort(componment_scales)
    active_index = sorted_index[componment_scales[sorted_index].cumsum() > 0][::-1]
    active_univariate_index = active_index[active_index < maineffect_count][:main_effect_num]
    active_interaction_index = active_index[active_index >= maineffect_count][:interaction_num]
    max_ids = len(active_univariate_index) + len(active_interaction_index)

    idx = 0
    fig = plt.figure(figsize=(5.2 * cols_per_row, 4 * int(np.ceil(max_ids / cols_per_row))))
    outer = gridspec.GridSpec(int(np.ceil(max_ids / cols_per_row)), cols_per_row, wspace=0.25, hspace=0.35)
    for indice in active_univariate_index:

        feature_name = list(data_dict_global.keys())[indice]
        if data_dict_global[feature_name]["type"] == "continuous":

            ax1 = plt.Subplot(fig, outer[idx])
            ax1.plot(data_dict_global[feature_name]["inputs"], data_dict_global[feature_name]["outputs"])
            ax1.set_title(feature_name, fontsize=12)
            fig.add_subplot(ax1)
            if len(str(ax1.get_xticks())) > 80:
                ax1.xaxis.set_tick_params(rotation=20)

        elif data_dict_global[feature_name]["type"] == "categorical":

            ax1 = plt.Subplot(fig, outer[idx])
            ax1.bar(np.arange(len(data_dict_global[feature_name]["inputs"])),
                        data_dict_global[feature_name]["outputs"])
            ax1.set_title(feature_name, fontsize=12)
            ax1.set_xticks(data_dict_global[feature_name]["input_ticks"])
            ax1.set_xticklabels(data_dict_global[feature_name]["input_labels"])
            fig.add_subplot(ax1)

        idx = idx + 1
        if len(str(ax1.get_xticks())) > 60:
            ax1.xaxis.set_tick_params(rotation=20)
        ax1.set_title(feature_name + " (" + str(np.round(100 * data_dict_global[feature_name]["importance"], 1)) + "%)", fontsize=12)

    for indice in active_interaction_index:

        feature_name = list(data_dict_global.keys())[indice]
        axis_extent = data_dict_global[feature_name]["axis_extent"]

        ax_main = plt.Subplot(fig, outer[idx])
        interact_plot = ax_main.imshow(data_dict_global[feature_name]["outputs"], interpolation="nearest",
                             aspect="auto", extent=axis_extent)

        if data_dict_global[feature_name]["xtype"] == "categorical":
            ax_main.set_xticks(data_dict_global[feature_name]["input1_ticks"])
            ax_main.set_xticklabels(data_dict_global[feature_name]["input1_labels"])
        if data_dict_global[feature_name]["ytype"] == "categorical":
            ax_main.set_yticks(data_dict_global[feature_name]["input2_ticks"])
            ax_main.set_yticklabels(data_dict_global[feature_name]["input2_labels"])

        response_precision = max(int(- np.log10(np.max(data_dict_global[feature_name]["outputs"])
                                   - np.min(data_dict_global[feature_name]["outputs"]))) + 2, 0)
        fig.colorbar(interact_plot, ax=ax_main, orientation="vertical",
                     format="%0." + str(response_precision) + "f", use_gridspec=True)

        ax_main.set_title(feature_name + " (" + str(np.round(100 * data_dict_global[feature_name]["importance"], 1)) + "%)", fontsize=12)
        fig.add_subplot(ax_main)

        idx = idx + 1
        if len(str(ax_main.get_xticks())) > 60:
            ax_main.xaxis.set_tick_params(rotation=20)

    if max_ids > 0:
        save_path = folder + name
        if save_eps:
            if not os.path.exists(folder):
                os.makedirs(folder)
            fig.savefig("%s.eps" % save_path, bbox_inches="tight", dpi=100)
        if save_png:
            if not os.path.exists(folder):
                os.makedirs(folder)
            fig.savefig("%s.png" % save_path, bbox_inches="tight", dpi=100)

def local_visualize(data_dict_local, folder="./results/", name="demo", save_png=False, save_eps=False):

    idx = np.argsort(np.abs(data_dict_local["scores"][data_dict_local["active_indice"]]))[::-1]

    max_ids = len(data_dict_local["active_indice"])
    fig = plt.figure(figsize=(round((len(data_dict_local["active_indice"]) + 1) * 0.6), 4))
    plt.bar(np.arange(len(data_dict_local["active_indice"])), data_dict_local["scores"][data_dict_local["active_indice"]][idx])
    plt.xticks(np.arange(len(data_dict_local["active_indice"])),
            data_dict_local["effect_names"][data_dict_local["active_indice"]][idx], rotation=60)

    if "actual" in data_dict_local.keys():
        title = "Predicted: %0.4f | Actual: %0.4f" % (data_dict_local["predicted"], data_dict_local["actual"])
    else:
        title = "Predicted: %0.4f" % (data_dict_local["predicted"])
    plt.title(title, fontsize=12)

    if max_ids > 0:
        save_path = folder + name
        if save_eps:
            if not os.path.exists(folder):
                os.makedirs(folder)
            fig.savefig("%s.eps" % save_path, bbox_inches="tight", dpi=100)
        if save_png:
            if not os.path.exists(folder):
                os.makedirs(folder)
            fig.savefig("%s.png" % save_path, bbox_inches="tight", dpi=100)

class CategNet(tf.keras.layers.Layer):

    def __init__(self, category_num, cagetnet_id):
        super(CategNet, self).__init__()
        self.category_num = category_num
        self.cagetnet_id = cagetnet_id

        self.output_layer_bias = self.add_weight(name="output_layer_bias_" + str(self.cagetnet_id),
                                     shape=[1, 1],
                                     initializer=tf.zeros_initializer(),
                                     trainable=False)
        self.categ_bias = self.add_weight(name="cate_bias_" + str(self.cagetnet_id),
                                         shape=[self.category_num, 1],
                                         initializer=tf.zeros_initializer(),
                                         trainable=True)
        self.moving_mean = self.add_weight(name="mean" + str(self.cagetnet_id), shape=[1],
                                           initializer=tf.zeros_initializer(), trainable=False)
        self.moving_norm = self.add_weight(name="norm" + str(self.cagetnet_id), shape=[1],
                                           initializer=tf.ones_initializer(), trainable=False)

    def call(self, inputs, sample_weight=None, training=False):

        dummy = tf.one_hot(indices=tf.cast(inputs[:, 0], tf.int32), depth=self.category_num)
        self.output_original = tf.matmul(dummy, self.categ_bias) + self.output_layer_bias

        if training:
            if sample_weight is None:
                if inputs.shape[0] is not None:
                    sample_weight = tf.ones([inputs.shape[0], 1])
                    self.subnet_mean, self.subnet_norm = tf.nn.weighted_moments(self.output_original,
                                                            frequency_weights=sample_weight, axes=0)
            else:
                sample_weight = tf.reshape(sample_weight, shape=(-1, 1))
                self.subnet_mean, self.subnet_norm = tf.nn.weighted_moments(self.output_original,
                                                        frequency_weights=sample_weight, axes=0)
            self.moving_mean.assign(self.subnet_mean)
            self.moving_norm.assign(self.subnet_norm)
        else:
            self.subnet_mean = self.moving_mean
            self.subnet_norm = self.moving_norm

        output = self.output_original
        return output

class NumerNet(tf.keras.layers.Layer):

    def __init__(self, subnet_arch, activation_func, subnet_id):
        super(NumerNet, self).__init__()
        self.layers = []
        self.subnet_arch = subnet_arch
        self.activation_func = activation_func
        self.subnet_id = subnet_id

        for nodes in self.subnet_arch:
            self.layers.append(layers.Dense(nodes, activation=self.activation_func, kernel_initializer=tf.keras.initializers.Orthogonal()))
        self.output_layer = layers.Dense(1, activation=tf.identity, kernel_initializer=tf.keras.initializers.Orthogonal())

        self.min_value = self.add_weight(name="min" + str(self.subnet_id), shape=[1], initializer=tf.constant_initializer(np.inf), trainable=False)
        self.max_value = self.add_weight(name="max" + str(self.subnet_id), shape=[1], initializer=tf.constant_initializer(-np.inf), trainable=False)
        self.moving_mean = self.add_weight(name="mean" + str(self.subnet_id), shape=[1], initializer=tf.zeros_initializer(), trainable=False)
        self.moving_norm = self.add_weight(name="norm" + str(self.subnet_id), shape=[1], initializer=tf.ones_initializer(), trainable=False)

    def call(self, inputs, sample_weight=None, training=False):

        if training:
            self.min_value.assign(tf.minimum(self.min_value, tf.reduce_min(inputs)))
            self.max_value.assign(tf.maximum(self.max_value, tf.reduce_max(inputs)))

        x = tf.clip_by_value(inputs, self.min_value, self.max_value)
        for dense_layer in self.layers:
            x = dense_layer(x)
        self.output_original = self.output_layer(x)

        if training:
            if sample_weight is None:
                if inputs.shape[0] is not None:
                    sample_weight = tf.ones([inputs.shape[0], 1])
                    self.subnet_mean, self.subnet_norm = tf.nn.weighted_moments(self.output_original,
                                                            frequency_weights=sample_weight, axes=0)
            else:
                sample_weight = tf.reshape(sample_weight, shape=(-1, 1))
                self.subnet_mean, self.subnet_norm = tf.nn.weighted_moments(self.output_original,
                                                        frequency_weights=sample_weight, axes=0)
            self.moving_mean.assign(self.subnet_mean)
            self.moving_norm.assign(self.subnet_norm)
        else:
            self.subnet_mean = self.moving_mean
            self.subnet_norm = self.moving_norm

        output = self.output_original
        return output

class MonoConNumerNet(tf.keras.layers.Layer):

    def __init__(self, monotonicity, convexity, lattice_size, subnet_id):
        super(MonoConNumerNet, self).__init__()

        self.subnet_id = subnet_id
        self.monotonicity = monotonicity
        self.convexity = convexity
        self.lattice_size = lattice_size
        self.lattice_layer = tfl.layers.Lattice(lattice_sizes=[self.lattice_size], monotonicities=['increasing'])
        
        self.lattice_layer_input = tfl.layers.PWLCalibration(input_keypoints=np.linspace(0, 1, num=8, dtype=np.float32),
                                        output_min=0.0, output_max=self.lattice_size - 1.0)
        if monotonicity:
            self.lattice_layer_input.monotonicity = monotonicity
        if convexity:
            self.lattice_layer_input.convexity = convexity
        self.lattice_layer_bias = self.add_weight(name="lattice_layer_bias_" + str(self.subnet_id), shape=[1],
                                    initializer=tf.zeros_initializer(), trainable=False)

        self.min_value = self.add_weight(name="min" + str(self.subnet_id), shape=[1], initializer=tf.constant_initializer(np.inf), trainable=False)
        self.max_value = self.add_weight(name="max" + str(self.subnet_id), shape=[1], initializer=tf.constant_initializer(-np.inf), trainable=False)
        self.moving_mean = self.add_weight(name="mean" + str(self.subnet_id), shape=[1], initializer=tf.zeros_initializer(), trainable=False)
        self.moving_norm = self.add_weight(name="norm" + str(self.subnet_id), shape=[1], initializer=tf.ones_initializer(), trainable=False)

    def call(self, inputs, sample_weight=None, training=False):

        if training:
            self.min_value.assign(tf.minimum(self.min_value, tf.reduce_min(inputs)))
            self.max_value.assign(tf.maximum(self.max_value, tf.reduce_max(inputs)))

        x = tf.clip_by_value(inputs, self.min_value, self.max_value)
        lattice_input = self.lattice_layer_input(x)
        self.output_original = self.lattice_layer(lattice_input) + self.lattice_layer_bias

        if training:
            if sample_weight is None:
                if inputs.shape[0] is not None:
                    sample_weight = tf.ones([inputs.shape[0], 1])
                    self.subnet_mean, self.subnet_norm = tf.nn.weighted_moments(self.output_original,
                                                            frequency_weights=sample_weight, axes=0)
            else:
                sample_weight = tf.reshape(sample_weight, shape=(-1, 1))
                self.subnet_mean, self.subnet_norm = tf.nn.weighted_moments(self.output_original,
                                                        frequency_weights=sample_weight, axes=0)
            self.moving_mean.assign(self.subnet_mean)
            self.moving_norm.assign(self.subnet_norm)
        else:
            self.subnet_mean = self.moving_mean
            self.subnet_norm = self.moving_norm

        output = self.output_original
        return output

class MainEffectBlock(tf.keras.layers.Layer):

    def __init__(self, feature_list, nfeature_index_list, cfeature_index_list, dummy_values,
                 subnet_arch, activation_func, mono_increasing_list, mono_decreasing_list, convex_list, concave_list, lattice_size):
        super(MainEffectBlock, self).__init__()

        self.subnet_arch = subnet_arch
        self.lattice_size = lattice_size
        self.activation_func = activation_func
        
        self.dummy_values = dummy_values
        self.feature_list = feature_list
        self.subnet_num = len(feature_list)
        self.nfeature_index_list = nfeature_index_list
        self.cfeature_index_list = cfeature_index_list
        self.mono_increasing_list = mono_increasing_list
        self.mono_decreasing_list = mono_decreasing_list
        self.convex_list = convex_list
        self.concave_list = concave_list

        self.subnets = []
        for i in range(self.subnet_num):
            if i in self.nfeature_index_list:
                convexity = None
                monotonicity = None
                if i in self.mono_increasing_list:
                    monotonicity = "increasing"
                elif i in self.mono_decreasing_list:
                    monotonicity = "decreasing"
                if i in self.convex_list:
                    convexity = "convex"
                elif i in self.concave_list:
                    convexity = "concave"
                if monotonicity or convexity:
                    self.subnets.append(MonoConNumerNet(monotonicity, convexity, self.lattice_size, subnet_id=i))
                else:
                    self.subnets.append(NumerNet(self.subnet_arch, self.activation_func, subnet_id=i))
            elif i in self.cfeature_index_list:
                feature_name = self.feature_list[i]
                self.subnets.append(CategNet(category_num=len(self.dummy_values[feature_name]), cagetnet_id=i))

    def call(self, inputs, sample_weight=None, training=False):

        self.subnet_outputs = []
        for i in range(self.subnet_num):
            subnet = self.subnets[i]
            subnet_output = subnet(tf.gather(inputs, [i], axis=1), sample_weight=sample_weight, training=training)
            self.subnet_outputs.append(subnet_output)
        output = tf.reshape(tf.squeeze(tf.stack(self.subnet_outputs, 1)), [-1, self.subnet_num])

        return output

class Interactnetwork(tf.keras.layers.Layer):

    def __init__(self, feature_list, cfeature_index_list, dummy_values, interact_arch,
                 activation_func, interact_id):
        super(Interactnetwork, self).__init__()

        self.feature_list = feature_list
        self.dummy_values = dummy_values
        self.cfeature_index_list = cfeature_index_list

        self.layers = []
        self.interact_arch = interact_arch
        self.activation_func = activation_func
        self.interact_id = interact_id
        self.interaction = None

    def set_interaction(self, interaction):

        self.interaction = interaction
        for nodes in self.interact_arch:
            self.layers.append(layers.Dense(nodes, activation=self.activation_func, kernel_initializer=tf.keras.initializers.Orthogonal()))
        self.output_layer = layers.Dense(1, activation=tf.identity, kernel_initializer=tf.keras.initializers.Orthogonal())
        
        self.min_value1 = self.add_weight(name="min1" + str(self.interact_id), shape=[1],
                                          initializer=tf.constant_initializer(np.inf), trainable=False)
        self.max_value1 = self.add_weight(name="max1" + str(self.interact_id), shape=[1],
                                          initializer=tf.constant_initializer(-np.inf), trainable=False)
        self.min_value2 = self.add_weight(name="min2" + str(self.interact_id), shape=[1],
                                          initializer=tf.constant_initializer(np.inf), trainable=False)
        self.max_value2 = self.add_weight(name="max2" + str(self.interact_id), shape=[1],
                                          initializer=tf.constant_initializer(-np.inf), trainable=False)

        self.moving_mean = self.add_weight(name="mean_" + str(self.interact_id),
                                shape=[1], initializer=tf.zeros_initializer(), trainable=False)
        self.moving_norm = self.add_weight(name="norm_" + str(self.interact_id),
                                shape=[1], initializer=tf.ones_initializer(), trainable=False)

    def preprocessing(self, inputs):

        interact_input_list = []
        if self.interaction[0] in self.cfeature_index_list:
            interact_input1 = tf.one_hot(indices=tf.cast(inputs[:, 0], tf.int32),
                               depth=len(self.dummy_values[self.feature_list[self.interaction[0]]]))
            interact_input_list.extend(tf.unstack(interact_input1, axis=-1))
        else:
            interact_input_list.append(tf.clip_by_value(inputs[:, 0], self.min_value1, self.max_value1))
        if self.interaction[1] in self.cfeature_index_list:
            interact_input2 = tf.one_hot(indices=tf.cast(inputs[:, 1], tf.int32),
                               depth=len(self.dummy_values[self.feature_list[self.interaction[1]]]))
            interact_input_list.extend(tf.unstack(interact_input2, axis=-1))
        else:
            interact_input_list.append(tf.clip_by_value(inputs[:, 1], self.min_value2, self.max_value2))
        return interact_input_list

    def call(self, inputs, sample_weight=None, training=False):

        if training:
            self.min_value1.assign(tf.minimum(self.min_value1, tf.reduce_min(inputs[:, 0])))
            self.max_value1.assign(tf.maximum(self.max_value1, tf.reduce_max(inputs[:, 0])))
            self.min_value2.assign(tf.minimum(self.min_value2, tf.reduce_min(inputs[:, 1])))
            self.max_value2.assign(tf.maximum(self.max_value2, tf.reduce_max(inputs[:, 1])))

        x = tf.stack(self.preprocessing(inputs), 1)
        for dense_layer in self.layers:
            x = dense_layer(x)
        self.output_original = self.output_layer(x)

        if training:
            if sample_weight is None:
                if inputs.shape[0] is not None:
                    sample_weight = tf.ones([inputs.shape[0], 1])
                    self.subnet_mean, self.subnet_norm = tf.nn.weighted_moments(self.output_original,
                                                            frequency_weights=sample_weight, axes=0)
            else:
                sample_weight = tf.reshape(sample_weight, shape=(-1, 1))
                self.subnet_mean, self.subnet_norm = tf.nn.weighted_moments(self.output_original,
                                                        frequency_weights=sample_weight, axes=0)
            self.moving_mean.assign(self.subnet_mean)
            self.moving_norm.assign(self.subnet_norm)
        else:
            self.subnet_mean = self.moving_mean
            self.subnet_norm = self.moving_norm

        output = self.output_original
        return output

class MonoConInteractnetwork(tf.keras.layers.Layer):

    def __init__(self, feature_list, cfeature_index_list, dummy_values, lattice_size, monotonicity, convexity, interact_id):
        super(MonoConInteractnetwork, self).__init__()

        self.feature_list = feature_list
        self.dummy_values = dummy_values
        self.cfeature_index_list = cfeature_index_list
        
        self.monotonicity = monotonicity
        self.convexity = convexity
        self.lattice_size = lattice_size
        self.interact_id = interact_id
        self.interaction = None

    def set_interaction(self, interaction):

        self.interaction = interaction
        if self.interaction[0] in self.cfeature_index_list:
            depth = len(self.dummy_values[self.feature_list[self.interaction[0]]])
            self.lattice_layer_input1 = tfl.layers.CategoricalCalibration(num_buckets=depth, output_min=0.0, output_max=1.0)
        else:
            self.lattice_layer_input1 = tfl.layers.PWLCalibration(input_keypoints=np.linspace(0, 1, num=8, dtype=np.float32),
                                            output_min=0.0, output_max=self.lattice_size[0] - 1.0)
            if self.monotonicity[0]:
                self.lattice_layer_input1.monotonicity = self.monotonicity[0]
            if self.convexity[0]:
                self.lattice_layer_input1.convexity = self.convexity[0]

        if self.interaction[1] in self.cfeature_index_list:
            depth = len(self.dummy_values[self.feature_list[self.interaction[1]]])
            self.lattice_layer_input2 = tfl.layers.CategoricalCalibration(num_buckets=depth, output_min=0.0, output_max=1.0)
        else:
            self.lattice_layer_input2 = tfl.layers.PWLCalibration(input_keypoints=np.linspace(0, 1, num=8, dtype=np.float32),
                                            output_min=0.0, output_max=self.lattice_size[1] - 1.0)
            if self.monotonicity[1]:
                self.lattice_layer_input2.monotonicity = self.monotonicity[1]
            if self.convexity[1]:
                self.lattice_layer_input2.convexity = self.convexity[1]

        self.lattice_layer2d = tfl.layers.Lattice(lattice_sizes=self.lattice_size, monotonicities=['increasing', 'increasing'])
        self.lattice_layer_bias = self.add_weight(name="lattice_layer2d_bias_" + str(self.interact_id), shape=[1],
                                    initializer=tf.zeros_initializer(), trainable=False)

        self.min_value1 = self.add_weight(name="min1" + str(self.interact_id), shape=[1],
                                          initializer=tf.constant_initializer(np.inf), trainable=False)
        self.max_value1 = self.add_weight(name="max1" + str(self.interact_id), shape=[1],
                                          initializer=tf.constant_initializer(-np.inf), trainable=False)
        self.min_value2 = self.add_weight(name="min2" + str(self.interact_id), shape=[1],
                                          initializer=tf.constant_initializer(np.inf), trainable=False)
        self.max_value2 = self.add_weight(name="max2" + str(self.interact_id), shape=[1],
                                          initializer=tf.constant_initializer(-np.inf), trainable=False)

        self.moving_mean = self.add_weight(name="mean_" + str(self.interact_id),
                                shape=[1], initializer=tf.zeros_initializer(), trainable=False)
        self.moving_norm = self.add_weight(name="norm_" + str(self.interact_id),
                                shape=[1], initializer=tf.ones_initializer(), trainable=False)

    def preprocessing(self, inputs):

        interact_input_list = []
        if self.interaction[0] in self.cfeature_index_list:
            interact_input_list.append(tf.reshape(inputs[:, 0], (-1, 1)))
        else:
            interact_input_list.append(tf.reshape(tf.clip_by_value(inputs[:, 0], self.min_value1, self.max_value1), (-1, 1)))
        if self.interaction[1] in self.cfeature_index_list:
            interact_input_list.append(tf.reshape(inputs[:, 1], (-1, 1)))
        else:
            interact_input_list.append(tf.reshape(tf.clip_by_value(inputs[:, 1], self.min_value2, self.max_value2), (-1, 1)))
        return interact_input_list

    def call(self, inputs, sample_weight=None, training=False):

        if training:
            self.min_value1.assign(tf.minimum(self.min_value1, tf.reduce_min(inputs[:, 0])))
            self.max_value1.assign(tf.maximum(self.max_value1, tf.reduce_max(inputs[:, 0])))
            self.min_value2.assign(tf.minimum(self.min_value2, tf.reduce_min(inputs[:, 1])))
            self.max_value2.assign(tf.maximum(self.max_value2, tf.reduce_max(inputs[:, 1])))

        x = self.preprocessing(inputs)
        lattice_input2d = tf.keras.layers.Concatenate(axis=1)([self.lattice_layer_input1(x[0]), self.lattice_layer_input2(x[1])])
        self.output_original = self.lattice_layer2d(lattice_input2d) + self.lattice_layer_bias

        if training:
            if sample_weight is None:
                if inputs.shape[0] is not None:
                    sample_weight = tf.ones([inputs.shape[0], 1])
                    self.subnet_mean, self.subnet_norm = tf.nn.weighted_moments(self.output_original,
                                                            frequency_weights=sample_weight, axes=0)
            else:
                sample_weight = tf.reshape(sample_weight, shape=(-1, 1))
                self.subnet_mean, self.subnet_norm = tf.nn.weighted_moments(self.output_original,
                                                        frequency_weights=sample_weight, axes=0)
            self.moving_mean.assign(self.subnet_mean)
            self.moving_norm.assign(self.subnet_norm)
        else:
            self.subnet_mean = self.moving_mean
            self.subnet_norm = self.moving_norm

        output = self.output_original
        return output


class InteractionBlock(tf.keras.layers.Layer):

    def __init__(self, interact_num, feature_list, cfeature_index_list, dummy_values,
                 interact_arch, activation_func, mono_increasing_list, mono_decreasing_list, convex_list, concave_list, lattice_size):

        super(InteractionBlock, self).__init__()

        self.feature_list = feature_list
        self.dummy_values = dummy_values
        self.cfeature_index_list = cfeature_index_list

        self.interact_num_added = 0
        self.interact_num = interact_num
        self.interact_arch = interact_arch
        self.activation_func = activation_func
        self.lattice_size = lattice_size
        self.mono_increasing_list = mono_increasing_list
        self.mono_decreasing_list = mono_decreasing_list
        self.mono_list = mono_increasing_list + mono_decreasing_list
        self.convex_list = convex_list
        self.concave_list = concave_list
        self.con_list = convex_list + concave_list

    def set_interaction_list(self, interaction_list):

        self.interacts = []
        self.interaction_list = interaction_list
        self.interact_num_added = len(interaction_list)
        for i in range(self.interact_num_added):
            if (interaction_list[i][0] in self.mono_list + self.con_list) or (interaction_list[i][1] in self.mono_list + self.con_list):
                lattice_size = [2, 2]
                convexity = [None, None]
                monotonicity = [None, None]
                if interaction_list[i][0] in self.mono_increasing_list:
                    monotonicity[0] = "increasing"
                    lattice_size[0] = self.lattice_size
                elif interaction_list[i][0] in self.mono_decreasing_list:
                    monotonicity[0] = "decreasing"
                    lattice_size[0] = self.lattice_size
                if interaction_list[i][0] in self.convex_list:
                    convexity[0] = "convex"
                    lattice_size[0] = self.lattice_size
                elif interaction_list[i][0] in self.concave_list:
                    convexity[0] = "concave"
                    lattice_size[0] = self.lattice_size

                if interaction_list[i][1] in self.mono_increasing_list:
                    monotonicity[1] = "increasing"
                    lattice_size[1] = self.lattice_size
                elif interaction_list[i][1] in self.mono_decreasing_list:
                    monotonicity[1] = "decreasing"
                    lattice_size[1] = self.lattice_size
                if interaction_list[i][1] in self.convex_list:
                    convexity[1] = "convex"
                    lattice_size[1] = self.lattice_size
                elif interaction_list[i][1] in self.concave_list:
                    convexity[1] = "concave"
                    lattice_size[1] = self.lattice_size

                interact = MonoConInteractnetwork(self.feature_list,
                                      self.cfeature_index_list,
                                      self.dummy_values,
                                      monotonicity=monotonicity,
                                      convexity=convexity,
                                      lattice_size=lattice_size,
                                      interact_id=i)
            else:
                interact = Interactnetwork(self.feature_list,
                                          self.cfeature_index_list,
                                          self.dummy_values,
                                          self.interact_arch,
                                          self.activation_func,
                                          interact_id=i)
            interact.set_interaction(interaction_list[i])
            self.interacts.append(interact)
            
    def call(self, inputs, sample_weight=None, training=False):

        self.interact_outputs = []
        for i in range(self.interact_num):
            if i >= self.interact_num_added:
                self.interact_outputs.append(tf.zeros([inputs.shape[0], 1]))
            else:
                interact = self.interacts[i]
                interact_input = tf.gather(inputs, self.interaction_list[i], axis=1)
                interact_output = interact(interact_input, sample_weight=sample_weight, training=training)
                self.interact_outputs.append(interact_output)

        if len(self.interact_outputs) > 0:
            output = tf.reshape(tf.squeeze(tf.stack(self.interact_outputs, 1)), [-1, self.interact_num])
        else:
            output = 0
        return output


class NonNegative(tf.keras.constraints.Constraint):

    def __init__(self, mono_increasing_list, mono_decreasing_list, convex_list, concave_list):

        self.mono_increasing_list = mono_increasing_list
        self.mono_decreasing_list = mono_decreasing_list
        self.convex_list = convex_list
        self.concave_list = concave_list

    def __call__(self, w):

        if len(self.mono_increasing_list) > 0:
            mono_increasing_weights = tf.abs(tf.gather(w, self.mono_increasing_list))
            w = tf.tensor_scatter_nd_update(w, [[item] for item in self.mono_increasing_list], mono_increasing_weights)
        if len(self.mono_decreasing_list) > 0:
            mono_decreasing_weights = tf.abs(tf.gather(w, self.mono_decreasing_list))
            w = tf.tensor_scatter_nd_update(w, [[item] for item in self.mono_decreasing_list], mono_decreasing_weights)
        if len(self.convex_list) > 0:
            convex_weights = tf.abs(tf.gather(w, self.convex_list))
            w = tf.tensor_scatter_nd_update(w, [[item] for item in self.convex_list], convex_weights)
        if len(self.concave_list) > 0:
            concave_weights = tf.abs(tf.gather(w, self.concave_list))
            w = tf.tensor_scatter_nd_update(w, [[item] for item in self.concave_list], concave_weights)
        return w

class OutputLayer(tf.keras.layers.Layer):

    def __init__(self, input_num, interact_num, mono_increasing_list, mono_decreasing_list, convex_list, concave_list):

        super(OutputLayer, self).__init__()

        self.interaction = []
        self.input_num = input_num
        self.interact_num_added = 0
        self.interact_num = interact_num
        self.mono_increasing_list = mono_increasing_list
        self.mono_decreasing_list = mono_decreasing_list
        self.convex_list = convex_list
        self.concave_list = concave_list

        self.main_effect_weights = self.add_weight(name="subnet_weights",
                                              shape=[self.input_num, 1],
                                              initializer=tf.keras.initializers.Orthogonal(),
                                              constraint=NonNegative(self.mono_increasing_list, self.mono_decreasing_list,
                                                            self.convex_list, self.concave_list),
                                              trainable=True)
        self.main_effect_switcher = self.add_weight(name="subnet_switcher",
                                              shape=[self.input_num, 1],
                                              initializer=tf.ones_initializer(),
                                              trainable=False)

        self.interaction_weights = self.add_weight(name="interaction_weights",
                                  shape=[self.interact_num, 1],
                                  initializer=tf.keras.initializers.Orthogonal(),
                                  constraint=NonNegative([], [], [], []),
                                  trainable=True)
        self.interaction_switcher = self.add_weight(name="interaction_switcher",
                                              shape=[self.interact_num, 1],
                                              initializer=tf.ones_initializer(),
                                              trainable=False)
        self.output_bias = self.add_weight(name="output_bias",
                                           shape=[1],
                                           initializer=tf.zeros_initializer(),
                                           trainable=True)

    def set_interaction_list(self, interaction_list):

        self.convex_interact_list = []
        self.concave_interact_list = []
        self.mono_increasing_interact_list = []
        self.mono_decreasing_interact_list = []
        self.interaction_list = interaction_list
        self.interact_num_added = len(interaction_list)
        for i, interaction in enumerate(self.interaction_list):
            if (interaction[0] in self.mono_increasing_list) or (interaction[1] in self.mono_increasing_list):
                self.mono_increasing_interact_list.append(i)
            if (interaction[0] in self.mono_decreasing_list) or (interaction[1] in self.mono_decreasing_list):
                self.mono_decreasing_interact_list.append(i)
                
            if (interaction[0] in self.convex_list) or (interaction[1] in self.convex_list):
                self.convex_interact_list.append(i)
            if (interaction[0] in self.concave_list) or (interaction[1] in self.concave_list):
                self.concave_interact_list.append(i)

        self.interaction_weights.constraint.mono_increasing_list = self.mono_increasing_interact_list
        self.interaction_weights.constraint.mono_decreasing_list = self.mono_decreasing_interact_list
        self.interaction_weights.constraint.convex_list = self.convex_interact_list
        self.interaction_weights.constraint.concave_list = self.concave_interact_list

    def call(self, inputs):

        self.input_main_effect = inputs[:, :self.input_num]
        if self.interact_num_added > 0:
            self.input_interaction = inputs[:, self.input_num:]
            output = (tf.matmul(self.input_main_effect, self.main_effect_switcher * self.main_effect_weights)
                   + tf.matmul(self.input_interaction, self.interaction_switcher * self.interaction_weights)
                   + self.output_bias)
        else:
            output = (tf.matmul(self.input_main_effect, self.main_effect_switcher * self.main_effect_weights)
                   + self.output_bias)
        return output

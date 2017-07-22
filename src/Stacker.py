from Model import Model
import pandas as pd
import numpy as np
import logging

class Stacker():
    def __init__(self, mdl_list, base_feature, stack_mdl, num_stack = 4):
        """
        Initialize stacker
        :param mdl_list: specify a list of models used for stacking. Different models can contain different features
        :param base_feature: combine different predictions and base features to generate next level prediction
        """
        self.mdl_list = mdl_list
        self._BaseFeature = base_feature

        # elements required for stacking
        self.stack_mdl = stack_mdl
        self.num_stack = num_stack
        self.fTrainLayer = []
        self.fTestLayer = []

    @property
    def train_out(self):
        return self._BaseFeature.train_out

    @property
    def BaseFeature(self):
        return self._BaseFeature

    @BaseFeature.setter
    def BaseFeature(self, value):
        raise Exception('Can not modify base feature in the Stacker')

    def _cvtrain(self, train_idx):
        self.fit(train_idx)
        self.stack_layer(train_idx)

    def _cvpredict(self, test_idx):
        ptest = np.zeros((len(test_idx), len(self.mdl_list)))
        for i in range(len(self.mdl_list)):
            ptest[:,i:(i+1)] = self.mdl_list[i]._cvpredict(test_idx)

        test_in = np.hstack((self.BaseFeature.fTrain[test_idx,:], ptest))
        y_pred = self.stack_mdl._predict(test_in)
        return y_pred

    def train_all(self):
        all_idx = np.arange(len(self.train_out))
        self.fit(all_idx)
        self.stack_layer(all_idx)

    def stack_layer(self, train_idx):
        logging.info('stacking layer')
        train_in = np.hstack((self.BaseFeature.fTrain[train_idx,:], self.ptrain))
        self.stack_mdl._train(train_in, self.train_out[train_idx])

    def fit(self, train_idx):
        self.ptrain = np.zeros((len(train_idx), len(self.mdl_list)))
        for i in range(len(self.mdl_list)):
            self.ptrain[:,i:(i+1)] = self.mdl_list[i].gen_stacking(train_idx, self.num_stack)
            self.mdl_list[i]._cvtrain(train_idx)

    def predict_all(self):
        data_in = self.BaseFeature.fTest
        ptest = np.zeros((data_in.shape[0], len(self.mdl_list)))
        for i in range(len(self.mdl_list)):
            ptest[:,i:(i+1)] = self.mdl_list[i].predict_all()
        test_in = np.hstack((self.BaseFeature.fTest, ptest))
        y_pred = self.stack_mdl._predict(test_in)
        return y_pred

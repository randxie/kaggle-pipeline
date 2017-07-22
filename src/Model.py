from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import xgboost as xgb
import logging

class Model(object):
    """
    A class for defining different models. The current implementation is only for binary classification problems.
    """
    def __init__(self, features, mdl_type, param):
        self._Features = features
        self.param = param
        self.mdl_type = mdl_type

        # select which model to use
        self.select_mdl(mdl_type, param)
        logging.info("Initialize model : %s" % (mdl_type))
        logging.info("Model paramters: %s"%(param))

    @property
    def train_out(self):
        return self.Features.train_out

    @property
    def Features(self):
        return self._Features

    @Features.setter
    def Features(self, value):
        raise Exception('Can not modify features in the model')

    def select_mdl(self, mdl_type, param):
        """
        # define classifier and parameters
        :param mdl_type: specify which model to initialize
        :param param: a dict storing model parameters
        """
        if (mdl_type == 'xgb'):
            self.mdl = xgb.XGBClassifier(**param)
        elif (mdl_type == 'lr'):
            self.mdl = LogisticRegression(**param)
        elif (mdl_type == 'rf'):
            self.mdl = RandomForestClassifier(**param)

    def _cvtrain(self, train_idx):
        """
        For the purpose of cross validation and stacking, specify the index for training local model
        :param train_idx: index for accessing data in Features.fTrain
        """
        self.mdl.fit(self.Features.fTrain[train_idx,:], self.Features.train_out[train_idx])

    def _cvpredict(self, test_idx):
        """
        For the purpose of cross validation and stacking, specify the index for testing local model
        :param test_idx: index for accessing data in Features.fTrain
        """
        test_in = self.Features.fTrain[test_idx,:]
        return self._predict(test_in)[:,None]

    def gen_stacking(self, all_idx, num_stack, seed=4242):
        """
        For stacking purpose
        :param all_idx: data index for generating stacking prediction
        :param num_stack: number of stacking fold
        :param seed: seed to k fold split
        :return: stacking prediction vector
        """
        logging.info('generating stacking fold for model %s with seed %d'%(self.mdl_type, seed))
        kf = KFold(all_idx.shape[0], n_folds=num_stack, shuffle=True, random_state=seed)
        ptrain = np.zeros((all_idx.shape[0], 1))

        for train_idx, test_idx in kf:
            logging.info('generating individual fold')
            # generate stacking for given index
            tmp_train_idx = all_idx[train_idx]
            tmp_test_idx = all_idx[test_idx]

            # fit model
            self._cvtrain(tmp_train_idx)
            test_pred = self._cvpredict(tmp_test_idx)
            ptrain[test_idx,:] = test_pred

        return ptrain

    def train_all(self):
        """
        Use all the training data to train a global model
        :return:
        """
        self.mdl.fit(self.Features.fTrain, self.Features.train_out)

    def predict_all(self):
        """
        Generate prediction for testing data stored in Features.fTest
        :return:
        """
        return self._predict(self.Features.fTest)[:,None]

    def _train(self, feature, label):
        """
        Train the model by specifying features and labels explicitly
        :param feature: features used for training
        :param label: corresponding labels
        :return:
        """
        self.mdl.fit(feature, label)

    # prediction with probability outcome
    def _predict(self, feature):
        """
        Generate prediction by specifying feature explicitly
        :param feature: feature used to generate prediction
        :return: prediction
        """
        if "predict_proba" in dir(self.mdl):
            pred = self.mdl.predict_proba(feature)[:,1]
        elif "predict" in dir(self.mdl):
            logging.warning("Model %s does not provide predict_proba, use predict instead"%self.mdl_type)
            pred = self.mdl.predict(feature)
        return pred

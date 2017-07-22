__author__ = 'randxie'
import logging, yaml
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class FeatureGenerator(object):
    '''
    A class to extract feature from data. Try to spend more time on feature engineering. It gives a lot of improvement.
    This feature generateor uses Kaggle-Titanic data for illustration. Most of time are spent in feature extraction.
    '''
    def __init__(self, datawarehouse, config):
        self._DW = datawarehouse
        self.config = config
        self.feature_names = []

    def compute_features(self):
        logging.info('Start to compute features')

        # get basic features
        self.fTrain = []
        self.fTest = []

        # append features as new columns
        fconfig = self.config['features']
        for k,v in fconfig.iteritems():
            if(v!=-1):
                self.feature_names.append(k)
                opt = v
                tmpTrain, tmpTest = eval('self.extract_%s(%s)'%(k,'opt'))
                logging.info('processing %s: added %d columns'%(k, tmpTrain.shape[1]))

                if(len(self.fTrain)):
                    self.fTrain = np.hstack((self.fTrain, tmpTrain))
                    self.fTest = np.hstack((self.fTest, tmpTest))
                else:
                    self.fTrain = tmpTrain
                    self.fTest = tmpTest

        logging.info("finish feature extraction")

    @property
    def train_out(self):
        return self.DW.train_out

    @property
    def DW(self):
        return self._DW

    @DW.setter
    def DW(self, value):
        raise Exception('Can not modify DataWarehouse through FeatureGenerator')

    def extract_sex(self, opt):
        tmpTrain = self.DW.train_in['Sex'].apply(lambda x: 1 if x=='male' else 0).as_matrix()[:,None]
        tmpTest = self.DW.test_in['Sex'].apply(lambda x: 1 if x=='male' else 0).as_matrix()[:,None]
        return tmpTrain, tmpTest

    def extract_age(self, opt):
        # impute date
        data_all = pd.concat([self.DW.train_in['Age'], self.DW.test_in['Age']])
        data_mean = data_all.mean()
        data_all = data_all.fillna(data_mean)

        n_train = self.DW.train_in.shape[0]
        tmpTrain = data_all[:n_train].as_matrix()[:,None]
        tmpTest = data_all[n_train:].as_matrix()[:,None]
        return tmpTrain, tmpTest

    def extract_sibsp(self, opt):
        return self.DW.train_in['SibSp'].as_matrix()[:,None], self.DW.test_in['SibSp'].as_matrix()[:,None]

    def extract_embarked(self, opt):
        data_all = pd.concat([self.DW.train_in['Embarked'], self.DW.test_in['Embarked']]).to_frame()
        num_count = data_all['Embarked'].value_counts().to_dict()
        data_all['Embarked'] = data_all['Embarked'].apply(lambda x: num_count[x] if x in num_count.keys() else 0)

        n_train = self.DW.train_in.shape[0]
        tmpTrain = data_all[:n_train].as_matrix()
        tmpTest = data_all[n_train:].as_matrix()
        return tmpTrain, tmpTest

    def normalize_features(self):
        # normalize feature matrices
        self.scaler = StandardScaler()
        self.scaler.fit(self.fTrain)
        self.fTrain = self.scaler.transform(self.fTrain)
        self.fTest = self.scaler.transform(self.fTest)

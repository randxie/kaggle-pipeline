from sklearn.cross_validation import KFold
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import log_loss
import numpy as np
import logging

class Validator():
    """
    Validator for single model
    """
    def __init__(self, Model):
        self.Mdl = Model

    def validate_kfold(self, num_fold, is_shuffle = True, metric = 'logloss'):
        """
        Split into k fold and find score for each fold. Try to develop an accurate validation scheme (that matches with private leader board)
        :param num_fold: number of fold for cross validation
        :param is_shuffle: whether to shuffle data
        :param metric: evaluation metric
        :return: none
        """
        kf = KFold(self.Mdl.train_out.shape[0], n_folds=num_fold, shuffle=is_shuffle)
        self.mdl_score = []

        for train_idx, test_idx in kf:
            logging.info('working on k-fold validation')
            test_out = self.Mdl.train_out[test_idx]

            # fit model
            self.Mdl._cvtrain(train_idx)
            test_pred = self.Mdl._cvpredict(test_idx)

            # get score
            self.mdl_score.append(self.compute_score(test_out, test_pred, metric))

            logging.info('Current model score: %f'%(self.mdl_score[-1]))

        # return score mean and std
        logging.info('Model score: Mean: %f; Std: %f'%(np.mean(self.mdl_score), np.std(self.mdl_score)))


    def compute_score(self, test_out, test_pred, metric):
        """
        Evalute model performance
        :param test_out: true label
        :param test_pred: prediceted label
        :param metric: evaluation metric
        :return:
        """
        if(metric=='logloss'):
            return log_loss(test_out, test_pred)
        elif(metric=='auc'):
            fpr, tpr, _ = roc_curve(test_out, test_pred)
            return auc(fpr, tpr)
        else:
            raise Exception('Undefined evalution metric')
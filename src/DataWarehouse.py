__author__ = 'randxie'

import os
import pandas as pd
import logging

# customized setting
from common import DATA_TRAIN, DATA_TEST, DATA_ID, DATA_OUT_FEATURE
from common import SUBMIT_ID, SUBMISSION_FOLDER

class DataWarehouse():
    """A class to handle data IO
    """

    def __init__(self):
        pass

    def read_data(self):
        """Read in train and test data. If the data is in multiple files, need more work to separate data and ID.
        """
        self.train_in = pd.read_csv(DATA_TRAIN)
        self.test_in = pd.read_csv(DATA_TEST)

        # get response variable and id
        self.train_out = self.train_in[DATA_OUT_FEATURE]
        self.train_id = self.train_in[DATA_ID]

        # remove unnecessary information
        del self.train_in[DATA_ID]
        del self.train_in[DATA_OUT_FEATURE]

        # remove unnecessary information from test
        self.test_id = self.test_in[SUBMIT_ID]
        del self.test_in[SUBMIT_ID]

    def generate_submission(self, ypred):
        """
        Generate submission given predicted output
        :param ypred: predicted output corresponding to test_id
        :return: none
        """
        out_df = pd.DataFrame(ypred)
        out_df.columns = [DATA_OUT_FEATURE]
        out_df[SUBMIT_ID] = self.test_id
        out_df.to_csv(os.path.join(SUBMISSION_FOLDER, "submission.csv"), index=False)


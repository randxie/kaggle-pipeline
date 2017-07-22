# load infrastructure
from DataWarehouse import DataWarehouse
from FeatureGenerator import FeatureGenerator
from Model import Model
from Stacker import Stacker
from Validator import Validator

# load config
from common import set_logging_config, get_model_config

# set logging config
set_logging_config()

if __name__ == "__main__":
    # read in data
    MyData = DataWarehouse()
    MyData.read_data()

    # read in configuration
    config = get_model_config('models/basic.yml')
    MyFeature = FeatureGenerator(MyData, config)
    MyFeature.compute_features()

    # initialize models
    mdl_type = 'lr'
    model_xgb = Model(MyFeature, mdl_type, config['models'][mdl_type])

    # use validator to do k-fold validation
    MyValidator = Validator(model_xgb)
    MyValidator.validate_kfold(4, metric='auc')

    #-------- Stacking ------------------
    mdl_type = 'xgb'
    stack_mdl = Model(MyFeature, mdl_type, config['models'][mdl_type])

    mdl_list = [model_xgb]
    base_feature = MyFeature
    stacker = Stacker(mdl_list, base_feature, stack_mdl, num_stack=4)

    MyValidator = Validator(stacker)
    MyValidator.validate_kfold(4, metric='auc')

    #---------Generating outputs ---------------
    stacker.train_all()
    ypred = stacker.predict_all()
    MyData.generate_submission(ypred)
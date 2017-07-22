# Kaggle Pipeline
The design of pipeline is based on my experience in Kaggle competitions and other data challenges. In this pipeline, I try to decouple different steps from data IO, feature extraction, building model, validation to stacking. The main purpose is for fast protytyping (not scalable). In terms of stacking, I am not a fan of multiple level stacking.

### Overview

[Defense against adversarial attack](https://www.kaggle.com/c/nips-2017-defense-against-adversarial-attack)

# File Structure
```
├── input  <-- train and test data
├── README.md
├── submission <--- final submission folder
├── src        <-- where code is stored
│   ├── main.py <--- main process
│   ├── DataWarehouse.py <--- handle data IO
│   ├── FeatureGenerator.py <--- convert raw data to feature matrix
│   ├── FeatureGenerator.py <--- convert raw data to feature matrix
│   ├── Model.py <--- used to initialize different machine learning models
│   ├── Stacker.py <--- stacking different models together with feature matrix
│   ├── Validator.py <--- k-fold validation of Model/Stacker
│   ├── common.py <--- define where files are stored and setup logging
│   ├── config <--- storing config files for stacking, models, features to extract
```

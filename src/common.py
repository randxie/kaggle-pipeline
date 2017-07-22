# training/testing file path
import os, logging, yaml

SRC_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_FOLDER = os.path.join(SRC_PATH, '../input/')

# define data csv files
DATA_TRAIN = os.path.join(DATA_FOLDER, 'train.csv')
DATA_TEST = os.path.join(DATA_FOLDER, 'test.csv')

# define ID and output variable
DATA_ID = 'PassengerId'
DATA_OUT_FEATURE = 'Survived'

# define submission
SUBMISSION_FOLDER = os.path.join(SRC_PATH, '../submission/')
SUBMIT_ID = 'PassengerId'

# logging and debug setting
LOG_FILE = 'record.log'
IS_DEBUG_MODE = 0

def set_logging_config():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=LOG_FILE,
                        filemode='w')

    # output logging info to screen as well (from python official website)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

# function for parsing yml file
def get_model_config(config_file):
    # read in config for related models
    with open(config_file, 'r') as f:
        config = yaml.load(f)
    return config
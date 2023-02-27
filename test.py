import OS
import argparse
import
torch
import lightgbm as lgb
import xgboost as xgb
import pandas as pd
from utils. data import seed_ worker, load_ pytorch_ test_ data, load _tree_ test_ data
om utils . model
import load pytorch_ model, count_ par ameters
from utils.misc
import same_ seed,
ad_ config, get_ logger
From utils. predict import pytorch_ predict, tree_ predict
if name__ == ' main_ ' :
# get config
parser = argparse. ArgumentParser(descript ion= ' Pytorch Template ') 
parser . add_ argument('-c', '--config', default='./config/LSTM 01.yaml', type=str,
nelp= ' config yaml file path (default: None) )
args = parser.parse args()
config = load_ config(args . config)
# get logger
if os.path. exists(os . path. join(config['test']['log_ save_ path'], 'test_ log. txt' )):
os .remove(os . path. join(config['test']
['log. save_ path'], 'test_ log. txt'))
logger = get_ _logger( 'info', os . path. join(
config['test']['log_ save_ path'], 'test_ log. txt')) 
logger .info(' ##
######## #####
############### )
logger .1nto( #### welcome to use Deepalpha ####### )
pgger . inf
##########
#####
####### ### #########
logger . info('Using test mode...'
# fixed all random seeds for reproducibility
same seed(0)
g = torch. Generator()
g . manual_ seed(0) 
# testing
logger . info('{} model is testing on {}'. format(
config[ ' system' ]['model_ name'], config[ 'system'][ ' phase']))
if config[ 'system']['ispytorch']:
####### PyTorch model #########
model = load_ pytorch_ _model(config)
model. load state_ dict(torch. load(
config[ 'training'][' model save path']))
logger . info( 'Total number of parameters is {: .2f}KB ({:. 2f}MB). '. format(
count parameters (model, "k"), count parameters ( model, "m") ))
test_ loader, test_ indices, test_ df = load_ pytorch_ test_ data(
config, seed_ worker)
logger . info( 'Test data load sucessfully.')
loader,
model, config)
if config[ 'system'][ 'istree']:
test_ data, test_ indices, test_ df = load _tree test. data(
est_ da
logger . info( 'Test data load sucessfully.')
if config[ 'system' ][ 'model_ name'] == 'L ightGBM' :
########## L ightGBM model ######## #####
model = lgb. Booster(
model_ file=config[ 'training' ]['model_ save_ path' ])
if config[ 'system' ][ 'model name'] == ' XGBoost':
######### XGBoost model #### ## ## 
model = xgb . Booster(
model_ file=config[ 'training ' ][ 'model_ save_ path' ])
preds = tree_ predict(test_ data, model )
# construct pred file and save
results = pd . DataF rame (test_ indices, columns=[
results.insert(loc=3, column= y _hat'，value=preds)
results = pd. merge
results = results[['ukey', 'DataDate', 'ticktime', 'flag', 'y', 'y_ hat']]
results.to feather(config[ 'test']['pred_ save_ path'])
# print out evaluation metrics
evaluation(results, config, logger )
# visualization and save figures
visualization rcor(results, config)
logger . info('Testing ends.')

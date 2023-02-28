import torch
from tqdm import tqdm
import xgboost as xgb


def pytorch_predict(test_loader, model, config):
    model.eval()  # Set your model to evaluation mode.
    preds = []
    for x in tqdm(test_loader):
        x = x[0].to(config['system']['device'])
    with torch.no_grad():
        if config['system']['model_name'] == 'TabNet':
            priors = torch.ones(x.shape[0], config['model']['inp_dim']).to(
                config['system']['device'])
            pred = model(x, priors)
            pred = pred[0].squeeze(1)
        if config['system']['model_name'] in ['MTL', 'MTLformer']:
            pred, _ = model(x)
        else:
            pred = model(x)
        preds.append(pred.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy
    return preds


def tree_predict(test_data, model):
    try:
        return model.predict(test_data)
    except:
        test_data = xgb.DMatrix(test_data)
        return model.predict(test_data)
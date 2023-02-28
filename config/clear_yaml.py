# Clear results folder based on model and id
import os
import re
import shutil
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pytorch Template')
    parser.add_argument('-c',
                        '--config',
                        default='LSTM/pahse1/LSTM_01.yaml',
                        type=str,
                        help='config yaml file path')

    args = parser.parse_args()
    phase, model, id = re.findall(r"/(.+?)/(.+?)_(.+?).yaml", args.config)[0]
    # remove config file
    os.remove(args.config)

    # remove subfolders in ../results
    for folder in ['analysis', 'ckpts', 'logs', 'preds', 'tsboards']:
        subfolder = os.path.join('../results/{}/'.format(folder), model, phase,
                                 id)
    if os.path.exists(subfolder):
        shutil.rmtree(subfolder)
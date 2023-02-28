# Change yaml configuration based on the model name and id

import re
import yaml
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pytorch')
    parser.add_argument('-c-config',
                        default='Linear/phase1/Linear_01.yaml',
                        type=str,
                        help='config yaml file path')

    args = parser.parse_args()
    phase, model, id = re.findall(r"/(.+?)/(.+?)_(.+?).yaml", args.config)[0]

with open(args.config, 'r') as f:
    content = yaml.load(f.read(), Loader=yaml.FullLoader)

    content['system']['model_name'] = model
    content['system']['experimental_id'] = id
    content['system']['phase'] = phase
    content['system']['device'] = 'cuda:{}'.format(int(phase[-1]) - 1)

    content['test'][
        'figures_save_path'] = "./results/analysis/{}/{}/{}".format(
            model, phase, id)
    content['test'][
        'pred_save_path'] = "./results/preds/{}/{}/{}/preds.feather".format(
            model, phase, id)
    content['test']['log_save_path'] = "./results/logs/{}/{}/{}/".format(
        model, phase, id)
    content['training']['log_save_path'] = "./results/logs/{}/{}/{}/".format(
        model, phase, id)
    content['training'][
        'tsboard_save_path'] = "./results/tsboards/{}/{}/{}/".format(
            model, phase, id)
    content['training'][
        'model_save_path'] = "./results/ckpts/{}/{}/{}/model.ckpt".format(
            model, phase, id)
    content['training']['data_load_path'] = "./temp/data/{}".format(phase)
    content['training']['seq_data_load_path'] = "./temp/seq_data/{}".format(
        phase)

    with open(args.config, 'w') as wf:
        yaml.dump(content, wf)
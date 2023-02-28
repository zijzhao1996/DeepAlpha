import sys
from model import *
import torch.nn as nn

sys.path.append('..')


def load_pytorch_model(config):
    """
    Load deep learning models built via PyTorch.
    """
    if config['system']['model_name'] == 'ALSTM':
        model = ALSTM(d_feat=config['model']['d_feat'],
                      hidden_size=config['model']['hidden_size'],
                      num_layers=config['mode1']['num_layers'],
                      dropout=config['model']['dropout']).to(
                          config['system']['device'])
    elif config['system']['model_name'] == 'LSTM':
        model = LSTM(d_feat=config['model']['d_feat'],
                     hidden_size=config['model']['hidden_size'],
                     num_layers=config['model']['num_layers'],
                     dropout=config['model']['dropout']).to(
                         config['system']['device'])
    elif config['system']['model_name'] == 'Transformer':
        model = Transformer(d_feat=config['model']['d_feat'],
                            d_model=config['model']['d_model'],
                            nhead=config['model']['nhead'],
                            num_layers=config['model']['num_layers'],
                            dim_feedforward=config['model']['dim_feedforward'],
                            dropout=config['model']['dropout']).to(
                                config['system']['device'])
    elif config['system']['model_name'] == 'GRNformer':
        model = GRNformer(d_feat=config['model']['d_feat'],
                          d_model=config['model']['d_model'],
                          nhead=config['model']['nhead'],
                          num_layers=config['model']['num_layers'],
                          dim_feedforward=config['model']['dim_feedforward'],
                          dropout=config['model']['dropout']).to(
                              config['system']['device'])
    elif config['system']['model_name'] == 'GATs':
        model = GATs(d_feat=config['model']['d_feat'],
                     hidden_size=config['model']['hidden_size'],
                     num_layers=config['model']['num_layers'],
                     dropout=config['model']['dropout'],
                     base_model=config['model']['base_model']).to(
                         config['system']['device'])
    elif config['system']['model_name'] == 'Localformer':
        model = Localformer(d_feat=config['model']['d_feat'],
                            d_model=config['model']['d_model'],
                            nhead=config['model']['nhead'],
                            num_layers=config['model']['num_layers'],
                            dim_feedforward=config['model']['dim_feedforward'],
                            dropout=config['model']['dropout']).to(
                                config['system']['device'])
    elif config['system']['model name'] == 'Ultraformer':
        model = Ultraformer(d_feat=config['model']['d_feat'],
                            d_model=config['model']['d_model'],
                            nhead=config['model']['nhead'],
                            num_layers=config['model']['num_layers'],
                            dim_feedforward=config['model']['dim_feedforward'],
                            dropout=config['model']['dropout']).to(
                                config['system']['device'])
    elif config['system']['model_name'] == 'TCN':
        model = TCN(num_input=config['model']['num_input'],
                    output_size=config['model']['output_size'],
                    num_channels=config['model']['num_channels'],
                    kernel_size=config['model']['kernel_size'],
                    dropout=config['model']['dropout']).to(
                        config['system']['device'])
    elif config['system']['model_name'] == 'MLP':
        model = MLP(input_dim=config['model']['input_dim'],
                    output_dim=config['model']['output_dim'],
                    layers=config['model']['layers'],
                    act=config['model']['activation']).to(
                        config['system']['device'])
    elif config['system']['model_name'] == 'MLPMoE':
        model = MLPMoE(input_size=config['model']['input_size'],
                       output_size=config['model']['output_size'],
                       num_experts=config['model']['num_experts'],
                       hidden_size=config['model']['hidden_size'],
                       noisy_gating=config['model']['noisy_gating'],
                       k=config['model']['k']).to(config['system']['device'])
    elif config['system']['model name'] == 'Linear':
        model = Linear(input_dim=config['model']['input_dim']).to
        (config['system']['device'])
    elif config['system']['model_name'] == 'TabNet':
        model = TabNet(inp_dim=config['model']['inp_dim'],
                       out_dim=config['model']['out_dim'],
                       n_d=config['model']['n_d'],
                       n_a=config['model']['n_a'],
                       n_shared=config['model']['n_shared'],
                       n_ind=config['model']['n_ind'],
                       n_steps=config['model']['n_steps'],
                       relax=config['model']['relax'],
                       vbs=config['model']['vbs']).to(
                           config['system']['device'])
    elif config['system']['model_name'] == 'Informer':
        model = Informer(enc_in=config['model']['enc_in'],
                         dec_in=config['model']['dec_in'],
                         label_len=config['model']['label_len'],
                         attn=config['model']['attention'],
                         d_model=config['model']['d_model'],
                         n_heads=config['mode1']['n_heads'],
                         factor=config['model']['factor'],
                         e_layers=config['model']['e_layers'],
                         d_layers=config['model']['d_layers'],
                         d_ff=config['model']['d_ff'],
                         dropout=config['model']['dropout'],
                         activation=config['model']['activation'],
                         output_attention=config['model']['output_attention'],
                         decode_method=config['model']['decode_method'],
                         distil=config['model']['distil']).to(
                             config['system']['device'])
    elif config['system']['model_name'] == 'Autoformer':
        model = Autoformer(
            enc_in=config['model']['enc_in'],
            dec_in=config['model']['dec_in'],
            d_model=config['model']['d_model'],
            n_heads=config['model']['n_heads'],
            factor=config['model']['factor'],
            e_layers=config['mode1']['e_layers'],
            d_layers=config['model']['d_layers'],
            d_ff=config['model']['d_ff'],
            dropout=config['model']['dropout'],
            activation=config['model']['activation'],
            output_attention=config['model']['output_attention'],
            kernel_size=config['model']['kernel_size'],
            moving_avg=config['model']['moving_avg'],
            decode_method=config['model']['decode_method']).to(
                config['system']['device'])
    elif config['system']['model_name'] == 'Fedformer':
        model = Fedformer(
            enc_in=config['model']['enc_in'],
            dec_in=config['model']['dec_in'],
            modes=config['model']['modes'],
            mode_select_method=config['model']['mode_select_method'],
            d_model=config['model']['d_model'],
            n_heads=config['model']['n_heads'],
            e_layers=config['model']['e_layers'],
            d_layers=config['model']['d_layers'],
            d_ff=config['model']['d_ff'],
            label_len=config['model']['label_len'],
            dropout=config['model']['dropout'],
            activation=config['model']['activation'],
            seq_len=config['model']['seq_len'],
            kernel_size=config['model']['kernel_size'],
            moving_avg=config['model']['moving_avg'],
            decode_method=config['model']['decode_method']).to(
                config['system']['device'])
    elif config['system']['model_name'] == 'Mixformer':
        model = Mixformer(enc_in=config['model']['enc_in'],
                          d_feat=config['model']['d_feat'],
                          d_model=config['model']['d_model'],
                          n_heads=config['model']['n_heads'],
                          factor=config['model']['factor'],
                          e_layers=config['model']['e_layers'],
                          d_layers=config['model']['d_layers'],
                          d_ff=config['model']['d_ff'],
                          dropout=config['model']['dropout'],
                          activation=config['model']['activation'],
                          output_attention=config['model']['output_attention'],
                          kernel_size=config['model']['kernel_size'],
                          moving_avg=config['model']['moving_avg'],
                          decode_method=config['model']['decode_method']).to(
                              config['system']['device'])
    elif config['system']['model_name'] == 'SFM':
        model = SFM(d_feat=config['model']['d_feat'],
                    output_dim=config['model']['output_dim'],
                    freq_dim=config['model']['freq_dim'],
                    hidden_size=config['model']['hidden_size'],
                    dropout_W=config['model']['dropout_W'],
                    dropout_U=config['model']['dropout_U'],
                    device=config['system']['device']).to(
                        config['system']['device'])
    elif config['system']['model_name'] == 'TFT':
        model = TFT(
            static_vars=config['model']['static_vars'],
            enc_dim=config['model']['enc_dim'],
            time_cat_vars=config['model']['time_cat_vars'],
            time_real_vars_enc=config['model']['time_real_vars_enc'],
            time_real_vars_dec=config['model']['time_real_vars_dec'],
            num_masked_series=config['model']['num_masked_series'],
            hidden_size=config['model']['hidden_size'],
            lstm_num_layers=config['model']['lstm_num_layers'],
            dropout=config['model']['dropout'],
            embed_dim=config['model']['embed_dim'],
            n_heads=config['model']['n_heads'],
            n_quantiles=config['model']['n_quantiles'],
            valid_quantiles=config['model']['valid_quantiles'],
            seq_length=config['model']['seq_length'],
            static_embed_vocab_size=config['model']['static_embed_vocab_size'],
            time_embed_vocab_size=config['model']['time_embed_vocab_size'],
            batch_size=config['training']['batch_size'],
            device=config['system']['device']).to(config['system']['device'])
    elif config['system']['model_name'] == 'DLinear':
        model = DLinear(input_dim=config['model']['input_dim'],
                        seq_length=config['model']['seq_length'],
                        kernel_size=config['model']['kernel_size']).to(
                            config['system']['device'])
    elif config['system']['model name'] == 'MTL':
        model = MTL(
            input_dim=config['model']['input_dim'],
            shared_hidden_size=config['model']['shared_hidden_size'],
            intra_tower_hidden_size=config['model']['intra_tower_hidden_size'],
            cross_tower_hidden_size=config['model']['cross_tower_hidden_size'],
            output_dim=config['model']['output_dim']).to(
                config['system']['device'])
    elif config['system']['model_name'] == 'MTLformer':
        model = MTLformer(
            d_feat=config['model']['d_feat'],
            d_model=config['model']['d_model'],
            intra_tower_hidden_size=config['model']['intra_tower_hidden_size'],
            cross_tower_hidden_size=config['model']['cross_tower_hidden_size'],
            nhead=config['model']['nhead'],
            num_layers=config['model']['num_layers'],
            dim_feedforward=config['model']['dim_feedforward'],
            dropout=config['model']['dropout']).to(config['system']['device'])
    else:
        raise Exception("unknown model name {}".format(
            config['system']['model_name']))

    return model


def count_parameters(models_or_parameters, unit="k"):
    """
    Obtain the storage size unit of a (or multiple) models.
    """
    if isinstance(models_or_parameters, nn.Module):
        counts = sum(v.numel() for v in models_or_parameters.parameters())
    elif isinstance(models_or_parameters, nn.Parameter):
        counts = models_or_parameters.numel()
    elif isinstance(models_or_parameters, (list, tuple)):
        return sum(count_parameters(x, unit) for x in models_or_parameters)
    else:
        counts = sum(v.numel() for v in models_or_parameters)
    unit = unit.lower()
    if unit in ("kb", "k"):
        counts /= 2**10
    elif unit in ("mb", "m"):
        counts /= 2**20
    elif unit in ("gb", "g"):
        counts /= 2**30
    elif unit is not None:
        raise ValueError("Unknown unit: {:}".format(unit))
    return counts
import torch


def mse_loss(output, target, config=None):
    loss = torch.mean((output - target)**2)
    return loss


def custom_loss(pred, y, config):
    # increase the postive loss weights
    pos_y = torch.masked_select(y, y > 0)
    pos_pred = torch.masked_select(pred, y > 0)
    pos_loss = torch.mean((pos_pred - pos_y)**2)

    neg_y = torch.masked_select(y, y < 0)
    neg_pred = torch.masked_select(pred, y < 0)
    neg_loss = torch.mean((neg_pred - neg_y)**2)

    total_loss = config['training']['lambda'] * pos_loss + (
        1 - config['training']['lambda ']) * neg_loss

    return total_loss
from torch import nn


class TransformerLoss(nn.Module):
    def __init__(self):
        super(TransformerLoss, self).__init__()

    def forward(self, model_outputs, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_output, mel_output_postnet, gate_output, *_ = model_outputs
        gate_output = gate_output.view(-1, 1)

        mel_loss = 0.5 * nn.MSELoss()(mel_output, mel_target) + \
            0.5 * nn.MSELoss()(mel_output_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_output, gate_target)
        return mel_loss + gate_loss

import os
import time
import argparse
import matplotlib.pylab as plt
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import hparams
from data_utils import TextMelLoader, TextMelCollate
from model import Transformer
from utils import parse_batch


def plot_data(data, index, path, figsize=(10, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(
            data[i], aspect="auto", origin="bottom", interpolation="none"
        )
    file = os.path.join(path, str(index) + "-mel.png")
    plt.savefig(file)
    plt.close()


def plot_attn(enc_attn_list, dec_attn_list, dec_enc_attn_list, index, path):
    # enc_attn_list[0]'s shape (b, h, lq, lk)
    layer_num = len(enc_attn_list)
    heads_num = enc_attn_list[0].size(1)

    # encoder attn image
    fig, axes = plt.subplots(layer_num, heads_num, figsize=(16, 12))
    for i in range(layer_num):
        for j in range(heads_num):
            # (lk, lq)
            enc_attn = enc_attn_list[i][0, j].detach().cpu().numpy().T
            axes[i][j].imshow(
                enc_attn, aspect="auto", origin="bottom", interpolation="none"
            )
    file = os.path.join(path, str(index) + "-enc-attn.png")
    plt.savefig(file)
    plt.close()

    # decoder attn image
    fig, axes = plt.subplots(layer_num, heads_num, figsize=(16, 12))
    for i in range(layer_num):
        for j in range(heads_num):
            # (lk, lq)
            enc_attn = dec_attn_list[i][0, j].detach().cpu().numpy().T
            axes[i][j].imshow(
                enc_attn, aspect="auto", origin="bottom", interpolation="none"
            )
    file = os.path.join(path, str(index) + "-dec-attn.png")
    plt.savefig(file)
    plt.close()

    # decoder-encoder image
    fig, axes = plt.subplots(layer_num, heads_num, figsize=(16, 12))
    for i in range(layer_num):
        for j in range(heads_num):
            # (lk, lq)
            enc_attn = dec_enc_attn_list[i][0, j].detach().cpu().numpy().T
            axes[i][j].imshow(
                enc_attn, aspect="auto", origin="bottom", interpolation="none"
            )
    file = os.path.join(path, str(index) + "-dec-enc-attn.png")
    plt.savefig(file)
    plt.close()


def main(args, hparams):
    # prepare data
    # warning: during the inference the batch_size should always be set to 1
    testset = TextMelLoader(hparams.test_files, hparams, shuffle=False)
    collate_fn = TextMelCollate(hparams.n_frames_per_step)
    test_loader = DataLoader(
        testset,
        num_workers=1,
        shuffle=False,
        batch_size=1,
        pin_memory=False,
        collate_fn=collate_fn,
    )

    # prepare the transformer model
    model = Transformer(hparams).cuda(device="cuda:0")
    checkpoint_restore = torch.load(args.checkpoint_path)["state_dict"]
    model.load_state_dict(checkpoint_restore)
    model.eval()
    print('# total parameters:', sum(p.numel() for p in model.parameters()))

    # infer
    duration_add = 0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader)):
            x, y = parse_batch(batch)

            # the start time
            start = time.perf_counter()
            (
                mel_output,
                mel_output_postnet,
                _,
                enc_attn_list,
                dec_attn_list,
                dec_enc_attn_list,
            ) = model.inference(x)

            # the end time
            duration = time.perf_counter() - start
            duration_add += duration

            # save the mels and attention plots
            mel_path = os.path.join(args.output_infer, str(i) + ".pt")
            torch.save(mel_output_postnet[0], mel_path)

            plot_data(
                (
                    mel_output.detach().cpu().numpy()[0],
                    mel_output_postnet.detach().cpu().numpy()[0],
                ),
                i,
                args.output_infer,
            )
            plot_attn(
                enc_attn_list,
                dec_attn_list,
                dec_enc_attn_list,
                i,
                args.output_infer,
            )

        duration_avg = duration_add / (i + 1)
        print("The average inference time is: %f" % duration_avg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output_infer",
        type=str,
        default="output_infer",
        help="directory to save infer outputs",
    )
    parser.add_argument(
        "-c",
        "--checkpoint_path",
        type=str,
        default=None,
        required=False,
        help="checkpoint path for infer model",
    )
    args = parser.parse_args()
    os.makedirs(args.output_infer, exist_ok=True)
    assert args.checkpoint_path is not None

    main(args, hparams)
    print("finished")

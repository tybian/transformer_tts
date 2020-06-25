import os
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import hparams as hparams
from model import Transformer
from data_utils import TextMelLoader, TextMelCollate
from loss_function import TransformerLoss
from logger import TransformerLogger
from utils import parse_batch


def lr_schdule(optimizer, iteration):
    _iteration = iteration + 1
    lr = hparams.learning_rate * min(
        _iteration * hparams.warmup_step ** -1.5, _iteration ** -0.5
    )

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return optimizer, lr


def load_model(hparams):
    model = Transformer(hparams).cuda(device="cuda:0")
    if torch.cuda.device_count() > 1:
        print("let's use", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model, device_ids=["cuda:0", "cuda:1"])
    return model


def prepare_directories_and_logger(output_directory, log_directory):
    os.makedirs(output_directory, exist_ok=True)
    os.chmod(output_directory, 0o775)
    os.makedirs(log_directory, exist_ok=True)
    os.chmod(log_directory, 0o775)
    logger = TransformerLogger(log_directory)
    return logger


def prepare_dataloaders(hparams):
    trainset = TextMelLoader(hparams.training_files, hparams)
    valset = TextMelLoader(hparams.validation_files, hparams)
    collate_fn = TextMelCollate(hparams.n_frames_per_step)

    train_loader = DataLoader(
        trainset,
        num_workers=1,
        shuffle=True,
        batch_size=hparams.batch_size,
        pin_memory=False,
        drop_last=True,
        collate_fn=collate_fn,
    )

    valid_loader = DataLoader(
        valset,
        num_workers=1,
        shuffle=False,
        batch_size=hparams.batch_size,
        pin_memory=False,
        collate_fn=collate_fn,
    )
    return train_loader, valid_loader


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint_dict["state_dict"])
    optimizer.load_state_dict(checkpoint_dict["optimizer"])
    learning_rate = checkpoint_dict["learning_rate"]
    iteration = checkpoint_dict["iteration"]
    print(
        "Loaded checkpoint '{}' from iteration {}".format(
            checkpoint_path, iteration
        )
    )
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print(
        "Saving model and optimizer state at iteration {} to {}".format(
            iteration, filepath
        )
    )
    torch.save(
        {
            "iteration": iteration,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        filepath,
    )


def validate(model, criterion, valid_loader):

    """Handles all the validation scoring and printing"""
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for i, batch in enumerate(valid_loader):
            x, y = parse_batch(batch)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            val_loss += loss.item()
        val_loss = val_loss / (i + 1)

    model.train()
    return val_loss


def train(
    model,
    optimizer,
    criterion,
    train_loader,
    valid_loader,
    iteration,
    epoch_offset,
    logger,
    output_directory,
):

    model.train()
    model.zero_grad()

    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, hparams.epochs):
        print("Epoch: {}".format(epoch))

        for batch in train_loader:
            optimizer, learning_rate = lr_schdule(optimizer, iteration)

            # the start time
            x, y = parse_batch(batch)
            start = time.perf_counter()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()

            # the end time
            duration = time.perf_counter() - start

            # clip abnormal gradient
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), hparams.grad_clip_thresh
            )

            # accumulate gradients
            # if iteration % 2 == 0:
            optimizer.step()
            model.zero_grad()

            print(
                "Train loss {} {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(
                    iteration, loss.item(), grad_norm, duration
                )
            )
            logger.log_training(
                loss.item(), grad_norm, learning_rate, duration, iteration
            )

            # validate the model
            if iteration % hparams.iters_per_checkpoint == 0:
                val_loss = validate(model, criterion, valid_loader)

                print(
                    "Validation loss {}: {:9f}  ".format(iteration, val_loss)
                )

                if torch.cuda.device_count() > 1:
                    # for data parallel
                    logger.log_validation(
                        val_loss, model.module, y, y_pred, iteration
                    )
                else:
                    logger.log_validation(
                        val_loss, model, y, y_pred, iteration
                    )

                checkpoint_path = os.path.join(
                    output_directory, "checkpoint_{}".format(iteration)
                )
                if torch.cuda.device_count() > 1:
                    # for data parallel
                    save_checkpoint(
                        model.module,
                        optimizer,
                        learning_rate,
                        iteration,
                        checkpoint_path,
                    )
                else:
                    save_checkpoint(
                        model,
                        optimizer,
                        learning_rate,
                        iteration,
                        checkpoint_path,
                    )
            iteration += 1


def main(output_directory, log_directory, checkpoint_path):
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    model = load_model(hparams)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hparams.learning_rate,
        betas=(hparams.adam_beta1, hparams.adam_beta2),
        eps=hparams.adam_eps,
        weight_decay=hparams.weight_decay,
    )
    criterion = TransformerLoss()
    logger = prepare_directories_and_logger(output_directory, log_directory)
    train_loader, valid_loader = prepare_dataloaders(hparams)

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    if checkpoint_path is not None:
        if torch.cuda.device_count > 1:
            # for data parallel
            model.module, optimizer, _, iteration = load_checkpoint(
                checkpoint_path, model.module, optimizer
            )
        else:
            model, optimizer, _, iteration = load_checkpoint(
                checkpoint_path, model, optimizer
            )
        iteration += 1  # next iteration is iteration + 1
        epoch_offset = max(0, int(iteration / len(train_loader)))

    train(
        model,
        optimizer,
        criterion,
        train_loader,
        valid_loader,
        iteration,
        epoch_offset,
        logger,
        output_directory,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output_directory",
        type=str,
        default="/home/server/disk1/checkpoints/transformer_tts/exp",
        help="directory to save checkpoints",
    )
    parser.add_argument(
        "-l",
        "--log_directory",
        type=str,
        default="/home/server/disk1/checkpoints/transformer_tts/log/exp",
        help="directory to save tensorboard logs",
    )
    parser.add_argument(
        "-c",
        "--checkpoint_path",
        type=str,
        default=None,
        required=False,
        help="checkpoint path",
    )
    args = parser.parse_args()

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)

    main(
        args.output_directory, args.log_directory, args.checkpoint_path,
    )

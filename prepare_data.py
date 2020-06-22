import torch
from torch.utils.data import Dataset, DataLoader

import os
import librosa
import numpy as np
from tqdm import tqdm

from stft import TacotronSTFT
from utils import preemphasis
import hparams as hparams


class PrepareData(Dataset):
    """
    Preparing the wav files to mel spectrograms and saving them on disk for
    speeding up training. The generated mel spectrograms are saved in the same
    path with .wav files.
    """

    def __init__(self, data_path, metadata_path):
        self.wav_list = []
        self.wav_path = data_path
        self.stft = TacotronSTFT(
            hparams.filter_length,
            hparams.hop_length,
            hparams.win_length,
            hparams.n_mel_channels,
            hparams.sampling_rate,
            hparams.mel_fmin,
            hparams.mel_fmax,
        )

        with open(metadata_path, "r") as f:
            for line in f:
                line = line.split("|")
                self.wav_list.append(line[0].strip())

    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, index):
        return self._get_mel(self.wav_list[index])

    def _get_mel(self, file):
        wav = os.path.join(self.wav_path, file + ".wav")
        mel = os.path.join(self.wav_path, file + ".mel.npy")

        # Loading sound file
        audio, sr = librosa.load(wav, sr=hparams.sampling_rate)

        # Trimming
        audio, _ = librosa.effects.trim(audio, top_db=55)

        # which normalization to use, Tacotron1 or Tacotron2
        if hparams.tacotron1_norm is True:
            audio = preemphasis(audio)

        # stft, using the same stft with data_utils.py
        audio_norm = torch.FloatTensor(audio.astype(np.float32)).unsqueeze(0)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = melspec.squeeze(0)
        np.save(mel, melspec.numpy())
        return melspec


if __name__ == "__main__":

    # data_path and metadata_path should be redefined as the case may be
    data_path = "/home/server/disk1/DATA/LJS/LJSpeech-1.1/wavs"
    metadata_path = "/home/server/disk1/DATA/LJS/LJSpeech-1.1/metadata.csv"
    dataset = PrepareData(data_path, metadata_path)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=2)
    for _ in tqdm(dataloader, desc="Preparing the specs: "):
        pass
    print("Preparing is finished !")

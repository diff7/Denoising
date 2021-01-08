import os

import librosa
import numpy as np
from torch.utils import data
from torch.utils.data import Dataset

# from util.utils import sample_fixed_length_data_aligned


def sample_fixed_length_data_aligned(data_a, data_b, sample_len):
    """sample with fixed length from two dataset"""
    assert len(data_a) == len(
        data_b
    ), "Inconsistent dataset length, unable to sampling"
    assert (
        len(data_a) >= sample_len
    ), f"len(data_a) is {len(data_a)}, sample_len is {sample_len}."

    frames_total = len(data_a)

    start = np.random.randint(frames_total - sample_len + 1)
    # print(f"Random crop from: {start}")
    end = start + sample_len

    return data_a[start:end], data_b[start:end]


def random_cut(signal, cut_len=0.05):
    num_cuts = np.random.randint(0, 5)
    cut_len = int(signal.shape[-1] * cut_len)
    for _ in range(num_cuts):
        start = np.random.randint(0, signal.shape[-1])
        signal[start : start + cut_len] = 0
    return signal


class DatasetAudio(data.Dataset):
    def __init__(
        self,
        file_path,
        sample_len=16384,
        mode="train",
        shift=0,
        use_random_cut=True,
    ):
        """Construct dataset for training and validation.
        Args:
            dataset (str): *.txt, the path of the dataset list file. See "Notes."
            limit (int): Return at most limit files in the list. If None, all files are returned.
            offset (int): Return files starting at an offset within the list. Use negative values to offset from the end of the list.
            sample_len(int): The model only supports fixed-length input. Use sample_len to specify the feature size of the input.
            mode(str): If mode is "train", return fixed-length signals. If mode is "validation", return original-length signals.

        Notes:
            dataset list file：
            <clean_1_path><space><noisy_1_path>
            <clean_2_path><space><noisy_2_path>
            ...
            <noisy_n_path><space><clean_n_path>

            e.g.
            /train/noisy/a.wav /train/clean/a.wav
            /train/noisy/b.wav /train/clean/b.wav
            ...

        Return:
            (mixture signals, clean signals, filename)
        """
        super(Dataset, self).__init__()
        dataset_list = [
            line.rstrip("\n").rstrip()
            for line in open(
                os.path.abspath(os.path.expanduser(file_path)), "r"
            )
        ]

        # dataset_list = dataset_list[offset:]
        # if limit:
        #    dataset_list = dataset_list[:limit]

        assert mode in (
            "train",
            "validation",
        ), "Mode must be one of 'train' or 'validation'."

        self.length = len(dataset_list)
        self.dataset_list = dataset_list
        self.sample_len = sample_len + shift
        self.mode = mode
        self.use_random_cut = use_random_cut

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        clean_path, mixture_path = self.dataset_list[item].split(" ")
        filename = os.path.splitext(os.path.basename(mixture_path))[0]
        mixture, _ = librosa.load(
            os.path.abspath(os.path.expanduser(mixture_path)), sr=None
        )

        clean, _ = librosa.load(
            os.path.abspath(os.path.expanduser(clean_path)), sr=None
        )

        assert len(clean) == len(mixture)

        if self.mode == "train":
            # The input of model should be fixed-length in the training.
            mixture, clean = sample_fixed_length_data_aligned(
                mixture, clean, self.sample_len
            )
            if self.use_random_cut:
                mixture = random_cut(mixture)

            return mixture.reshape(1, -1), clean.reshape(1, -1), filename
        else:
            # TODO Rewrite val with collate to match longest file in the batch.
            return mixture.reshape(1, -1), clean.reshape(1, -1), filename

import os
import re

import numpy as np
import pandas as pd
from torch.utils.data.dataset import T_co
from tqdm import tqdm

from lib.datasets.generic_action_dataset import GenericActionDataset

sims_simple_dataset_encoding = {
    "Cook":         0,
    "Drink":        1,
    "Eat":          2,
    "GetupSitdown": 3,
    "Readbook":     4,
    "Usecomputer":  5,
    "Usephone":     6,
    "Usetablet":    7,
    "Walk":         8,
    "WatchTV":      9
    }

sims_simple_dataset_decoding = {val: key for key, val in sims_simple_dataset_encoding.items()}


class SimsDataset(GenericActionDataset):
    def __init__(self,
                 max_bodies=5,
                 split_train_file=os.path.expanduser("~/datasets/sims_dataset/SimsSplitsCompleteVideos.csv"),
                 split_val_file=os.path.expanduser("~/datasets/sims_dataset/SimsSplitsCompleteVideos.csv"),
                 split_test_file=None,
                 return_data=("vclip", "label"),
                 action_set="sims4action",
                 dataset_name="Sims Dataset",
                 **kwargs) -> None:
        self.action_pat = re.compile(r"^([^_]*)_S(\d*)([^_]*\d*)_(fC\d*|m\d*)")

        self.action_set = action_set
        if action_set is None or self.action_set == "sims4action":
            self.action_dict_encode = sims_simple_dataset_encoding
            self.action_dict_decode = sims_simple_dataset_decoding
        else:
            raise NotImplementedError(f"Missing information to select the action set {action_set}.")

        super().__init__(max_bodies=max_bodies,
                         split_train_file=split_train_file,
                         split_val_file=split_val_file,
                         split_test_file=split_test_file,
                         return_data=return_data,
                         dataset_name=dataset_name,
                         action_dict_encode=self.action_dict_encode,
                         action_dict_decode=self.action_dict_decode,
                         **kwargs)

    def __getitem__(self, index) -> T_co:
        return super().__getitem__(index)

    def extract_info_from_path(self, path):
        vid_id = os.path.split(path)[1]

        act = os.path.split(os.path.split(path)[0])[1]

        fn = os.path.split(path)[1]

        m = self.action_pat.match(fn)

        _, sub, loc, cam = m.group(1), int(m.group(2)), m.group(3), m.group(4)  # act is already defined.

        return vid_id, act, sub, loc, cam  # Second would be subaction if applicable

    def detect_videos(self, dataset_root, cache_folder, use_cache, dataset_name):
        """
        In its generic form, this method expects a file system structure like "dataset_root/<action_names>/<video_ids>".
        It returns a dataframe which extracts this information and also contains the number of frames per video.
        :param dataset_name:
        :param dataset_root: The root folder of the dataset.
        :param cache_folder: If the cache is used, the dataframe is read from a file in this folder, instead.
        :param use_cache: If False, the dataframe is not read from file, but still written to a file.
        :return: A dataframe with columns ["video_id", "action", "frame_count", "base_path"]
        """

        vid_cache_name = f"video_info_cache_{dataset_name.replace(' ', '-')}.csv"
        if use_cache and os.path.exists(os.path.join(cache_folder, vid_cache_name)):
            video_info = pd.read_csv(os.path.join(cache_folder, vid_cache_name), index_col=False)
            # video_info = video_info.set_index("sample_id")
            print("Loaded video info from cache.")
        else:
            print("Searching for video folders on the file system...", end="")
            video_paths = self.index_video_paths(dataset_root)
            print("\b\b\b finished.")

            infos = [self.extract_info_from_path(p) for p in video_paths]

            video_info = pd.DataFrame(infos, columns=["sample_id", "action", "subject", "location", "camera"])

            print("Determining frame count per video...")
            video_info["vid_path"] = [os.path.relpath(p, dataset_root) for p in video_paths]
            video_info["frame_count"] = list(tqdm(map(lambda p: GenericActionDataset.count_frames(p), video_paths)))

            # video_info = video_info.set_index("sample_id", drop=False)

            if cache_folder is not None:
                if not os.path.exists(cache_folder):
                    os.makedirs(cache_folder)
                video_info.to_csv(os.path.join(cache_folder, vid_cache_name))

        return video_info

    def get_split_from_file(self, video_info: pd.DataFrame):
        raise NotImplementedError
        # split_df = pd.read_csv(self.split_train_file)

        # split_df["vid_id"] = split_df["VideoName"].apply(lambda vn: os.path.splitext(vn)[0])

        # video_info = video_info.merge(split_df[["vid_id", "Split"]], on="vid_id", validate="one_to_one")

        # video_info = video_info[video_info["Split"] == self.split_mode]

        # return video_info

    def prepare_split(self, video_info: pd.DataFrame, random_state=42):
        rng = np.random.default_rng(seed=random_state)

        if self.split_policy == "frac":
            train_msk = rng.random(size=len(video_info)) < self.split_train_frac
            not_train_msk = [not b for b in train_msk]

            rest_sample_count = sum(not_train_msk)

            val_msk = rng.random(size=rest_sample_count) < self.split_val_frac / (1 - self.split_train_frac)
            test_msk = [not b for b in val_msk]

            if self.split_mode == "train":
                return video_info[train_msk]
            elif self.split_mode == "val":
                not_train_samples = video_info[not_train_msk]
                return not_train_samples[val_msk]
            elif self.split_mode == "test":
                not_train_samples = video_info[not_train_msk]
                return not_train_samples[test_msk]
            else:
                raise ValueError()
        elif self.split_policy == "file":
            return self.get_split_from_file(video_info)

        elif self.split_policy == "cross-subject":
            train_subjects = [1, 2, 3, 5, 6]
            val_subjects = [7]
            test_subjects = [4, 8]

            if self.split_mode == "train":
                video_info = video_info[video_info["subject"].isin(train_subjects)]
                return video_info
            elif self.split_mode == "val":
                video_info = video_info[video_info["subject"].isin(val_subjects)]
                return video_info
            elif self.split_mode == "test":
                video_info = video_info[video_info["subject"].isin(test_subjects)]
                return video_info
            else:
                raise ValueError
        else:
            raise ValueError()


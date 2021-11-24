import collections
import glob
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils import data
from torch.utils.data.dataset import T_co
from tqdm import tqdm

from lib.datasets.dataset import DatasetUtils
from lib.datasets.dataset_kinetics import KineticsDatasetUtils


class GenericActionDataset(data.Dataset):

    def __init__(self,
                 frame_root=None,
                 skele_motion_root=None,
                 max_bodies=5,
                 split_mode='train',
                 vid_transforms=None,
                 time_shifts=None,
                 seq_len=32,
                 downsample_vid=1,
                 sample_limit=None,
                 sample_limit_val=None,
                 split_policy="frac",
                 split_val_frac=0.1,
                 split_test_frac=0.1,
                 split_train_file=None,
                 split_val_file=None,
                 split_test_file=None,
                 chunk_length=None,
                 chunk_shift=None,
                 return_data=("vclip", "label"),
                 skele_motion_sharded=False,
                 use_cache=True,
                 cache_folder="cache",
                 frame_name_template="image_{:05}.jpg",
                 random_state=42,
                 per_class_frac=None,
                 test_multi_sampling=10,
                 dataset_name="Generic Action Dataset",
                 action_dict_encode=None,
                 action_dict_decode=None,
                 action_dict_check=True,
                 ) -> None:
        super().__init__()
        self.sample_limit_val = sample_limit_val
        self.frame_root = frame_root
        self.skele_motion_root = skele_motion_root
        self.max_bodies = max_bodies
        self.split_mode = split_mode
        self.vid_transforms = vid_transforms
        self.seq_len = seq_len
        self.time_shifts = time_shifts if time_shifts is not None else [0]  # [0 for _ in vid_transforms] TODO
        self.downsample_vid = downsample_vid
        self.sample_limit = sample_limit
        self.chunk_length = chunk_length
        self.chunk_shift = self.chunk_length if chunk_shift is None else chunk_shift
        self.per_class_frac = per_class_frac
        self.skele_motion_sharded = skele_motion_sharded
        self.test_multisampling = test_multi_sampling
        self.action_dict_check = action_dict_check

        max_time_shift = max(shift for shift in self.time_shifts if shift is not None)
        min_time_shift = min(shift for shift in self.time_shifts if shift is not None)

        self.time_span = 0 if self.time_shifts is None else max_time_shift - min_time_shift

        self.min_length = self.seq_len + self.time_span
        # self.min_length = max(self.min_length, chunk_length) if chunk_length is not None else self.min_length

        self.split_policy = split_policy

        if split_policy == "frac":
            self.split_val_frac = split_val_frac if split_val_frac is not None else 0.
            self.split_test_frac = split_test_frac if split_test_frac is not None else 0.
            self.split_train_frac = 1.0 - self.split_val_frac - self.split_test_frac
        elif split_policy == "file":
            self.split_train_file = split_train_file
            self.split_val_file = split_val_file
            self.split_test_file = split_test_file

        self.return_data = return_data

        self.use_cache = use_cache
        self.cache_folder = cache_folder

        self.frame_name_template = frame_name_template

        self.random_state = random_state

        self.dataset_name = dataset_name

        gad = GenericActionDataset
        dsu = DatasetUtils

        print("=================================")
        print(f'{self.dataset_name} ({self.split_mode} set). Dataset root: \n{self.frame_root}')
        print(f'Split policy: Split {self.split_policy}')

        self.sample_info = self.detect_videos(self.frame_root, self.cache_folder, self.use_cache,
                                              self.dataset_name)
        self.sample_info = self.sample_info.set_index("sample_id", drop=False)

        print(f'Total number of video samples in this dataset: {len(self.sample_info)}')

        print(f'Frames per sequence: {self.seq_len}\n'
              f'Downsampling on video frames: {self.downsample_vid}')

        # get action list
        actions = sorted(list(set(self.sample_info["action"]))) if "action" in self.sample_info.columns else []

        if action_dict_encode is not None and self.action_dict_check:
            self.action_dict_encode = action_dict_encode
            dataset_action_set = set(self.action_dict_encode.keys())
            if set(actions) != dataset_action_set and len(actions) > 0:
                print("WARNING: An action encoding was manually specified, it does not cover the dataset completely.")
                print("Following dataset actions can not be encoded:")
                print(set(actions).difference(dataset_action_set))

                old_len = len(self.sample_info)
                self.sample_info = self.sample_info[self.sample_info["action"].isin(dataset_action_set)]
                print(f"Lost {old_len - len(self.sample_info)} of {old_len} samples due to missing action encodings.")
        else:
            self.action_dict_encode = {a: i for i, a in enumerate(actions)}
        if action_dict_decode is not None:
            self.action_dict_decode = action_dict_decode
        else:
            self.action_dict_decode = {i: a for a, i in self.action_dict_encode.items()}

        # filter out too short videos:
        # drop_idx = []
        # for idx, row in tqdm(self.sample_info.iterrows(), total=len(self.sample_info)):
        #    vlen = row.frame_count
        #    if vlen < self.min_length:
        #        drop_idx.append(idx)

        sample_count = len(self.sample_info)

        self.sample_info = dsu.filter_too_short(self.sample_info, self.min_length)

        print(f"Dropped {sample_count - len(self.sample_info)} samples due to insufficient length "
              f"(less than {self.min_length} frames).\n"
              f"Remaining dataset size: {len(self.sample_info)}")

        # self.sample_info = self.sample_info.drop(drop_idx, axis=0)

        self.sample_info = self.prepare_split(self.sample_info, random_state=random_state)

        print(f"Number of samples in split {self.split_mode}: {len(self.sample_info)}")

        if "skmotion" in self.return_data:
            assert self.skele_motion_root is not None, "Undefined skele motion root folder."

            sample_count = len(self.sample_info)

            sk_cache_name = "sk_info_cache_{}.csv".format(self.dataset_name.replace(" ", "-"))

            if use_cache and os.path.exists(os.path.join(cache_folder, sk_cache_name)):
                self.sk_info = pd.read_csv(os.path.join(cache_folder, sk_cache_name), index_col=False)
                self.sk_info = self.sk_info.set_index(["sample_id", "body"], verify_integrity=True, drop=False)

                self.sk_info["caetano_magnitude_path"] = self.sk_info["caetano_magnitude_path"].apply(
                    lambda p: os.path.join(self.skele_motion_root, p))
                self.sk_info["caetano_orientation_path"] = self.sk_info["caetano_orientation_path"].apply(
                    lambda p: os.path.join(self.skele_motion_root, p))

                print("Loaded skeleton info from cache.")
            else:
                print("Loading and preparing skeleton info. This might take time.")
                start_time = time.perf_counter()
                self.sk_info = dsu.get_skeleton_info(skele_motion_root)
                stop_time = time.perf_counter()

                print(f"Computed skeleton info ({stop_time - start_time} s)")

                if cache_folder is not None:
                    if not os.path.exists(cache_folder):
                        os.makedirs(cache_folder)
                    self.sk_info["caetano_magnitude_path"] = self.sk_info["caetano_magnitude_path"].apply(
                        lambda p: os.path.split(p)[1])
                    self.sk_info["caetano_orientation_path"] = self.sk_info["caetano_orientation_path"].apply(
                        lambda p: os.path.split(p)[1])
                    self.sk_info.to_csv(os.path.join(cache_folder, sk_cache_name))

            sk_count = len(self.sk_info)

            print("Filtering skeleton info...", end="")
            self.sk_info = dsu.filter_too_short(self.sk_info,
                                                self.min_length)  # TODO: Only makes sense for fixed size videos.

            print("\rFiltered skeleton info and dropped {} of {} skeleton samples due to insufficient sequence length "
                  "({} frames needed).".format(sk_count - len(self.sk_info), sk_count, self.min_length))

            print("Found {} skeleton sequences of sufficient length.".format(len(self.sk_info)))

            start_time = time.perf_counter()
            self.sample_info: pd.DataFrame = gad.filter_by_missing_skeleton_info(self.sample_info, self.sk_info)
            drop_count = sample_count - len(self.sample_info)
            stop_time = time.perf_counter()
            print(
                "Dropped {} of {} samples due to missing skeleton information. ({} s)".format(drop_count,
                                                                                              sample_count,
                                                                                              stop_time - start_time))

            self.sample_info = self.sample_info.merge(self.sk_info["frame_count"].groupby(level=[0]).min(),
                                                      left_index=True, right_index=True, suffixes=("", "_skinfo"))
            self.sample_info["frame_count"] = self.sample_info[["frame_count", "frame_count_skinfo"]].min(axis=1)

        if self.sample_limit is not None and self.split_mode != "val":
            print(f"Limiting to {self.sample_limit} samples.")
            self.sample_info = self.sample_info.sample(min(self.sample_limit, len(self.sample_info)),
                                                       random_state=random_state)
            print(f"Remaining dataset size: {len(self.sample_info)}")
        elif self.split_mode == "val" and self.sample_limit_val is not None and self.sample_limit_val < len(
                self.sample_info):
            print(f"Limiting validation set to {self.sample_limit_val} samples.")
            self.sample_info = self.sample_info.sample(min(self.sample_limit_val, len(self.sample_info)),
                                                       random_state=random_state)
            print(f"Remaining dataset size: {len(self.sample_info)}")

        print(f"Sample ids unique:{self.sample_info['sample_id'].is_unique}")
        self.sample_info = self.sample_info.drop_duplicates(subset="sample_id")  # Todo: better printouts.
        print(f"Sample ids unique:{self.sample_info['sample_id'].is_unique}")

        if self.chunk_length:
            print(f"Interpreting videos as collections of chunk samples of length {self.chunk_length}.")
            self.sample_info, self.chunk_to_vid_idxs = self.prepare_chunks(self.sample_info, self.chunk_length,
                                                                           self.chunk_shift)
            if self.split_mode != "test":
                print(f"This increased the number of samples to {len(self.chunk_to_vid_idxs)}.")
            else:
                print("In mode test, only the middle chunk is used for evaluation.")

        if self.per_class_frac is not None:
            sample_count = len(self.sample_info)
            expected_count = int(self.per_class_frac * sample_count)
            class_sample_list = []
            for action in actions:
                action_samples = self.sample_info[self.sample_info["action"] == action]
                if len(action_samples) > 0:
                    if len(action_samples) < expected_count:
                        print(f"Sampling class {action} multiple times because it does not contain "
                              f"enough samples ({len(action_samples)}/{expected_count}))")
                        class_sample_list.append(action_samples.sample(expected_count, replace=True))
                    else:
                        print(f"Sub-sampling class {action} because it contains too much"
                              f"samples ({len(action_samples)}/{expected_count}))")
                        class_sample_list.append(action_samples.sample(expected_count, replace=False))

            self.sample_info = pd.concat(class_sample_list, ignore_index=True)

        if self.split_mode == "test": print(f"Test multi-sampling is {self.test_multisampling}")
        print("=================================")

    def __getitem__(self, index) -> T_co:
        ret_dict = {}

        gad = GenericActionDataset
        if self.chunk_length is None:
            # operating on videos
            sample = self.sample_info.iloc[index]
            start_frame = 0
            end_frame = sample["frame_count"]
        else:
            if self.split_mode == "test":  # Only using middle chunk for testing.
                sample = self.sample_info.iloc[index]
                chunk = sample["chunk_count"] // 2
            else:
                vid_idx = self.chunk_to_vid_idxs[index]
                sample = self.sample_info.loc[vid_idx]
                chunk = index - sample["chunk_start_idx"]
            start_frame = self.chunk_shift * chunk
            end_frame = self.chunk_shift * chunk + self.chunk_length
            ret_dict["chunk"] = chunk

        ret_dict["sample_id"] = sample["sample_id"]

        if isinstance(self.vid_transforms, collections.abc.Sequence):
            vid_transforms = self.vid_transforms
        else:
            vid_transforms = (self.vid_transforms,)

        view_count = len(vid_transforms)

        if "label" in self.return_data:
            try:
                ret_dict["label"] = torch.tensor(self.action_dict_encode[sample["action"]])
            except TypeError as te:
                print(te)
                print(sample["action"])
                print(self.action_dict_encode)

        v_len = sample["frame_count"]

        if not self.split_mode == "test" and not self.split_mode == "feature_eval":
            frame_indices = []
            for i in range(view_count):
                frame_indices.extend(gad.idx_sampler(vlen=v_len,
                                                     seq_len=self.seq_len,
                                                     vpath=sample["vid_path"],
                                                     multi_time_shifts=self.time_shifts,
                                                     frame_range=(start_frame, end_frame)))
        else:
            frame_indices = []
            view_count = self.test_multisampling
            vid_transforms = vid_transforms * self.test_multisampling

            for i in range(self.test_multisampling):
                frame_indices.extend(gad.idx_sampler(vlen=v_len,
                                                     seq_len=self.seq_len,
                                                     vpath=sample["vid_path"],
                                                     multi_time_shifts=self.time_shifts,
                                                     frame_range=(start_frame, end_frame)))

        start_frames = [idxs[0] for idxs in frame_indices]
        ret_dict["start_frame"] = start_frames

        if "vclip" in self.return_data:
            frame_indices_vid = [idxs[::self.downsample_vid] for idxs in frame_indices]

            # If sequences are overlapping, we save IO time by only loading images once. Often, IO is the bottleneck.
            file_paths = [
                [os.path.join(self.frame_root, sample["vid_path"], self.frame_name_template.format(i + 1)) for i in
                 idxs] for idxs in frame_indices_vid]

            file_path_set = set([fp for subl in file_paths for fp in subl])
            img_dict = {fp: gad.pil_loader(fp) for fp in file_path_set}

            seq_vids = [[img_dict[fp].copy() for fp in subl] for subl in file_paths]

            del img_dict

            # At this point we only make use of a single clip per sample. Can be changed in the future.
            # seq_vid = seq_vids[0]

            t_seqs = [torch.stack(vt(seq), 0) for vt, seq in zip(vid_transforms, seq_vids)]

            # del seq_vid, seq_vids

            # (self.seq_len, C, H, W) -> (C, self.seq_len, H, W)
            t_seqs = torch.stack(t_seqs, 0).transpose(1, 2)

            ret_dict["vclips"] = t_seqs

        if "skmotion" in self.return_data:
            kdu = KineticsDatasetUtils
            sk_seqs, skeleton_frame_count = kdu.load_skeleton_seqs(self.sk_info, sample["sample_id"],
                                                                   self.skele_motion_root, frame_indices,
                                                                   max_bodies=self.max_bodies,
                                                                   load_sharded=self.skele_motion_sharded)

            # sk_seqs = kdu.select_skeleton_seqs(sk_seqs, frame_indices)

            sk_seqs = torch.tensor(sk_seqs, dtype=torch.float)

            body_count = sk_seqs.shape[1]

            # Bodies for which movement magnitude is zero in all clips:
            body_mag_view = sk_seqs[:, :, :, :, 3:].reshape(view_count, body_count, -1)
            zero_seq = torch.all(body_mag_view == 0, dim=2)
            zero_seq = torch.all(zero_seq,
                                 dim=0)  # Policy on missing bodies within multiple views: Only remove if all views are missing?
            if torch.any(zero_seq):
                if not torch.all(zero_seq):
                    old_shape = sk_seqs.shape
                    sk_seqs = sk_seqs[:, torch.logical_not(zero_seq)].reshape(view_count, -1, *old_shape[2:])
                else:
                    sk_seqs = sk_seqs[:, :1]

            # The skeleton image connsists of joint values over time. H = Joints, W = Time steps (num_seq * seq_len).
            # (sk_J, sk_T, sk_C) = sk_seqs[0].shape

            sk_seqs = sk_seqs.transpose(2, 3)  # (cl, Bo, sk_T, sk_J, sk_C)
            sk_seqs = sk_seqs.transpose(3, 4).transpose(2, 3)  # (cl, Bo, C, T, J)

            body_count = sk_seqs.shape[1]

            if body_count >= self.max_bodies:
                sk_seqs = sk_seqs[:, :self.max_bodies]
                body_count = sk_seqs.shape[1]
            else:
                # This part is due to the weird design choice of the torch padding function.
                # Just filling up missing bodies with zero.
                pad = [0, 0] * 3
                pad.extend([0, self.max_bodies - body_count])
                sk_seqs = F.pad(sk_seqs, pad)

            ret_dict["skmotion"] = sk_seqs
            ret_dict["body_count"] = body_count

        assert len(ret_dict.keys()) > 0

        return ret_dict

    def __len__(self):
        samples = len(self.sample_info)
        chunks = None
        if hasattr(self, "chunk_to_vid_idxs"):
            chunks = len(self.chunk_to_vid_idxs)

        return samples if not hasattr(self, "chunk_to_vid_idxs") or self.split_mode == "test" else chunks

    def get_num_actions(self):
        return len(self.action_dict_decode)

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
            elif self.split_mode == "all":
                return video_info
            else:
                raise ValueError()
        elif self.split_policy == "file":
            return self.get_split_from_file(video_info)
        elif self.split_policy == "all":
            return video_info
        else:
            raise ValueError()

    def class_count(self):
        return len(self.action_dict_decode)

    def get_split_from_file(self, video_info):
        raise NotImplementedError

    def index_video_paths(self, dataset_root):
        return glob.glob(os.path.join(dataset_root, "*/*"))

    def extract_info_from_path(self, path):
        return os.path.split(os.path.split(path)[0])[1], None  # Second would be subaction if applicable

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

            video_ids = [os.path.split(p)[1] for p in video_paths]
            actions = [self.extract_info_from_path(p) for p in video_paths]
            actions = [a[0] if a[1] is None else ".".join(a) for a in actions]

            print("Determining frame count per video...")
            vinfo = {"sample_id":   video_ids,
                     "vid_path":    [os.path.relpath(p, dataset_root) for p in video_paths],
                     "action":      actions,
                     "frame_count": list(tqdm(map(lambda p: GenericActionDataset.count_frames(p), video_paths)))}

            video_info = pd.DataFrame(vinfo)
            # video_info = video_info.set_index("sample_id", drop=False)

            if cache_folder is not None:
                if not os.path.exists(cache_folder):
                    os.makedirs(cache_folder)
                video_info.to_csv(os.path.join(cache_folder, vid_cache_name))

        return video_info

    @staticmethod
    def count_frames(path, ending=".jpg"):
        return len(glob.glob(os.path.join(path, "*" + ending)))

    @staticmethod
    def idx_sampler(vlen, seq_len, vpath, sample_discretization=None, start_frame=None, frame_range=None,
                    multi_time_shifts=None):
        # cases:
        # - sampling with start frame
        # - sampling randomly along discretely chosen blocks
        # - sampling with random start

        # Special case handling: if multi time shifts is not None, but one of its entries, it means that the entry
        # should be chosen randomly within a possible range.

        # Copy is necessary, otherwise random value is replaced for None only at the first time and then propagated.
        shift_span = 0

        if multi_time_shifts is not None:
            multi_time_shifts = multi_time_shifts.copy()

            for idx, shift in enumerate(multi_time_shifts):
                if shift is not None:
                    continue
                else:  # Replace with random.
                    min_shift = min(shift for shift in multi_time_shifts if shift is not None)
                    max_shift = max(shift for shift in multi_time_shifts if shift is not None)

                    span = max_shift - min_shift

                    remainer = vlen - (span + seq_len)

                    possible_shifts = list(range(min_shift - remainer + 1, max_shift + remainer - 1))
                    if len(possible_shifts) > 4 * seq_len:
                        for i in range(-seq_len, seq_len):
                            possible_shifts.remove(i)

                    possible_shifts = possible_shifts if len(possible_shifts) > 0 else [
                        0]  # First and last are the same.

                    multi_time_shifts[idx] = np.random.choice(possible_shifts)

                multi_time_shifts = multi_time_shifts + np.abs(min(multi_time_shifts))

            # - Either a single sequence or multiple shifted sequences.

            shift_span = int(0 if multi_time_shifts is None else max(multi_time_shifts) - min(multi_time_shifts))

            # Make sure video filtering worked correctly.
            if vlen - (seq_len + shift_span) < 0:
                print(f"Tried to sample a video which is too short. \nVideo path: {vpath}")
                return [None]

        first_possible_start = int(0)

        last_possible_start = int(vlen - (seq_len + shift_span))

        assert first_possible_start <= last_possible_start

        if frame_range is not None:
            first_possible_start = frame_range[0]
            last_possible_start = min(frame_range[1] - seq_len, last_possible_start)

        # Sampling with a pre-chosen start-frame:
        if start_frame is not None:
            if first_possible_start <= start_frame <= last_possible_start:
                start_idx = start_frame
            else:
                print(f"Not all frames were available at position {start_frame}, for limited vlen {vlen} of {vpath}."
                      f" Sampling in the middle.")
                start_idx = first_possible_start + (last_possible_start - first_possible_start) // 2

        # Sampling discrete blocks.
        elif sample_discretization is not None:
            starts = range(first_possible_start, last_possible_start, sample_discretization)
            starts = starts if len(starts) > 0 else [first_possible_start]  # First and last are the same.

            start_idx = np.random.choice(starts)

        # Base case: sample a random start.
        else:
            starts = list(range(first_possible_start, last_possible_start))
            starts = starts if len(starts) > 0 else [first_possible_start]  # First and last are the same.

            start_idx = np.random.choice(starts)

        # Here we have one start index and we know that it is possible to sample all provided time shifts (if any)
        if multi_time_shifts is not None:
            seq_idxs = [start_idx + np.arange(seq_len) + time_shift for time_shift in multi_time_shifts]
        else:
            seq_idxs = [start_idx + np.arange(seq_len)]

        return seq_idxs

    @staticmethod
    def pil_loader(path):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                img.load()
                return img.convert('RGB')

    def prepare_chunks(self, video_info: pd.DataFrame, chunk_length, chunk_shift):
        # video_info["chunk_count"] = (video_info["frame_count"] // chunk_length).apply(lambda v: max(v, 1))
        # video_info["chunk_count"] = ((video_info["chunk_count"] - 1) * chunk_length // chunk_shift).apply(
        #    lambda v: int(max(v, 1)))  # TODO: Check

        if chunk_shift <= chunk_length:
            # Take care since last chunk might go out of bounds.
            video_info["chunk_count"] = (
                np.ceil(((video_info["frame_count"] - chunk_length) / chunk_shift).round(10))).astype(int)

            exact_match_chunk = (((video_info["frame_count"] - chunk_length) % chunk_shift) == 0).astype(int)
            video_info["chunk_count"] += exact_match_chunk
        else:
            # If another shift would be possible, so would be another chunk. If not, it could still be.
            video_info["chunk_count"] = video_info["frame_count"] // chunk_shift
            video_info["chunk_count"] += (video_info["frame_count"] % chunk_shift) >= chunk_length

        remainder_frames = video_info["frame_count"] - ((video_info["chunk_count"] - 1) * chunk_shift + chunk_length)
        assert ((0 <= remainder_frames) & (remainder_frames <= chunk_length)).all()
        video_info["chunk_start_idx"] = 0
        video_info["chunk_start_idx"] = [0] + list(video_info["chunk_count"].cumsum().iloc[:-1])

        chunk_to_vid_idx = []

        for idx, row in video_info.iterrows():
            chunk_to_vid_idx.extend([idx] * row.chunk_count)

        return video_info, chunk_to_vid_idx

    @staticmethod
    def get_skeleton_info(skele_motion_root, worker_count=None,
                          re_id_bod_type=r"(.*)_(\d*)_([^.]*)\.[^.]*\.[^.]*$") -> pd.DataFrame:
        gdu = DatasetUtils

        return gdu.get_skeleton_info(skele_motion_root, worker_count, re_id_bod_type)

    @staticmethod
    def get_skeleton_length(path):
        try:
            sk_seq = np.load(path)

            sk_seq = sk_seq['arr_0']

            (J, T, C) = sk_seq.shape

            return T
        except IOError as ioe:
            print(path + " I/O error: " + str(ioe))
        except EOFError as eof:
            print(path + " EOF Error:" + str(eof))
        except:
            print(path + " Unexpected error.")

        return 0

    @staticmethod
    def df_get_skeleton_length(df: pd.DataFrame):
        du = DatasetUtils

        frame_counts = []

        for row in tqdm(df.itertuples(), total=len(df)):
            sk_path = row.sk_path
            length = du.get_skeleton_length(sk_path)
            frame_counts.append(length)

        df["frame_count"] = frame_counts

        return df

    @staticmethod
    def filter_by_missing_skeleton_info(sample_info: pd.DataFrame, skeleton_info: pd.DataFrame):
        sk_ids = skeleton_info.index.get_level_values("sample_id")
        sk_ids = set(sk_ids)

        vid_ids = list(sample_info.index.get_level_values("sample_id"))
        ids = [vid_id for vid_id in vid_ids if vid_id in sk_ids]
        return sample_info.loc[ids]

    @staticmethod
    def load_skeleton_seqs(sk_info: pd.DataFrame, sample_id, skele_motion_root, max_bodies=None) -> (np.ndarray, int):
        """
        Loads a skele-motion representation and selects the columns which are indexed by idx_block.
        Returns a tensor of shape (Joints, Length, Channels).
        The length describes the number of time steps (frame count when downsampling is 1).
        First 3 channels are orientation, last channel is magnitude.
        """
        sk_body_infos = sk_info.xs(sample_id, level="id")

        sk_seqs_mag = []
        sk_seqs_ori = []

        for body_id in list(sk_body_infos.index.values)[:(1 if max_bodies is None else max_bodies)]:
            sk_seq_mag_path = os.path.join(skele_motion_root, sk_body_infos.loc[body_id]["caetano_magnitude_path"])
            sk_seq_ori_path = os.path.join(skele_motion_root, sk_body_infos.loc[body_id]["caetano_orientation_path"])

            try:
                sk_mag = np.load(sk_seq_mag_path)
                sk_ori = np.load(sk_seq_ori_path)

                sk_mag = sk_mag['arr_0']
                sk_ori = sk_ori['arr_0']
            except EOFError as e:
                print(f"Path Mag: {sk_seq_mag_path} \nPath Ori: {sk_seq_ori_path}")
                raise e

            try:
                (J_m, T_m, C_m) = sk_mag.shape
                (J_o, T_o, C_o) = sk_ori.shape
            except:
                print(f"Could not read one of the skeleton files: \n "
                      f"{os.path.join(skele_motion_root, sk_seq_mag_path)}\n"
                      f"{os.path.join(skele_motion_root, sk_seq_ori_path)}")
                continue

            assert J_m == J_o and T_m == T_o

            sk_seqs_mag.append(sk_mag)
            sk_seqs_ori.append(sk_ori)

        sk_seqs = [np.concatenate((sk_s_ori, sk_s_mag), axis=-1) for sk_s_ori, sk_s_mag in
                   zip(sk_seqs_ori, sk_seqs_mag)]  # Concatenating on channel dimension.

        sk_seqs = np.stack(sk_seqs, axis=0)  # (Bo, J, T, C)
        is_zero_seqs = [np.all(sk_s_mag == 0) for sk_s_mag in sk_seqs_mag]

        if not all(is_zero_seqs):
            sk_seqs = sk_seqs[np.logical_not(is_zero_seqs)]
        else:
            sk_seqs = sk_seqs[:1]

        # sk_seqs_mag = np.stack(sk_seqs_mag)
        # sk_seqs_ori = np.stack(sk_seqs_ori)

        T = sk_seqs.shape[2]

        sk_seqs = DatasetUtils.check_sk_seq_nan_inf(sk_seqs)

        return sk_seqs, T

    @staticmethod
    def check_sk_seq_nan_inf(sk_seq):
        if np.isnan(sk_seq).any() or np.isinf(sk_seq).any():
            print("Skeleton sequence for contained nan or inf. Converting to 0.")
        sk_seq = np.nan_to_num(sk_seq)

        return sk_seq

    @staticmethod
    def select_skeleton_seqs(sk_seq, frame_indices):
        (Bo, J, T, C) = sk_seq.shape

        mask = [False] * T
        for i in frame_indices:
            mask[i] = True

        sk_seq = sk_seq[:, :, mask, :]

        (Bo, J, T, C) = sk_seq.shape

        assert T == len(frame_indices)

        return sk_seq

    @staticmethod
    def subsample_discretely(sample_info: pd.DataFrame, sample_discretion: int, seq_len: int):
        subs_sample_info = {col: [] for col in sample_info.columns}
        subs_sample_info["start_frame"] = []

        for idx, row in tqdm(sample_info.iterrows(), total=len(sample_info)):
            sub_count = (row["frame_count"] - seq_len) // sample_discretion

            if row["frame_count"] > 305:
                print(f'Frame count is unusually high: {row["frame_count"]}, frame count {row["frame_count"]} '
                      f'for path {row["path"]}. Skipping video.')

                continue

            for i in range(sub_count):
                row["start_frame"] = i * sample_discretion

                for key, val in row.to_dict().items():
                    subs_sample_info[key].append(val)

        subs_sample_info = pd.DataFrame.from_dict(subs_sample_info)
        subs_sample_info = subs_sample_info.set_index(["id", "start_frame"], drop=False)

        return subs_sample_info

    @staticmethod
    def _extract_skeleton_info_type(file):
        mag_match = DatasetUtils.sk_magnitude_pattern.match(file)
        if mag_match:
            return mag_match.group(1)
        else:
            ori_match = DatasetUtils.sk_orientation_pattern.match(file)

            if ori_match:
                return ori_match.group(1)
            else:
                return np.NaN

    @staticmethod
    def read_video_info(video_info, extract_infos=True, max_samples=None) -> pd.DataFrame:
        raise NotImplementedError

    @staticmethod
    def extract_infos(sample_infos: pd.DataFrame):
        raise NotImplementedError

    @staticmethod
    def encode_action(action_name, zero_indexed=True):
        '''give action name, return category'''
        raise NotImplementedError

    @staticmethod
    def decode_action(action_code, zero_indexed=True):
        '''give action code, return action name'''
        raise NotImplementedError

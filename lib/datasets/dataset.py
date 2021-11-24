"""
------------
dataset_3d
------------

What?
--------
A module which provides the basic logic for working with the datasets in this project.
In the end, everything here is used in PyTorch dataloaders.

How?
--------
This module contains utility classes with logic which can be generalized for the datasets.

Some code has been used from Tengda Han: https://github.com/TengdaHan/DPC
"""

import glob
import multiprocessing as mp
import os
import re

import pandas as pd
from tqdm import tqdm

from lib.utils.augmentation import *


class DatasetUtils:
    sk_magnitude_pattern = re.compile(r".*(CaetanoMagnitude).*")
    sk_orientation_pattern = re.compile(r".*(CaetanoOrientation).*")

    @staticmethod
    def filter_too_short(video_info: pd.DataFrame, min_frame_count: int) -> pd.DataFrame:
        # Filtering videos based on min length.
        video_info = video_info.loc[video_info["frame_count"] >= min_frame_count]

        return video_info

    @staticmethod
    def filter_too_short_skeletons(skeleton_info: pd.DataFrame, video_info: pd.DataFrame) -> pd.DataFrame:
        # Filtering videos based on min length.
        skeleton_info = skeleton_info.merge(video_info[["sample_id", "frame_count"]], on="sample_id",
                                            suffixes=("", "_vinfo"))
        skeleton_info = skeleton_info[skeleton_info["frame_count"] + 10 > skeleton_info["frame_count_vinfo"]]
        return video_info

    @staticmethod
    def idx_sampler(vlen, seq_len, vpath, sample_discretization=None, start_frame=None, multi_time_shifts=None):
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

            # - Either a single sequence or multiple shifted sequences.

            shift_span = int(0 if multi_time_shifts is None else max(multi_time_shifts) - min(multi_time_shifts))

            # Make sure video filtering worked correctly.
            if vlen - (seq_len + shift_span) < 0:
                print(f"Tried to sample a video which is too short. \nVideo path: {vpath}")
                return [None]

        # Find the boundaries which are not out of range.
        time_shifts_positive = multi_time_shifts is None or min(multi_time_shifts) > 0

        first_possible_start = int(0 if time_shifts_positive else 0 - min(multi_time_shifts))

        last_possible_start = int(vlen - (seq_len + shift_span))

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

    @staticmethod
    def load_img_buffer(sample, i):
        with open(os.path.join(sample["path"], 'image_%05d.jpg' % (i + 1)), 'rb') as f:
            return np.frombuffer(f.read(), dtype=np.uint8)

    @staticmethod
    def get_skeleton_info(skele_motion_root, worker_count=None,
                          re_id_bod_type=r"(.*)_(\d*)_([^.]*)\.[^.]*\.[^.]*$") -> pd.DataFrame:
        gdu = DatasetUtils

        skeleton_paths = glob.glob(os.path.join(skele_motion_root, "*.npz"))
        skeleton_paths = list(filter(lambda p: "shard" not in os.path.split(p)[1], skeleton_paths))
        sk_info = pd.DataFrame(skeleton_paths, columns=["sk_path"])

        sk_info["sk_file"] = sk_info["sk_path"].apply(lambda p: os.path.split(p)[1])

        re_pat = re.compile(re_id_bod_type)
        sk_info["sample_id"] = sk_info["sk_file"].apply(lambda fl: re_pat.match(fl).group(1))
        sk_info = sk_info.astype(dtype={"sample_id": 'string'})

        sk_info["body"] = sk_info["sk_file"].apply(lambda fl: int(re_pat.match(fl).group(2)))

        sk_info["skeleton_info_type"] = sk_info["sk_file"].apply(lambda fl: re_pat.match(fl).group(3))

        if worker_count != 0:
            procs = mp.cpu_count() if worker_count is None else worker_count
            print("Using multiprocessing with {} processes.".format(procs))
            df_split = np.array_split(sk_info, procs)
            pool = mp.Pool(procs)

            sk_info = pd.concat(pool.map(gdu.df_get_skeleton_length, df_split))

            pool.close()
            pool.join()
        else:
            sk_info["frame_count"] = sk_info["sk_path"].apply(gdu.get_skeleton_length)

        sk_info = sk_info[sk_info["frame_count"] != 0]

        sk_info = sk_info.drop(columns=["sk_file"])

        sk_info_magnitude = sk_info.loc[sk_info["skeleton_info_type"] == "CaetanoMagnitude"]
        sk_info_magnitude = sk_info_magnitude.rename(columns={"sk_path": "caetano_magnitude_path"})
        sk_info_magnitude = sk_info_magnitude.drop(columns=["skeleton_info_type"])
        sk_info_magnitude = sk_info_magnitude.set_index(["sample_id", "body"], verify_integrity=True, drop=False)

        sk_info_orientation = sk_info.loc[sk_info["skeleton_info_type"] == "CaetanoOrientation"]
        sk_info_orientation = sk_info_orientation.rename(columns={"sk_path": "caetano_orientation_path"})
        sk_info_orientation = sk_info_orientation.drop(columns=["skeleton_info_type"])
        sk_info_orientation = sk_info_orientation.set_index(["sample_id", "body"], verify_integrity=True, drop=False)

        sk_info = sk_info_magnitude.join(sk_info_orientation, rsuffix="_right")

        # Apparently pandas can not join on index if index columns are not dropped (Column overlap not ignored).
        sk_info = sk_info.drop(columns=["sample_id_right", "body_right"])

        count = len(sk_info)
        sk_info = sk_info.dropna()

        if count > len(sk_info):
            print("Dropped {} of {} skeleton samples due to missing information.".format(count - len(sk_info), count))

        return sk_info

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
        except OSError as ose:
            print(path + "OS Error:" + str(ose))
        except BaseException as e:
            print(path + " Unexpected error.")
            print(e)


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
        sk_ids = skeleton_info.index.get_level_values("id")
        sk_ids = set(sk_ids)

        vid_ids = list(sample_info.index.get_level_values("id"))
        ids = [vid_id for vid_id in vid_ids if vid_id in sk_ids]
        return sample_info.loc[ids]

    @staticmethod
    def _load_skeleton_seqs_files(mag_path, ori_path, skele_motion_root) -> (np.ndarray, int):
        sk_mag, sk_ori = None, None
        (J_m, T_m, C_m) = None, None, None
        (J_o, T_o, C_o) = None, None, None

        try:
            sk_mag = np.load(mag_path)
            sk_ori = np.load(ori_path)

            sk_mag = sk_mag['arr_0']
            sk_ori = sk_ori['arr_0']
        except BaseException as e:
            print(f"Path Mag: {mag_path} \nPath Ori: {ori_path}")
            raise e

        try:
            (J_m, T_m, C_m) = sk_mag.shape
            (J_o, T_o, C_o) = sk_ori.shape
        except:
            print(f"Could not read one of the skeleton files: \n "
                  f"{os.path.join(skele_motion_root, mag_path)}\n"
                  f"{os.path.join(skele_motion_root, ori_path)}")

        assert J_m == J_o and T_m == T_o

        return sk_mag, sk_ori

    @staticmethod
    def load_skeleton_seqs(sk_info: pd.DataFrame, sample_id, sk_motion_root, frame_indices=None, max_bodies=None,
                           load_sharded=False, shard_length=300) -> (np.ndarray, int):
        """
        Loads a skele-motion representation and selects the columns which are indexed by idx_block.
        Returns a tensor of shape (Joints, Length, Channels).
        The length describes the number of time steps (frame count when downsampling is 1).
        First 3 channels are orientation, last channel is magnitude.
        """
        sk_body_infos = sk_info.xs(sample_id, level="sample_id")

        sk_seqs_mag = []
        sk_seqs_ori = []

        if load_sharded:
            assert frame_indices is not None

            min_idx, max_idx = np.min(frame_indices), np.max(frame_indices)
            min_shard_idx = min_idx // shard_length
            max_shard_idx = max_idx // shard_length

        for body_id in list(sk_body_infos.index.values)[:(len(sk_body_infos) if max_bodies is None else max_bodies)]:
            mag_path = os.path.join(sk_motion_root, sk_body_infos.loc[body_id]["caetano_magnitude_path"])
            ori_path = os.path.join(sk_motion_root, sk_body_infos.loc[body_id]["caetano_orientation_path"])

            sk_mag, sk_ori = None, None

            if frame_indices is None or not load_sharded:  # Load complete file.
                sk_mag, sk_ori = DatasetUtils._load_skeleton_seqs_files(mag_path, ori_path, sk_motion_root)

            else:  # Frame indices available and sharded loading
                mag_shards = []
                ori_shards = []

                for i in range(min_shard_idx, max_shard_idx + 1):
                    mag_path_shard = os.path.splitext(mag_path)[0] + f".shard{i:05d}.npz"
                    ori_path_shard = os.path.splitext(ori_path)[0] + f".shard{i:05d}.npz"

                    sk_mag_shard, sk_ori_shard = DatasetUtils._load_skeleton_seqs_files(mag_path_shard, ori_path_shard,
                                                                                        sk_motion_root)
                    mag_shards.append(sk_mag_shard), ori_shards.append(sk_ori_shard)

                if any(n is None for n in mag_shards + ori_shards):
                    print(f"Skipping body since not all shards were available ({min_shard_idx}-{max_shard_idx}): "
                          f"\n{mag_path}\n{ori_path}")
                    continue

                sk_mag = np.concatenate(mag_shards, axis=1)  # (J, T, C) -> Concat on time
                sk_ori = np.concatenate(ori_shards, axis=1)

            if not (sk_mag is None or sk_ori is None):
                sk_seqs_mag.append(sk_mag)
                sk_seqs_ori.append(sk_ori)
            else:
                print("Found sk seq which is None.")

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

        if frame_indices is not None:
            (BoA, JA, TA, CA) = sk_seqs.shape

            if load_sharded:
                frame_indices = frame_indices - min_shard_idx * shard_length  # subtract offset of shards.

            ret_sk_seqs = []

            for index_clip in frame_indices:
                mask = [False] * TA
                for i in index_clip:
                    mask[i] = True

                ret_sk_seq = sk_seqs[:, :, mask, :]

                (Bo, J, T, C) = ret_sk_seq.shape

                assert T == len(index_clip)

                ret_sk_seqs.append(ret_sk_seq)

            return np.stack(ret_sk_seqs, axis=0), T

        return sk_seqs, T

        #######

    @staticmethod
    def check_sk_seq_nan_inf(sk_seq):
        if np.isnan(sk_seq).any() or np.isinf(sk_seq).any():
            print("Skeleton sequence for contained nan or inf. Converting to 0.")
        sk_seq = np.nan_to_num(sk_seq)

        return sk_seq

    @staticmethod
    def select_skeleton_seqs(sk_seq, frame_indices):
        (BoA, JA, TA, CA) = sk_seq.shape

        ret_sk_seqs = []

        for index_clip in frame_indices:
            mask = [False] * TA
            for i in index_clip:
                mask[i] = True

            ret_sk_seq = sk_seq[:, :, mask, :]

            (Bo, J, T, C) = ret_sk_seq.shape

            assert T == len(index_clip)

            ret_sk_seqs.append(ret_sk_seq)

        return np.stack(ret_sk_seqs, axis=0)

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

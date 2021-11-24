import glob
import multiprocessing as mp
import os
import re
import time

import numpy as np
import pandas as pd
import torch
from torch.utils import data

from lib.datasets.dataset import DatasetUtils

hmdb_action_decoding = {
    1:  "cartwheel",
    2:  "ride_horse",
    3:  "dribble",
    4:  "handstand",
    5:  "situp",
    6:  "run",
    7:  "draw_sword",
    8:  "drink",
    9:  "sword_exercise",
    10: "sword",
    11: "shoot_gun",
    12: "dive",
    13: "pullup",
    14: "fencing",
    15: "pick",
    16: "flic_flac",
    17: "sit",
    18: "catch",
    19: "walk",
    20: "hit",
    21: "laugh",
    22: "golf",
    23: "pour",
    24: "hug",
    25: "wave",
    26: "shoot_ball",
    27: "throw",
    28: "ride_bike",
    29: "shake_hands",
    30: "jump",
    31: "chew",
    32: "punch",
    33: "swing_baseball",
    34: "fall_floor",
    35: "kick",
    36: "climb",
    37: "smoke",
    38: "turn",
    39: "kiss",
    40: "clap",
    41: "push",
    42: "smile",
    43: "somersault",
    44: "kick_ball",
    45: "climb_stairs",
    46: "talk",
    47: "pushup",
    48: "eat",
    49: "brush_hair",
    50: "shoot_bow",
    51: "stand"
    }

hmdb_action_encoding = {val: key for key, val in hmdb_action_decoding.items()}


class HMDB51Dataset(data.Dataset):
    def __init__(self,
                 mode='train',
                 transform=None,
                 seq_len=30,
                 downsample_vid=1,
                 epsilon=5,
                 which_split=1,
                 max_samples=None,
                 random_state=42,
                 test_shift=None,
                 discrete_subsamples=10,
                 return_rel_path=False,
                 return_sk=True,
                 frame_location=os.path.expanduser('~/datasets/hmdb51/frames'),
                 split_location=os.path.expanduser('~/datasets/hmdb51/split/frame_splits'),
                 skele_motion_root=os.path.expanduser("~/datasets/hmdb51/hmdb51-skele-motion"),
                 max_bodies=3,
                 class_mapping_file=os.path.expanduser('~/datasets/hmdb51/classInd.txt'),
                 use_cache=True,
                 cache_folder="cache"
                 ):
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.min_length = max(self.seq_len, test_shift if test_shift is not None else 0)
        self.downsample_vid = downsample_vid
        self.epsilon = epsilon
        self.which_split = which_split
        self.frame_location = os.path.expanduser(frame_location)

        self.test_shift = test_shift
        self.test_subsamples = discrete_subsamples

        self.return_rel_path = return_rel_path

        self.return_sk = return_sk
        self.max_bodies = max_bodies

        print("=================================")
        print(f'Dataset HMDB51 split {which_split}: {mode} set.')
        # splits
        if mode == 'train' or mode == "train_nn":
            split = os.path.join(split_location, f"train_split{self.which_split:02d}.csv")
            video_info = pd.read_csv(split, header=None)
        elif (mode == 'val') or (mode == 'test') or mode == "test_nn":
            # TODO: Separate Test split?
            split = os.path.join(split_location, f"test_split{self.which_split:02d}.csv")
            video_info = pd.read_csv(split, header=None)
        else:
            raise ValueError('wrong mode')

        video_info[0] = video_info[0].apply(lambda s: os.path.join(self.frame_location, s))

        print(f'Total number of video samples: {len(video_info)}')

        print(f'Frames per sequence: {seq_len}\n'
              f'Downsampling on video frames: {downsample_vid}')

        # get action list
        self.action_dict_encode = {}
        self.action_dict_decode = {}

        action_df = pd.read_csv(class_mapping_file, sep=' ', header=None)
        for _, row in action_df.iterrows():
            act_id, act_name = row
            act_id = int(act_id) - 1  # let id start from 0
            self.action_dict_decode[act_id] = act_name
            self.action_dict_encode[act_name] = act_id

        # filter out too short videos:
        drop_idx = []
        for idx, row in video_info.iterrows():
            vpath, vlen = row
            if vlen - self.min_length <= 0:
                drop_idx.append(idx)

        print(f"Dropped {len(drop_idx)} samples due to insufficient length (less than {seq_len} frames).\n"
              f"Remaining dataset size: {len(video_info) - len(drop_idx)}")

        self.sample_info = video_info.drop(drop_idx, axis=0)

        self.sample_info = self.sample_info.rename({0: "file_name", 1: "frames"}, axis=1)
        self.sample_info["id"] = self.sample_info["file_name"].apply(lambda fl: os.path.basename(os.path.normpath(fl)))
        self.sample_info = self.sample_info.set_index(["id"], drop=False)
        ###
        hdu = HMDBDatasetUtils

        if self.return_sk:
            sample_count = len(self.sample_info)

            sk_cache_name = "sk_info_hmdb_cache_mfc-{}.csv".format(seq_len)

            if use_cache and os.path.exists(os.path.join(cache_folder, sk_cache_name)):
                self.sk_info = pd.read_csv(os.path.join(cache_folder, sk_cache_name), index_col=False)
                self.sk_info = self.sk_info.set_index(["id", "body"], verify_integrity=True, drop=False)
                print("Loaded skeleton info from cache.")
            else:
                print("Loading and preparing skeleton info. This might take time.")
                start_time = time.perf_counter()
                self.sk_info = hdu.get_skeleton_info(skele_motion_root)
                sk_count = len(self.sk_info)
                self.sk_info = hdu.filter_too_short(self.sk_info, self.seq_len)  # In theory there should be 300 frames.
                stop_time = time.perf_counter()
                print("Loaded skeleton info and dropped {} of {} skeleton samples due to insufficient sequence length "
                      "({} frames needed) ({} s).".format(
                    sk_count - len(self.sk_info), sk_count, self.seq_len, stop_time - start_time))

                if cache_folder is not None:
                    if not os.path.exists(cache_folder):
                        os.makedirs(cache_folder)
                    self.sk_info.to_csv(os.path.join(cache_folder, sk_cache_name))

            print("Found {} skeleton sequences of sufficient length.".format(len(self.sk_info)))

            start_time = time.perf_counter()
            self.sample_info = hdu.filter_by_missing_skeleton_info(self.sample_info, self.sk_info)
            drop_count = sample_count - len(self.sample_info)
            stop_time = time.perf_counter()
            print(
                "Dropped {} of {} samples due to missing skeleton information. ({} s)".format(drop_count, sample_count,
                                                                                              stop_time - start_time))

        ###

        if max_samples is not None:
            self.sample_info = self.sample_info.sample(max_samples, random_state=random_state)

        if mode == 'val':
            self.sample_info = self.sample_info.sample(frac=0.3)  # TODO: This makes no sense with splits.
        # shuffle not required

        if mode == "test" or mode == "test_nn" or mode == "train_nn":
            if self.test_shift is None:
                print(f'In mode test, {self.test_subsamples} uniformly chosen sequence samples of length {seq_len} '
                      f'are chosen from each video sample for evaluation instead of a single one.')
            else:
                print(f'In mode test, all samples with a shift of {self.test_shift} frames '
                      f'are used instead of a single one.')

        print("=================================")

    def class_count(self):
        return len(self.action_dict_encode)

    def __getitem__(self, index):
        hdu = HMDBDatasetUtils
        sample = self.sample_info.iloc[index]
        vpath, vlen = sample.file_name, sample.frames

        sk_seqs = None
        body_counts = None
        frame_idxs_ls = None

        if self.return_sk:
            sk_seqs_all, skeleton_frame_count = hdu.load_skeleton_seqs(self.sk_info, sample["id"],
                                                                       max_bodies=self.max_bodies)

            # This is because it is not certain to have all sk data.
            vlen = min(vlen, skeleton_frame_count)

        if self.mode == "train" or self.mode == "val":
            frame_idxs = hdu.idx_sampler(vlen, self.seq_len, vpath)[0]
            frame_idxs_vid = frame_idxs[::self.downsample_vid]
            frame_idxs_ls = [frame_idxs_vid]
        elif self.mode == "test" or self.mode == "train_nn" or self.mode == "test_nn":
            # Choose 10 samples.
            frame_idxs_ls = []

            if self.test_shift is None:
                for i in range(self.test_subsamples):  # Randomized uniformly chosen.
                    frame_idxs_ls.append(hdu.idx_sampler(vlen, self.seq_len, vpath)[0])
            else:
                vlen = (vlen // self.test_shift) * self.test_shift

                for i in range(vlen // self.test_shift):  # Sampling with shift.
                    frame_idxs_ls.append(
                        hdu.idx_sampler(vlen, self.seq_len, vpath, start_frame=i * self.test_shift).flatten())

            for i in range(len(frame_idxs_ls)):
                frame_idxs_ls[i] = frame_idxs_ls[i][::self.downsample_vid]

            frame_idxs_vid = np.concatenate(frame_idxs_ls, axis=None)
        else:
            raise ValueError

        seq = [DatasetUtils.pil_loader(os.path.join(vpath, 'image_%05d.jpg' % (i + 1))) for i in frame_idxs_vid]

        if self.mode == "test" or self.mode == "train_nn" or self.mode == "test_nn":
            fr_per_seq = len(seq) // self.test_subsamples
            seqs = [seq[i * fr_per_seq: (i + 1) * fr_per_seq] for i in range(self.test_subsamples)]
            seqs = [self.transform(s) for s in seqs]
            t_seq = [im for im_list in seqs for im in im_list]
        else:
            t_seq = self.transform(seq)  # apply same transform

        (C, H, W) = t_seq[0].size()
        t_seq = torch.stack(t_seq, 0)

        # if self.mode == "train" or self.mode == "val":
        #    t_seq = t_seq.view(len(frame_idxs_vid), C, H, W).transpose(0, 1)
        # else:
        t_seq = t_seq.reshape(-1, self.seq_len, C, H, W).transpose(1, 2)

        if self.return_sk:
            sk_seqs = [hdu.select_skeleton_seqs(sk_seqs_all, fr_list) for fr_list in frame_idxs_ls]

            sk_seqs = [torch.tensor(sk_seq, dtype=torch.float) for sk_seq in sk_seqs]

            body_counts = []

            for i, sk_seq in enumerate(sk_seqs):
                # Filter out bodys which do not appear in this sequence:
                zero_seq = torch.all(sk_seq[:, :, :, 3:] == 0, dim=3)  # Checking if mag is zero.
                zero_seq = zero_seq.view(sk_seq.shape[0], -1)
                zero_seq = zero_seq.all(dim=1)

                if torch.any(zero_seq):
                    if not torch.all(zero_seq):
                        sk_seq = sk_seq[~zero_seq]
                    else:
                        sk_seq = sk_seq[:1]

                # The skeleton image connsists of joint values over time. H = Joints, W = Time steps (num_seq * seq_len).
                # (sk_J, sk_T, sk_C) = sk_seqs[0].shape

                sk_seq = sk_seq.transpose(1, 2)  # (Bo, sk_T, sk_J, sk_C)
                sk_seq = sk_seq.transpose(2, 3).transpose(1, 2)  # (Bo, C, T, J)

                body_counts.append(sk_seq.shape[0])

                # This is necessary because default collate can not handle varying tensor sizes.
                ret_sk_seq_sh = list(sk_seq.shape)
                ret_sk_seq_sh[0] = self.max_bodies

                ret_sk_seq = torch.zeros(ret_sk_seq_sh)
                ret_sk_seq[:len(sk_seq)] = sk_seq[:]

                sk_seqs[i] = ret_sk_seq

            if self.mode == "test" or self.mode == "train_nn" or self.mode == "test_nn":
                sk_seqs = torch.stack(sk_seqs)
            else:
                assert len(sk_seqs) == 1
                sk_seqs = sk_seqs[0]

        try:
            vname = os.path.normpath(vpath).split('/')[-2]
            vid = self.encode_action(vname)
            label = torch.tensor(vid, dtype=torch.long)
        except:
            print(f"Unexpected, could not extract action name from folder: {vpath}")
            label = None

        return_dict = {"vclips": t_seq, "label": label, "sample_id": sample["id"]}

        if sk_seqs is not None:
            return_dict["skmotion"] = sk_seqs
            return_dict["body_count"] = torch.tensor(body_counts, dtype=torch.long)

        if self.return_rel_path:
            return_dict["rel_path"] = os.path.join(*vpath.split('/')[-3:])

        return return_dict

    def __len__(self):
        return len(self.sample_info)

    def encode_action(self, action_name):
        '''give action name, return category'''
        return self.action_dict_encode[action_name]

    def decode_action(self, action_code):
        '''give action code, return action name'''
        return self.action_dict_decode[action_code]


class HMDBDatasetUtils(DatasetUtils):
    info_reo = re.compile("^(.+)_([0-9]+)_(CaetanoMagnitude|CaetanoOrientation).json.npz$")

    @staticmethod
    def get_skeleton_info(skele_motion_root, worker_count=None) -> pd.DataFrame:
        hdu = HMDBDatasetUtils

        skeleton_paths = glob.glob(os.path.join(skele_motion_root, "*.npz"))
        sk_info = pd.DataFrame(skeleton_paths, columns=["sk_path"])

        sk_info["sk_file"] = sk_info["sk_path"].apply(lambda p: os.path.split(p)[1])
        info_reos = sk_info["sk_file"].apply(lambda fl: hdu.info_reo.match(fl))
        sk_info["id"] = info_reos.apply(lambda ir: ir.group(1))
        sk_info = sk_info.astype(dtype={"id": 'string'})

        sk_info["body"] = info_reos.apply(lambda ir: ir.group(2))

        sk_info["skeleton_info_type"] = info_reos.apply(lambda ir: ir.group(3))

        if worker_count != 0:
            procs = mp.cpu_count() if worker_count is None else worker_count
            print("Using multiprocessing with {} processes.".format(procs))
            df_split = np.array_split(sk_info, procs)
            pool = mp.Pool(procs)

            sk_info = pd.concat(pool.map(hdu.df_get_skeleton_length, df_split))

            pool.close()
            pool.join()
        else:
            sk_info["frame_count"] = sk_info["sk_path"].apply(hdu.get_skeleton_length)

        sk_info = sk_info[sk_info["frame_count"] != 0]

        sk_info = sk_info.drop(columns=["sk_file"])

        sk_info_magnitude = sk_info.loc[sk_info["skeleton_info_type"] == "CaetanoMagnitude"]
        sk_info_magnitude = sk_info_magnitude.rename(columns={"sk_path": "caetano_magnitude_path"})
        sk_info_magnitude = sk_info_magnitude.drop(columns=["skeleton_info_type"])
        sk_info_magnitude = sk_info_magnitude.set_index(["id", "body"], verify_integrity=True, drop=False)

        sk_info_orientation = sk_info.loc[sk_info["skeleton_info_type"] == "CaetanoOrientation"]
        sk_info_orientation = sk_info_orientation.rename(columns={"sk_path": "caetano_orientation_path"})
        sk_info_orientation = sk_info_orientation.drop(columns=["skeleton_info_type"])
        sk_info_orientation = sk_info_orientation.set_index(["id", "body"], verify_integrity=True, drop=False)

        sk_info = sk_info_magnitude.join(sk_info_orientation, rsuffix="_right")

        # Apparently pandas can not join on index if index columns are not dropped (Column overlap not ignored).
        sk_info = sk_info.drop(columns=["id_right", "body_right"])

        count = len(sk_info)
        sk_info = sk_info.dropna()

        if count > len(sk_info):
            print("Dropped {} of {} skeleton samples due to missing information.".format(count - len(sk_info), count))

        return sk_info

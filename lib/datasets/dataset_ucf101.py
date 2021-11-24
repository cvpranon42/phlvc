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

ucf_action_encoding = {'ApplyEyeMakeup':    1, 'ApplyLipstick': 2, 'Archery': 3, 'BabyCrawling': 4, 'BalanceBeam': 5,
                   'BandMarching':      6, 'BaseballPitch': 7, 'Basketball': 8, 'BasketballDunk': 9, 'BenchPress': 10,
                   'Biking':            11, 'Billiards': 12, 'BlowDryHair': 13, 'BlowingCandles': 14,
                   'BodyWeightSquats':  15, 'Bowling': 16, 'BoxingPunchingBag': 17, 'BoxingSpeedBag': 18,
                   'BreastStroke':      19, 'BrushingTeeth': 20, 'CleanAndJerk': 21, 'CliffDiving': 22,
                   'CricketBowling':    23, 'CricketShot': 24, 'CuttingInKitchen': 25, 'Diving': 26, 'Drumming': 27,
                   'Fencing':           28, 'FieldHockeyPenalty': 29, 'FloorGymnastics': 30, 'FrisbeeCatch': 31,
                   'FrontCrawl':        32, 'GolfSwing': 33, 'Haircut': 34, 'Hammering': 35, 'HammerThrow': 36,
                   'HandstandPushups':  37, 'HandstandWalking': 38, 'HeadMassage': 39, 'HighJump': 40, 'HorseRace': 41,
                   'HorseRiding':       42, 'HulaHoop': 43, 'IceDancing': 44, 'JavelinThrow': 45, 'JugglingBalls': 46,
                   'JumpingJack':       47, 'JumpRope': 48, 'Kayaking': 49, 'Knitting': 50, 'LongJump': 51,
                   'Lunges':            52, 'MilitaryParade': 53, 'Mixing': 54, 'MoppingFloor': 55, 'Nunchucks': 56,
                   'ParallelBars':      57, 'PizzaTossing': 58, 'PlayingCello': 59, 'PlayingDaf': 60, 'PlayingDhol': 61,
                   'PlayingFlute':      62, 'PlayingGuitar': 63, 'PlayingPiano': 64, 'PlayingSitar': 65,
                   'PlayingTabla':      66, 'PlayingViolin': 67, 'PoleVault': 68, 'PommelHorse': 69, 'PullUps': 70,
                   'Punch':             71, 'PushUps': 72, 'Rafting': 73, 'RockClimbingIndoor': 74, 'RopeClimbing': 75,
                   'Rowing':            76, 'SalsaSpin': 77, 'ShavingBeard': 78, 'Shotput': 79, 'SkateBoarding': 80,
                   'Skiing':            81, 'Skijet': 82, 'SkyDiving': 83, 'SoccerJuggling': 84, 'SoccerPenalty': 85,
                   'StillRings':        86, 'SumoWrestling': 87, 'Surfing': 88, 'Swing': 89, 'TableTennisShot': 90,
                   'TaiChi':            91, 'TennisSwing': 92, 'ThrowDiscus': 93, 'TrampolineJumping': 94, 'Typing': 95,
                   'UnevenBars':        96, 'VolleyballSpiking': 97, 'WalkingWithDog': 98, 'WallPushups': 99,
                   'WritingOnBoard':    100, 'YoYo': 101}


class UCF101Dataset(data.Dataset):
    def __init__(self,
                 mode='train',
                 transform=None,
                 seq_len=30,
                 test_shift=None,
                 downsample_vid=1,
                 epsilon=5,
                 which_split=1,
                 max_samples=None,
                 discrete_subsamples=10,
                 return_rel_path=False,
                 return_sk=True,
                 frame_location=os.path.expanduser('~/datasets/ucf101/frames'),
                 split_location=os.path.expanduser('~/datasets/ucf101/split/frame_splits'),
                 skele_motion_root=os.path.expanduser("~/datasets/ucf101/ucf-skele-motion"),
                 max_bodies=3,
                 class_mapping_file=os.path.expanduser('~/datasets/ucf101/classInd.txt'),
                 use_cache=True,
                 cache_folder="cache",
                 random_state=42):
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.downsample_vid = downsample_vid
        self.epsilon = epsilon
        self.which_split = which_split

        self.test_shift = test_shift
        self.test_subsamples = discrete_subsamples
        self.return_rel_path = return_rel_path
        self.return_sk = return_sk

        self.max_bodies = max_bodies

        print("=================================")
        print(f'Dataset UCF101 split {which_split}: {mode} set.')
        # splits
        if mode == 'train' or mode == "train_nn":
            split = os.path.join(split_location, f"train_split{self.which_split:02d}.csv")
            video_info = pd.read_csv(split, header=None)
        elif (mode == 'val') or (mode == 'test') or mode == "test_nn":
            split = os.path.join(split_location, f"test_split{self.which_split:02d}.csv")
            video_info = pd.read_csv(split, header=None)
        else:
            raise ValueError('wrong mode')

        video_info[0] = video_info[0].apply(lambda s: os.path.join(frame_location, s))

        print(f'Total number of video samples: {len(video_info)}')

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
            if vlen - self.seq_len <= 0:
                drop_idx.append(idx)

        print(f"Dropped {len(drop_idx)} samples due to insufficient length (less than {seq_len} frames).\n"
              f"Remaining dataset size: {len(video_info) - len(drop_idx)}")

        self.sample_info = video_info.drop(drop_idx, axis=0)

        self.sample_info = self.sample_info.rename({0: "file_name", 1: "frames"}, axis=1)
        self.sample_info["id"] = self.sample_info["file_name"].apply(lambda fl: os.path.basename(os.path.normpath(fl)))
        self.sample_info = self.sample_info.set_index(["id"], drop=False)

        udu = UCFDatasetUtils

        if self.return_sk:
            sample_count = len(self.sample_info)

            sk_cache_name = "sk_info_ucf_cache_mfc-{}.csv".format(seq_len)

            if use_cache and os.path.exists(os.path.join(cache_folder, sk_cache_name)):
                self.sk_info = pd.read_csv(os.path.join(cache_folder, sk_cache_name), index_col=False)
                self.sk_info = self.sk_info.set_index(["id", "body"], verify_integrity=True, drop=False)
                print("Loaded skeleton info from cache.")
            else:
                print("Loading and preparing skeleton info. This might take time.")
                start_time = time.perf_counter()
                self.sk_info = udu.get_skeleton_info(skele_motion_root)
                sk_count = len(self.sk_info)
                self.sk_info = udu.filter_too_short(self.sk_info, self.seq_len)  # In theory there should be 300 frames.
                stop_time = time.perf_counter()
                print("Loaded skeleton info and dropped {} of {} skeleton samples due to insufficient sequence length "
                      "({} frames needed) ({} s).".format(sk_count - len(self.sk_info), sk_count, self.seq_len,
                                                          stop_time - start_time))

                if cache_folder is not None:
                    if not os.path.exists(cache_folder):
                        os.makedirs(cache_folder)
                    self.sk_info.to_csv(os.path.join(cache_folder, sk_cache_name))

            print("Found {} skeleton sequences of sufficient length.".format(len(self.sk_info)))
            start_time = time.perf_counter()
            self.sample_info = udu.filter_by_missing_skeleton_info(self.sample_info, self.sk_info)
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

    def __getitem__(self, index):
        udu = UCFDatasetUtils
        sample = self.sample_info.iloc[index]
        vpath, vlen = sample.file_name, sample.frames

        sk_seqs = None
        sk_seqs_all = None
        body_counts = None

        if self.return_sk:
            sk_seqs_all, skeleton_frame_count = udu.load_skeleton_seqs(self.sk_info, sample["id"],
                                                                       max_bodies=self.max_bodies)

            # This is because it is not certain to have all sk data.
            vlen = min(vlen, skeleton_frame_count)

        if self.mode == "train" or self.mode == "val":
            frame_idxs = udu.idx_sampler(vlen, self.seq_len, vpath)[0]
            frame_idxs_vid = frame_idxs[::self.downsample_vid]
            frame_idxs_ls = [frame_idxs_vid]
        elif self.mode == "test" or self.mode == "train_nn" or self.mode == "test_nn":
            # Choose 10 samples.
            frame_idxs_ls = []

            if self.test_shift is None:
                for i in range(self.test_subsamples):  # Randomized uniformly chosen.
                    frame_idxs_ls.append(udu.idx_sampler(vlen, self.seq_len, vpath)[0])
            else:
                vlen = (vlen // self.test_shift) * self.test_shift

                for i in range(vlen // self.test_shift):  # Sampling with shift.
                    frame_idxs_ls.append(
                        udu.idx_sampler(vlen, self.seq_len, vpath, start_frame=i * self.test_shift).flatten())

            for i in range(len(frame_idxs_ls)):
                frame_idxs_ls[i] = frame_idxs_ls[i][::self.downsample_vid]

            frame_idxs_vid = np.concatenate(frame_idxs_ls, axis=None)
        else:
            raise ValueError

        seq = [udu.pil_loader(os.path.join(vpath, 'image_%05d.jpg' % (i + 1))) for i in frame_idxs_vid]

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
        #    t_seq = t_seq.reshape(len(frame_idxs_ls), -1, C, H, W).transpose(1, 2)

        t_seq = t_seq.reshape(-1, self.seq_len, C, H, W).transpose(1, 2)

        if self.return_sk:
            sk_seqs = [udu.select_skeleton_seqs(sk_seqs_all, fr_list) for fr_list in frame_idxs_ls]

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

                # The skeleton image connsists of joint values over time.
                # H = Joints, W = Time steps (num_seq * seq_len).
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
            return_dict["sk_seqs"] = sk_seqs
            return_dict["body_count"] = torch.tensor(body_counts, dtype=torch.long)

        if self.return_rel_path:
            return_dict["rel_path"] = os.path.join(*vpath.split('/')[-3:])

        return return_dict

    def class_count(self):
        return len(self.action_dict_encode)

    def __len__(self):
        return len(self.sample_info)

    def encode_action(self, action_name):
        '''give action name, return category'''
        return self.action_dict_encode[action_name]

    def decode_action(self, action_code):
        '''give action code, return action name'''
        return self.action_dict_decode[action_code]


class UCFDatasetUtils(DatasetUtils):
    info_reo = re.compile("^(.+)_([0-9]+)_(CaetanoMagnitude|CaetanoOrientation).json.npz$")

    @staticmethod
    def get_skeleton_info(skele_motion_root, worker_count=None) -> pd.DataFrame:
        hdu = UCFDatasetUtils

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

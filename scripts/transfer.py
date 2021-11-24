import argparse
import os
import re
from collections import OrderedDict

import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils import data

import lib.utils.augmentation as augutil
from lib.datasets.dataset_adl import ADLDataset
from lib.datasets.dataset_hmdb51 import HMDB51Dataset
from lib.datasets.dataset_kinetics_gad import Kinetics400Dataset
from lib.datasets.dataset_sims import SimsDataset
from lib.datasets.dataset_ucf101 import UCF101Dataset
from lib.datasets.dataset_ucfhmdb_full import UCFHMDBFull
from lib.models.model_3d_lc import *
from lib.transfer import train_classifier, test_classifier as tcl, train_with_teacher
from lib.transfer.eval_nearest_neighbour import nearest_neighbour_retrieval
from scripts.scripts_util import prepare_aug_settings, set_path, prepare_cuda, get_summary_writers
from scripts.scripts_util import write_settings_file, multi_step_lr_scheduler, prepare_transforms

parser = argparse.ArgumentParser()
parser.add_argument('--exp_num', default=None, type=int,
                    help='Experiment number. This is used in the folder name: exp-{exp-num}. '
                         'Filled automatically if not set.')

parser.add_argument('--exp_suffix', default=None, type=str,
                    help='Experiment suffix for annotations. '
                         'If set, this is also used in the folder name: exp-{exp-num}_{exp-suffix}.')

parser.add_argument('--gpu', default=[0], type=int, nargs='+')
parser.add_argument('--num_workers', default=16, type=int)
parser.add_argument('--prefetch_factor', default=2, type=int)

parser.add_argument('--epochs', default=250, type=int, help='number of total epochs to run')
parser.add_argument('--batch_size', default=1, type=int)

parser.add_argument('--dataset', default='ucf101', type=str)
parser.add_argument('--action_set', default=None,
                    choices=["adl", "sims4action", "kinetics", "nturgbd", "ucf101", "hmdb51", "hmdb12", "ucf12"],
                    type=str)
parser.add_argument('--split', default=1, type=int)
parser.add_argument('--num_classes', default=None, type=int)
parser.add_argument('--split_policy', default="frac",
                    choices=["frac", "cross-subject", "cross-view-1", "cross-view-2", "file"],
                    type=str, help="This determines how train, test and val split are generated. "
                                   "Different datasets offer different types of policies.")

parser.add_argument('--per_class_frac', default=None, type=float)
parser.add_argument('--discrete_samples', default=10, type=int,
                    help='Number of discrete samples in mode test or for nearest neighbour.')
parser.add_argument('--max_samples', default=None, type=int, help='Limit for samples.')
parser.add_argument('--test_multisampling', default=10, type=int, help='Limit for samples.')
parser.add_argument('--chunk_length', default=None, type=int)

parser.add_argument('--seq_len', default=32, type=int)
parser.add_argument('--ds_vid', default=1, type=int)
parser.add_argument('--img_dim', default=128, type=int)
parser.add_argument('--max_bodies', default=5, type=int)

parser.add_argument('--model', default="sk-cont-s3d", type=str,
                    choices=["dpc-resnet", "sk-cont-r21d", "sk-cont-resnet", "sk-cont-s3d"])
parser.add_argument('--vid_backbone', default=None, type=str, choices=['r2+1d18', 'resnet18', "r3d_18", "s3d", "None"])
parser.add_argument('--backbone_naming', default='vid_backbone', type=str, choices=['vid_backbone', 'backbone'])
parser.add_argument('--sk_backbone', default=None, type=str, choices=["sk-motion-7", "None"],
                    help='Enable the skeleton stream.')
parser.add_argument('--sep_vid_classifier', default=False, action='store_true')
parser.add_argument('--sep_skm_classifier', default=False, action='store_true')
parser.add_argument('--classify_with_backbone', default=False, action='store_true')
parser.add_argument('--use_multi_layer_classifier', default=False, action='store_true')
parser.add_argument('--class_distribution', default=None, type=str)

parser.add_argument('--transfer_mode', default="fine_tuning", type=str,
                    choices=["fine_tuning", "nearest_neighbour", "teacher"])
parser.add_argument('--teacher_threshold', default=0.0, type=float)
parser.add_argument('--teacher_balance_predictions', default=None, type=float)
parser.add_argument('--teacher_modality', default="fine_tuning", type=str,
                    choices=["common", "skm", "vid"])
parser.add_argument('--sinkhorn_epsilon', default=0.05, type=float)

parser.add_argument('--label_metric_for_teaching', default=False, action='store_true')

parser.add_argument('--dropout', default=0.5, type=float)
parser.add_argument('--representation_size', default=256, type=int)
parser.add_argument('--hidden_width', default=512, type=int)
parser.add_argument('--swav_prototypes', type=int, default=10000, help='Use SWAV training.')
parser.add_argument('--class_tapping', default=-2, type=int,
                    help="How many fully connected layers to go back to attach the classification layer.")
parser.add_argument('--multi_mode', default=None, type=str,
                    choices=["individualhead", "individualhead_combineconcat", "individualhead_combinemerge",
                             "commonhead", "commonhead_combineconcat", "commonhead_combinemerge"])

parser.add_argument('--exp_root', default="experiments", type=str,
                    help="The base folder for result storage.")

parser.add_argument('--optimizer', default="Adam", choices=["Adam", "SGD"], type=str)
parser.add_argument('--scheduler_steps', default=[30, 50, 100, 150], type=int, nargs="+")

parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--fine_tuning', default=0.1, type=float, help='A ratio which determines the learning rate '
                                                                   'for the backbone in relation to lr.'
                                                                   'Backbone will be frozen if 0.')
parser.add_argument('--head_lr_mult', default=None, type=float, help='learning rate multiplier for prototypes')
parser.add_argument('--head_vid_lr_mult', default=None, type=float, help='learning rate multiplier for prototypes')
parser.add_argument('--head_skm_lr_mult', default=None, type=float, help='learning rate multiplier for prototypes')
parser.add_argument('--prot_lr_mult', default=None, type=float, help='learning rate multiplier for prototypes')
parser.add_argument('--skm_lr_mult', default=None, type=float, help='learning rate multiplier for prototypes')
parser.add_argument('--vid_lr_mult', default=None, type=float, help='learning rate multiplier for prototypes')

parser.add_argument('--wd', default=1e-3, type=float, help='weight decay')

parser.add_argument('--print_freq', default=5, type=int)

parser.add_argument('--save_best_val_loss', type=bool, default=False, help='Save model with best Val Loss.')
parser.add_argument('--save_best_val_acc', type=bool, default=True, help='Save model with best Val Accuracy.')
parser.add_argument('--save_best_train_loss', type=bool, default=False, help='Save model with best Train Loss.')
parser.add_argument('--save_best_train_acc', type=bool, default=True, help='Save model with best Train Accuracy.')
parser.add_argument('--save_interval', type=bool, default=None,
                    help='Save model during fixed intervals.')
parser.add_argument('--save_interval_best_val_acc', type=bool, default=None,
                    help='Save best val acc model on fixed intervals.')

parser.add_argument('--pretrain', default=None, type=str)
parser.add_argument('--pretrain_vid', default=None, type=str)
parser.add_argument('--pretrain_sk', default=None, type=str)
parser.add_argument('--pretrain_prot', default=None, type=str)
parser.add_argument('--pretrain_head', default=None, type=str)

parser.add_argument('--resume', default='', type=str)
parser.add_argument('--reset_lr', action='store_true', help='Reset learning rate when resume training?')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')

parser.add_argument('--eval', default=None, type=str)
parser.add_argument('--test', default=None, type=str)
parser.add_argument('--test_modality', default="vid", type=str, choices=["vid", "skm", "common"])
parser.add_argument('--captum', default=False, action='store_true')

parser.add_argument('--weightfile_backbone', default=None, type=str)
parser.add_argument('--skip_size_mismatch', default=False, action='store_true')

parser.add_argument('--hmdb_frames', default=os.path.expanduser('~/datasets/hmdb51/frames'), type=str)
parser.add_argument('--hmdb_splits', default=os.path.expanduser('~/datasets/hmdb51/split/frame_splits'), type=str)
parser.add_argument('--hmdb_skele_motion', default=os.path.expanduser("~/datasets/hmdb51/hmdb51-skele-motion"),
                    type=str)
parser.add_argument('--hmdb_class_mapping', default=os.path.expanduser('~/datasets/hmdb51/classInd.txt'), type=str)

parser.add_argument('--ucf_frames', default=os.path.expanduser('~/datasets/ucf101/frames'), type=str)
parser.add_argument('--ucf_splits', default=os.path.expanduser('~/datasets/ucf101/splits/frame_splits'), type=str)
parser.add_argument('--ucf_skele_motion', default=os.path.expanduser("~/datasets/ucf101/ucf101-skele-motion"), type=str)
parser.add_argument('--ucf_class_mapping', default=os.path.expanduser('~/datasets/ucf101/classInd.txt'), type=str)

parser.add_argument('--sims_frames', default=os.path.expanduser("~/datasets/sims_dataset/frames"), type=str)
parser.add_argument('--sims_skele_motion', default=os.path.expanduser("~/datasets/sims_dataset/skele-motion"), type=str)

parser.add_argument('--adl_frames', default=os.path.expanduser("~/datasets/adl/frames"), type=str)
parser.add_argument('--adl_skele_motion', default=os.path.expanduser("~/datasets/adl/skele-motion"), type=str)

parser.add_argument('--kinetics-frame-root',
                    default=os.path.expanduser("~/datasets/kinetics/kinetics400/frames"), type=str)
parser.add_argument('--kinetics-skele-motion',
                    default=os.path.expanduser("~/datasets/kinetics/kinetics400-skeleton/skele-motion"), type=str)
parser.add_argument('--kinetics-split-train-file',
                    default=os.path.expanduser("~/datasets/kinetics/kinetics400/train.csv"), type=str)
parser.add_argument('--kinetics-split-val-file',
                    default=os.path.expanduser("~/datasets/kinetics/kinetics400/validate.csv"), type=str)
parser.add_argument('--kinetics-split-test-file',
                    default=os.path.expanduser("~/datasets/kinetics/kinetics400/test.csv"), type=str)

parser.add_argument('--no_cache', default=False, action='store_true', help='Recreate dataset caches.')

parser.add_argument('--no_cuda', default=False, action='store_true')

parser.add_argument('--aug_rotation_range', default=[20.], type=float, nargs='+')

parser.add_argument('--aug_hue_range', default=[0.5], type=float, nargs='+')
parser.add_argument('--aug_saturation_range', default=[1], type=float, nargs='+')
parser.add_argument('--aug_value_range', default=[0.8], type=float, nargs='+')
parser.add_argument('--aug_contrast_range', default=[0], type=float, nargs='+')

parser.add_argument('--aug_crop_min_area', default=0.05, type=float)
parser.add_argument('--aug_crop_max_area', default=1., type=float)
parser.add_argument('--aug_force_crop_inside', default=False, action='store_true',
                    help='Make sure, that crops are performed within image boundaries (no black regions).')
parser.add_argument('--aug_horizontal_flip_prob', default=0., type=float)


def check_and_prepare_args(args):
    for k, v in args.__dict__.items():  # This allows to pass None on the command line (for string attributes).
        if v == "None":
            setattr(args, k, None)

    assert not args.transfer_mode == "teacher_sk" or args.multi_mode == "individual"

    # Paths
    args.log_path, args.model_path, args.exp_path = set_path(args, mode="transfer")

    if args.num_classes == None:
        args.num_class = {"hmdb51":   51, "ucf101": 101, "hmdb12": 12, "ucf12": 12, "adl": 31, "sims4action": 10,
                          "kinetics": 400, "nturgbd": 120, "armar-demo": 31}[args.dataset]
    else:
        args.num_class = args.num_classes  # TODO

    print(f"Creating model with {args.num_class} classes.")

    if args.class_distribution:
        if args.class_distribution == "adl-cs":
            args.class_distribution = [0.1170393777345649, 0.21572678658240155, 0.05736509479824988,
                                       0.12408847836655323, 0.06648031113271755, 0.02345649003403014,
                                       0.03366553232863393, 0.003889158969372873, 0.30627126883811373,
                                       0.05201750121536218]

        elif args.class_distribution == "adl-cw2":
            args.class_distribution = [0.0, 0.2689291101055807, 0.077526395173454, 0.13906485671191554,
                                       0.07119155354449472, 0.03227752639517346,
                                       0.03815987933634993, 0.004223227752639517, 0.3686274509803922, 0.0]
        elif args.class_distribution == "uniform":
            args.class_distribution = [1 / args.num_class] * args.num_class
        else:
            raise ValueError()

    return args


def main():
    args = parser.parse_args()

    check_and_prepare_args(args)

    print("Startup parameters:")
    print(args)
    print()
    augmentation_settings = prepare_aug_settings(args)

    cuda_dev, args.gpu = prepare_cuda(args.gpu)  # Filters and selects the GPUs according to input argument.

    model = select_model(args)  # Selects the model according to the selected backbone.

    model = nn.DataParallel(model)

    if not args.no_cuda:
        model = model.to(cuda_dev)

    writer_train, writer_val = get_summary_writers(args.log_path, "transfer")

    if args.transfer_mode not in ["nearest_neighbour", "feature_extraction"]:
        # load data ###
        train_trans, val_trans = prepare_transforms(augmentation_settings, args)
        train_loader = get_data(train_trans, args, 'train')
        val_loader = get_data(val_trans, args, 'val')

        # noinspection PyUnresolvedReferences
        if args.num_class == None:
            args.num_class = train_loader.dataset.class_count()

        criterion = nn.NLLLoss()

        args.start_iteration = 0

        if args.eval:
            eval_classifier(model, criterion, cuda_dev, args)
            sys.exit()
        if args.test:  # No optimizer needed, own model loading logic.
            test_classifier(model, criterion, cuda_dev, args)
            sys.exit()

        optimizer, scheduler = prepare_optimizer(model, args)

        args = load_pretrained_or_resume(model, optimizer, args)

        write_settings_file(args, args.exp_path)

        if args.transfer_mode == "teacher":
            import copy
            model_t = copy.deepcopy(model)
            model_t.eval()
            train_with_teacher.training_loop_teacher_sk(model, model_t, optimizer, scheduler, criterion, train_loader,
                                                        val_loader,
                                                        cuda_dev, writer_train, writer_val, args)
        else:
            train_classifier.training_loop_ft(model, optimizer, scheduler, criterion, train_loader, val_loader,
                                              cuda_dev,
                                              writer_train, writer_val, args)

        args.test = os.path.join(args.model_path, "model_last.pth.tar")
        test_classifier(model, criterion, cuda_dev, args)

        if args.save_best_val_acc:
            args.test = os.path.join(args.model_path, "model_best_val_acc.pth.tar")
            test_classifier(model, criterion, cuda_dev, args)

        if args.save_best_val_loss:
            args.test = os.path.join(args.model_path, "model_best_val_loss.pth.tar")
            test_classifier(model, criterion, cuda_dev, args)

        if args.save_best_train_acc:
            args.test = os.path.join(args.model_path, "model_best_train_acc.pth.tar")
            test_classifier(model, criterion, cuda_dev, args)

        if args.save_best_train_loss:
            args.test = os.path.join(args.model_path, "model_best_train_loss.pth.tar")
            test_classifier(model, criterion, cuda_dev, args)

    elif args.transfer_mode == "nearest_neighbour":
        # get raw features at certain layer
        # single query clip feature from val
        # sliding window clips from train?
        # Buchler: extract 10 frames per video
        # Speednet: 10 clips per video
        # MemDPC: Multiple blocks with 8 frames with sliding window and average

        # 10 clips per video, both train set and test set (as in ft testing)
        args.epoch = 0
        if args.pretrain and os.path.isfile(args.pretrain):
            checkpoint = torch.load(args.pretrain, map_location=torch.device('cpu'))

            model.module.load_weights_state_dict(checkpoint['state_dict'], model=model)
            print("=> loaded pretrained checkpoint '{}' (epoch {})".format(args.pretrain, checkpoint['epoch']))

            if not args.no_cuda:
                model.cuda()
            args.epoch = checkpoint["epoch"]
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrain))

        # No transformations except for scaling and random crop.
        trans = transforms.Compose([
            augutil.RandomSizedCrop(consistent=True, size=130, p=0.6, crop_area=(0.6, 1), force_inside=True),
            augutil.Scale(size=(args.img_dim, args.img_dim)),
            augutil.ToTensor(),
            augutil.Normalize()
            ])

        train_ds_mode, test_ds_mode = "train_nn", "test_nn"  # 10 clips per video
        train_loader, val_loader = get_data(trans, args, train_ds_mode), get_data(trans, args, test_ds_mode)

        nearest_neighbour_retrieval(model, train_loader, val_loader, cuda_dev, args)


def test_classifier(model, criterion, cuda_dev, args):
    if os.path.isfile(args.test):
        print('\n==========Testing Model===========')
        print("Loading testing checkpoint '{}'".format(args.test))
        checkpoint = torch.load(args.test)

        # model.load_state_dict(checkpoint['state_dict'])
        otl = ""  # "module."
        owl = ""  # "module.vid_backbone."
        if isinstance(model, nn.DataParallel):
            model.module.load_weights_state_dict(checkpoint['state_dict'], model=model, own_lstrip=owl,
                                                 other_lstrip=otl)
        else:
            model.load_weights_state_dict(checkpoint['state_dict'], model=model, own_lstrip=owl,
                                          other_lstrip=otl)

        print(f"Successfully loaded testing checkpoint '{args.test}' (epoch {checkpoint['epoch']})")
        checkpoint_epoch = checkpoint['epoch']

        if args.weightfile_backbone is not None:
            print("Loading alternative video backbone '{}'".format(args.weightfile_backbone))
            checkpoint_b = torch.load(args.weightfile_backbone)
            model.module.load_weights_state_dict(checkpoint_b['state_dict'], ignore_layers=[".*final_fc_classifier.*"],
                                                 strip_module=True)

    elif args.test == 'random':
        checkpoint_epoch = 0
        print("=> [Warning] loaded random weights")
    else:
        print(args.test)
        raise ValueError

    transform = transforms.Compose([
        augutil.Scale(size=224),
        augutil.RandomSizedCrop(consistent=True, size=130, p=0.6, crop_area=(0.6, 1),
                                force_inside=True),
        augutil.Scale(size=args.img_dim),
        augutil.ToTensor(),
        augutil.Normalize()
        ])

    if not args.captum:
        test_loader = get_data(transform, args, 'test')
        tcl.test(test_loader, model, criterion, checkpoint_epoch, cuda_dev, args)
    else:
        test_loader = get_data(transform, args, 'test', random_test_sampler=True)
        tcl.test_captum(test_loader, model, criterion, checkpoint_epoch, cuda_dev, args)


def eval_classifier(model, criterion, cuda_dev, args):
    if os.path.isfile(args.eval):
        print('\n==========Evaluating Model (No Targets needed)===========')
        print("Loading testing checkpoint '{}'".format(args.eval))
        checkpoint = torch.load(args.eval)

        # model.load_state_dict(checkpoint['state_dict'])
        otl = ""  # "module."
        owl = ""  # "module.vid_backbone."
        if isinstance(checkpoint, OrderedDict):
            weight_dict = checkpoint
            otl = "module."
            owl = "module.vid_backbone."
            print(f"=> loading pretrained checkpoint '{args.eval}'")
        else:
            weight_dict = checkpoint['state_dict']
        model.module.load_weights_state_dict(weight_dict, model=model, own_lstrip=owl, other_lstrip=otl)

        print(f"Successfully loaded testing checkpoint '{args.eval}'")

        if args.weightfile_backbone is not None:
            print("Loading alternative video backbone '{}'".format(args.weightfile_backbone))
            checkpoint_b = torch.load(args.weightfile_backbone)
            model.module.load_weights_state_dict(checkpoint_b['state_dict'], ignore_layers=[".*final_fc_classifier.*"],
                                                 strip_module=True)
    else:
        raise ValueError

    transform = transforms.Compose([
        augutil.Scale(size=224),
        augutil.RandomSizedCrop(consistent=True, size=130, p=0.6, crop_area=(0.6, 1), force_inside=True),
        augutil.Scale(size=args.img_dim),
        augutil.ToTensor(),
        augutil.Normalize()
        ])
    test_loader = get_data(transform, args, 'test')

    train_classifier.evaluate(test_loader, model, cuda_dev, args)

    print("Finished evaluation.")


def get_data(transform, args, mode='train', random_test_sampler=False):
    print('Loading data for "%s" ...' % mode)
    return_data = ["label"]
    if args.vid_backbone is not None:
        return_data.append("vclip")
    if args.sk_backbone is not None:
        return_data.append("skmotion")

    if args.dataset == 'ucf101':
        dataset = UCF101Dataset(mode=mode,
                                transform=transform,
                                seq_len=args.seq_len,
                                downsample_vid=args.ds_vid,
                                which_split=args.split,
                                max_samples=args.max_samples,
                                frame_location=args.ucf_frames,
                                split_location=args.ucf_splits,
                                class_mapping_file=args.ucf_class_mapping,
                                skele_motion_root=args.ucf_skele_motion,
                                max_bodies=args.max_bodies,
                                return_sk=False,
                                use_cache=not args.no_cache,
                                discrete_subsamples=args.discrete_samples
                                )
    elif args.dataset == 'hmdb51':
        dataset = HMDB51Dataset(mode=mode,
                                transform=transform,
                                seq_len=args.seq_len,
                                downsample_vid=args.ds_vid,
                                which_split=args.split,
                                max_samples=args.max_samples,
                                frame_location=args.hmdb_frames,
                                split_location=args.hmdb_splits,
                                class_mapping_file=args.hmdb_class_mapping,
                                skele_motion_root=args.hmdb_skele_motion,
                                max_bodies=args.max_bodies,
                                return_sk=False,
                                use_cache=not args.no_cache,
                                discrete_subsamples=args.discrete_samples
                                )
    elif args.dataset == 'hmdb12' or args.dataset == 'ucf12':
        is_hmdb = args.dataset == 'hmdb12'
        frame_root = args.hmdb_frames if is_hmdb else args.ucf_frames
        split_root = args.hmdb_splits if is_hmdb else args.ucf_splits
        skele_motion_root = args.hmdb_skele_motion if is_hmdb else args.ucf_skele_motion

        dataset = UCFHMDBFull(split_mode=mode,
                              vid_transforms=transform,
                              frame_root=frame_root,
                              split_policy=args.split_policy,
                              split_train_file=split_root + f"/train_split0{args.split}.csv",
                              split_val_file=None,
                              split_test_file=split_root + f"/test_split0{args.split}.csv",
                              skele_motion_root=skele_motion_root,
                              skele_motion_sharded=False,
                              seq_len=args.seq_len,
                              downsample_vid=args.ds_vid,
                              sample_limit=args.max_samples,
                              test_multi_sampling=args.test_multisampling,
                              max_bodies=args.max_bodies,
                              chunk_length=args.chunk_length,
                              use_cache=not args.no_cache,
                              dataset_name="HMDB12" if is_hmdb else "UCF12"
                              )
    elif args.dataset == 'sims4action':
        dataset = SimsDataset(split_mode=mode,
                              vid_transforms=transform,
                              frame_root=args.sims_frames,
                              split_policy=args.split_policy,
                              skele_motion_root=args.sims_skele_motion,
                              seq_len=args.seq_len,
                              downsample_vid=args.ds_vid,
                              sample_limit=args.max_samples,
                              max_bodies=args.max_bodies,
                              chunk_length=args.chunk_length,
                              use_cache=not args.no_cache,
                              action_set=args.action_set,
                              return_data=return_data,
                              test_multi_sampling=args.test_multisampling
                              )
    elif args.dataset == 'adl':
        dataset = ADLDataset(split_mode=mode,
                             split_policy=args.split_policy,
                             vid_transforms=transform,
                             video_root=args.adl_frames,
                             skele_motion_root=args.adl_skele_motion,
                             seq_len=args.seq_len,
                             downsample_vid=args.ds_vid,
                             sample_limit=args.max_samples,
                             max_bodies=args.max_bodies,
                             chunk_length=args.chunk_length,
                             use_cache=not args.no_cache,
                             action_set=args.action_set,
                             per_class_frac=args.per_class_frac,
                             return_data=return_data,
                             test_multi_sampling=args.test_multisampling
                             )
    elif args.dataset == 'kinetics':
        dataset = Kinetics400Dataset(split_mode=mode,
                                     split_policy=args.split_policy,
                                     vid_transforms=transform,
                                     frame_root=args.kinetics_frame_root,
                                     skele_motion_root=args.kinetics_skele_motion,
                                     seq_len=args.seq_len,
                                     downsample_vid=args.ds_vid,
                                     sample_limit=args.max_samples,
                                     max_bodies=args.max_bodies,
                                     use_cache=not args.no_cache,
                                     action_set=args.action_set,
                                     per_class_frac=args.per_class_frac,
                                     return_data=return_data,
                                     test_multi_sampling=args.test_multisampling
                                     )
    else:
        raise ValueError('dataset not supported')
    my_sampler = data.RandomSampler(dataset)
    if mode == 'train':
        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      sampler=my_sampler,
                                      shuffle=False,
                                      num_workers=args.num_workers,
                                      pin_memory=True,
                                      drop_last=True,
                                      prefetch_factor=args.prefetch_factor)
    elif mode == 'val':
        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      sampler=my_sampler,
                                      shuffle=False,
                                      num_workers=args.num_workers,
                                      pin_memory=True,
                                      drop_last=True,
                                      prefetch_factor=args.prefetch_factor
                                      )
    else:
        if not random_test_sampler:
            my_sampler = data.SequentialSampler(dataset)

        if mode == 'test':
            data_loader = data.DataLoader(dataset,
                                          batch_size=args.batch_size,
                                          sampler=my_sampler,
                                          shuffle=False,
                                          num_workers=args.num_workers,
                                          pin_memory=True,
                                          drop_last=False)
        elif mode == 'train_nn':
            data_loader = data.DataLoader(dataset,
                                          batch_size=args.batch_size,
                                          sampler=my_sampler,
                                          shuffle=False,
                                          num_workers=args.num_workers,
                                          pin_memory=True,
                                          drop_last=False)
        elif mode == 'test_nn':
            data_loader = data.DataLoader(dataset,
                                          batch_size=args.batch_size,
                                          sampler=my_sampler,
                                          shuffle=False,
                                          num_workers=args.num_workers,
                                          pin_memory=True,
                                          drop_last=False)
        else:
            raise NotImplementedError

    print('"%s" dataset size: %d' % (mode, len(dataset)))
    return data_loader


def prepare_resume(model, optimizer, args):
    if os.path.isfile(args.resume):
        print("=> loading resumed checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
        args.start_epoch = checkpoint['epoch']
        args.start_iteration = checkpoint['iteration']

        args.best_train_loss = checkpoint['best_train_loss']
        args.best_train_acc = checkpoint['best_train_acc']
        args.best_val_loss = checkpoint['best_val_loss']
        args.best_val_acc = checkpoint['best_val_acc']

        model.load_state_dict(checkpoint['state_dict'])
        if not args.reset_lr:  # if didn't reset lr, load old optimizer
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print('==== Forcing lr  to %f ====' % (args.lr))
        print(f"=> loaded checkpoint '{args.resume}' (ep {checkpoint['epoch']} / it {checkpoint['iteration']} )")
    else:
        print(f"=> no checkpoint found at '{args.resume}'")


def prepare_optimizer(model, args):
    # optimizer
    args.vid_lr_mult = args.fine_tuning if args.vid_lr_mult is None else args.vid_lr_mult
    args.skm_lr_mult = args.fine_tuning if args.skm_lr_mult is None else args.skm_lr_mult
    args.prot_lr_mult = args.fine_tuning if args.prot_lr_mult is None else args.prot_lr_mult
    args.head_lr_mult = args.fine_tuning if args.head_lr_mult is None else args.head_lr_mult
    args.head_skm_lr_mult = args.head_lr_mult if args.head_skm_lr_mult is None else args.head_skm_lr_mult
    args.head_vid_lr_mult = args.head_lr_mult if args.head_vid_lr_mult is None else args.head_vid_lr_mult

    if min(args.vid_lr_mult, args.skm_lr_mult, args.prot_lr_mult) < 0.:
        raise ValueError
    else:
        print("====Training with Fine Tuning====")
        print(f'The classifier is finetuned with a learning rate of {args.head_lr_mult} x main learning rate.')
        print(f'The video backbone is finetuned with a learning rate of {args.vid_lr_mult} x main learning rate.')
        print(f'The skeleton backbone is finetuned with a learning rate of {args.skm_lr_mult} x main learning rate.')
        print(f'The prototype layer (if available) is finetuned with a learning rate of '
              f'{args.prot_lr_mult} x main learning rate.')

        print(f"{'Name':<42} {'Requires Grad':<6} Learning Rate")
        params = []
        for name, param in model.module.named_parameters():
            if is_head_layer(name):
                par_lr = args.lr * args.head_lr_mult

                if par_lr > 0:
                    params.append({'params': param, 'lr': par_lr, "name": name})
                else:
                    param.requires_grad = False

                print(f"{name:<50} {str(param.requires_grad):<6} {par_lr}")

            elif is_vid_head_layer(name, args):
                par_lr = args.lr * args.head_vid_lr_mult

                if par_lr > 0:
                    params.append({'params': param, 'lr': par_lr, "name": name})
                else:
                    param.requires_grad = False

                print(f"{name:<50} {str(param.requires_grad):<6} {par_lr}")

            elif is_skm_head_layer(name):
                par_lr = args.lr * args.head_skm_lr_mult

                if par_lr > 0:
                    params.append({'params': param, 'lr': par_lr, "name": name})
                else:
                    param.requires_grad = False

                print(f"{name:<50} {str(param.requires_grad):<6} {par_lr}")

            elif is_vid_layer(name):
                par_lr = args.lr * args.vid_lr_mult

                if par_lr > 0:
                    params.append({'params': param, 'lr': par_lr, "name": name})
                else:
                    param.requires_grad = False

                print(f"{name:<50} {str(param.requires_grad):<6} {par_lr}")

            elif is_sk_layer(name):
                par_lr = args.lr * args.skm_lr_mult

                if par_lr > 0:
                    params.append({'params': param, 'lr': par_lr, "name": name})
                else:
                    param.requires_grad = False

                print(f"{name:<50} {str(param.requires_grad):<6} {par_lr}")

            elif is_prot_layer(name):
                par_lr = args.lr * args.prot_lr_mult

                if par_lr > 0:
                    params.append({'params': param, 'lr': par_lr, "name": name})
                else:
                    param.requires_grad = False

                print(f"{name:<50} {str(param.requires_grad):<6} {par_lr}")

            else:
                raise ValueError
    if args.optimizer == "Adam":
        optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == "SGD":
        optimizer = optim.SGD(params, lr=args.lr, momentum=0.95, nesterov=True)
    else:
        raise ValueError

    print(f"Optimizer: {args.optimizer}")

    def lr_scheduler(ep):
        return multi_step_lr_scheduler(ep, gamma=0.1, step=args.scheduler_steps)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scheduler)

    return optimizer, scheduler


def select_model(args):
    if args.model == "dpc-resnet":
        model = DPCResnetClassifier(img_dim=args.img_dim,
                                    seq_len=args.seq_len,
                                    downsampling=args.ds_vid,
                                    vid_backbone=args.vid_backbone,
                                    num_class=args.num_class,
                                    dropout=args.dropout,
                                    representation_size=args.representation_size,
                                    hidden_width=args.hidden_width,
                                    classification_tapping=args.class_tapping,
                                    backbone_naming=args.backbone_naming,
                                    use_sk=args.use_skeleton_stream,
                                    mode=args.multi_mode,
                                    return_raw_features=(args.transfer_mode == "nearest_neighbour"),
                                    apply_log_soft=False
                                    )
    elif args.model == "sk-cont-s3d":
        model = S3DClassifier(vid_backbone=args.vid_backbone,
                              sk_backbone=args.sk_backbone,
                              num_class=args.num_class,
                              dropout=args.dropout,
                              representation_size=args.representation_size,
                              hidden_width=args.hidden_width,
                              swav_prototype_count=args.swav_prototypes,
                              classification_tapping=args.class_tapping,
                              mode=args.multi_mode,
                              return_raw_features=(args.transfer_mode == "nearest_neighbour"),
                              return_vid_tensor_instead_dict=args.captum,
                              apply_log_soft=False,
                              separate_vid_classifier=args.sep_vid_classifier,
                              separate_skm_classifier=args.sep_skm_classifier,
                              classify_with_backbone=args.classify_with_backbone,
                              multi_layer_classifier=args.use_multi_layer_classifier
                              )
    elif args.model == "r2+1d":
        model = R2plus1DClassifier(backbone=args.vid_backbone,
                                   num_class=args.num_class,
                                   dropout=args.dropout,
                                   representation_size=args.representation_size,
                                   hidden_fc_width=args.hidden_width,
                                   classification_tapping=args.class_tapping
                                   )
    else:
        raise ValueError('wrong model!')

    return model


def load_pretrained_or_resume(model, optimizer, args):
    args.old_lr = None

    if args.resume:
        prepare_resume(model, optimizer, args)
    else:
        args.best_train_loss = None
        args.best_train_acc = None
        args.best_val_loss = None
        args.best_val_acc = None

    if not args.resume and args.pretrain and args.pretrain != 'random':
        for sel, f, desc in zip([lambda n: True, is_vid_layer, is_sk_layer, is_prot_layer, is_head_layer],
                                [args.pretrain, args.pretrain_vid, args.pretrain_sk,
                                 args.pretrain_prot, args.pretrain_head],
                                ["all weights", "video backbone weights", "sk backbone weights", "class head weights"]):
            if f is None: continue

            print("---------")
            print(f"Loading {desc} from file.")
            if os.path.isfile(f):
                checkpoint = torch.load(f, map_location=torch.device('cpu'))

                owl, otl = "", ""
                if isinstance(checkpoint, OrderedDict):
                    weight_dict = checkpoint
                    otl = "module."
                    owl = "module.vid_backbone."
                    print(f"=> loading pretrained checkpoint '{f}'")
                else:
                    weight_dict = checkpoint['state_dict']
                    print(f"=> loading pretrained checkpoint '{f}' (epoch {checkpoint['epoch']})")
                model.module.load_weights_state_dict(weight_dict, model=model, selector=sel, print_miss=True,
                                                     own_lstrip=owl, other_lstrip=otl,
                                                     skip_size_mismatch=args.skip_size_mismatch)

                print(f"=> loaded pretrained checkpoint '{f}'")
            else:
                print(f"=> no checkpoint found at '{f}'")

    elif args.pretrain == 'random':
        print('=> using random weights')

    return args


def is_head_layer(nm):
    return any(re.match(il, nm) for il in [r".*final_fc_classifier\..*"])


def is_vid_head_layer(nm, args):
    vid_head_layers = [r".*final_fc_classifier_vid.*"]
    if args.classify_with_backbone: vid_head_layers.append(r".*vid_backbone.fc\..*")

    return any(re.match(il, nm) for il in vid_head_layers)


def is_skm_head_layer(nm):
    return any(re.match(il, nm) for il in [r".*final_fc_classifier_skm.*"])


def is_vid_layer(nm):
    return any(re.match(il, nm) for il in [".*vid_backbone.*", ".*vid_fc2.*", ".*vid_fc_rep"])


def is_sk_layer(nm):
    return any(re.match(il, nm) for il in [".*sk_backbone.*", ".*sk_fc_rep.*"])


def is_prot_layer(nm):
    return any(re.match(il, nm) for il in [".*prototype.*"])


if __name__ == '__main__':
    main()

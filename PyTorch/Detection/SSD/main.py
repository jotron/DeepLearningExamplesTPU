# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
from argparse import ArgumentParser
import torch
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR
import torch.utils.data.distributed

from ssd.model import SSD300, ResNet, Loss
from ssd.utils import dboxes300_coco, Encoder
from ssd.logger import Logger, BenchLogger
from ssd.evaluate import evaluate
from ssd.train import train_loop, tencent_trick, load_checkpoint, benchmark_train_loop, benchmark_inference_loop
from ssd.data import get_train_loader, get_val_dataset, get_val_dataloader, get_coco_ground_truth

import dllogger as DLLogger

import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.utils as xu

# Apex imports
try:
    from apex.parallel import DistributedDataParallel as DDP
except ImportError:
    print("APEX not installed")

def generate_mean_std(args, device):
    mean_val = [0.485, 0.456, 0.406]
    std_val = [0.229, 0.224, 0.225]

    mean = torch.tensor(mean_val).to(device)
    std = torch.tensor(std_val).to(device)

    view = [1, len(mean_val), 1, 1]

    mean = mean.view(*view)
    std = std.view(*view)

    return mean, std


def make_parser():
    parser = ArgumentParser(description="Train Single Shot MultiBox Detector"
                                        " on COCO")
    parser.add_argument('--data', '-d', type=str, default='/coco', required=True,
                        help='path to test and training data files')
    parser.add_argument('--epochs', '-e', type=int, default=65,
                        help='number of epochs for training')
    parser.add_argument('--batch-size', '--bs', type=int, default=32,
                        help='number of examples for each iteration')
    parser.add_argument('--eval-batch-size', '--ebs', type=int, default=32,
                        help='number of examples for each evaluation iteration')
    parser.add_argument('--no-cuda', action='store_true',
                        help='use available GPUs')
    parser.add_argument('--seed', '-s', type=int,
                        help='manually set random seed for torch')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='path to model checkpoint file')
    parser.add_argument('--save', type=str, default=None,
                        help='save model checkpoints in the specified directory')
    parser.add_argument('--mode', type=str, default='training',
                        choices=['training', 'evaluation', 'benchmark-training', 'benchmark-inference'])
    parser.add_argument('--evaluation', nargs='*', type=int, default=[21, 31, 37, 42, 48, 53, 59, 64],
                        help='epochs at which to evaluate')
    parser.add_argument('--multistep', nargs='*', type=int, default=[43, 54],
                        help='epochs at which to decay learning rate')

    # Hyperparameters
    parser.add_argument('--learning-rate', '--lr', type=float, default=2.6e-3,
                        help='learning rate')
    parser.add_argument('--momentum', '-m', type=float, default=0.9,
                        help='momentum argument for SGD optimizer')
    parser.add_argument('--weight-decay', '--wd', type=float, default=0.0005,
                        help='momentum argument for SGD optimizer')

    parser.add_argument('--warmup', type=int, default=None)
    parser.add_argument('--benchmark-iterations', type=int, default=20, metavar='N',
                        help='Run N iterations while benchmarking (ignored when training and validation)')
    parser.add_argument('--benchmark-warmup', type=int, default=20, metavar='N',
                        help='Number of warmup iterations for benchmarking')

    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
    parser.add_argument('--backbone-path', type=str, default=None,
                        help='Path to chekcpointed backbone. It should match the'
                             ' backbone model declared with the --backbone argument.'
                             ' When it is not provided, pretrained model from torchvision'
                             ' will be downloaded.')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--amp', action='store_true',
                        help='Whether to enable AMP ops. When false, uses TF32 on A100 and FP32 on V100 GPUS.')
    parser.add_argument('--log-interval', type=int, default=20,
                        help='Logging interval.')
    parser.add_argument('--json-summary', type=str, default=None,
                        help='If provided, the json summary will be written to'
                             'the specified file.')

    # Distributed
    parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK',0), type=int,
                        help='Used for multi-process training. Can either be manually set ' +
                             'or automatically set by using \'python -m multiproc\'.')

    # TPU
    parser.add_argument('--parallel_loader', action='store_true',
                        help='Upload data to devie in Background')
    parser.add_argument('--num_cores', type=int, default=8,
                        help='Number of tpu cores')
    parser.add_argument('--suppress_loss_report', action='store_true',
                        help='Dont print loss')
    parser.add_argument('--accumulation',  type=int, default=1,
                        help='Accumulation')
    parser.add_argument('--fake_data',  action='store_true',
                        help='Use data generated on the fly')

    return parser


def train(index, train_loop_func, logger, args):
    # Setup multi-GPU if necessary
    args.distributed = False
    args.N_gpu = 1

    if args.seed is None:
        args.seed = np.random.randint(1e4)

    if args.distributed:
        args.seed = (args.seed + torch.distributed.get_rank()) % 2**32
    print("Using seed = {}".format(args.seed))
    torch.manual_seed(args.seed)
    np.random.seed(seed=args.seed)

    # Setup TPU
    device = xm.xla_device()
    args.local_rank = xm.get_ordinal()
    setattr(args, 'world_size', xm.xrt_world_size())
    xm.master_print(f"Global Batchsize is {args.world_size * args.batch_size * args.accumulation}")
    xm.rendezvous("setup of training")
    print(f"XLA DEVICE SETUP. {args.local_rank}")

    # DEBUG LOGGER
    if args.local_rank == 0 and logger is None:
        logger = Logger('Training logger', log_interval=args.log_interval,
                        json_output=args.json_summary)
        log_params(logger, args)
        print("Initialized Logger!")

    # Setup data, defaults
    dboxes = dboxes300_coco()
    encoder = Encoder(dboxes)
    cocoGt = get_coco_ground_truth(args)

    train_loader = None
    if not args.fake_data:
        train_loader = get_train_loader(args, args.seed - 2**31)
    else:
        xm.master_print("Using fake data!")
        train_loader = fake_train_loader(args, device)

    val_dataset = get_val_dataset(args)
    val_dataloader = get_val_dataloader(val_dataset, args)

    # Enable background data upload
    if (args.parallel_loader):
        xm.master_print("--parallel_loader enabled!")
        train_loader = pl.MpDeviceLoader(train_loader, device)

    # Upload model to device
    ssd300 = SSD300(backbone=ResNet(args.backbone, args.backbone_path)).to(device)
    args.learning_rate = args.learning_rate * xm.xrt_world_size() * args.accumulation * (args.batch_size / 32)
    start_epoch = 0
    iteration = 0
    loss_func = Loss(dboxes)

    optimizer = torch.optim.SGD(tencent_trick(ssd300), lr=args.learning_rate,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer=optimizer, milestones=args.multistep, gamma=0.1)

    if args.distributed:
        ssd300 = DDP(ssd300)

    if args.checkpoint is not None:
        if os.path.isfile(args.checkpoint):
            load_checkpoint(ssd300.module if args.distributed else ssd300, args.checkpoint)
            checkpoint = torch.load(args.checkpoint,
                                    map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()))
            start_epoch = checkpoint['epoch']
            iteration = checkpoint['iteration']
            scheduler.load_state_dict(checkpoint['scheduler'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print('Provided checkpoint is not path to a file')
            return

    inv_map = {v: k for k, v in val_dataset.label_map.items()}

    total_time = 0

    if args.mode == 'evaluation':
        if args.num_cores > 1:
            xm.master_print("Evaluation with multiple cores not supported yet!")
            return
        acc = evaluate(ssd300, val_dataloader, cocoGt, encoder, inv_map, args, device)
        if args.local_rank == 0:
            print('Model precision {} mAP'.format(acc))
        return

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    mean, std = generate_mean_std(args, device)

    for epoch in range(start_epoch, args.epochs):
        start_epoch_time = time.time()
        iteration = train_loop_func(ssd300, loss_func, scaler,
                                    epoch, optimizer, train_loader, val_dataloader, encoder, iteration,
                                    logger, args, mean, std, device)
        if args.mode in ["training", "benchmark-training"]:
            scheduler.step()
        end_epoch_time = time.time() - start_epoch_time
        total_time += end_epoch_time

        if args.local_rank == 0:
            logger.update_epoch_time(epoch, end_epoch_time)

        if epoch in args.evaluation:
            if args.local_rank == 0:
                acc = evaluate(ssd300, val_dataloader, cocoGt, encoder, inv_map, args, device)
                logger.update_epoch(epoch, acc)
            xm.rendezvous("post-evaluation")

        if args.save and args.local_rank == 0:
            print("saving model...")
            obj = {'epoch': epoch + 1,
                   'iteration': iteration,
                   'optimizer': optimizer.state_dict(),
                   'scheduler': scheduler.state_dict(),
                   'label_map': val_dataset.label_info}
            if args.distributed:
                obj['model'] = ssd300.module.state_dict()
            else:
                obj['model'] = ssd300.state_dict()
            save_path = os.path.join(args.save, f'epoch_{epoch}.pt')
            torch.save(obj, save_path)
            logger.log('model path', save_path)
        if not args.parallel_loader:
            train_loader.reset()
        else:
            train_loader._loader.reset()
    if args.local_rank == 0:
        DLLogger.log((), { 'total time': total_time })
        logger.log_summary()


def log_params(logger, args):
    logger.log_params({
        "dataset path": args.data,
        "epochs": args.epochs,
        "batch size": args.batch_size,
        "eval batch size": args.eval_batch_size,
        "no cuda": args.no_cuda,
        "seed": args.seed,
        "checkpoint path": args.checkpoint,
        "mode": args.mode,
        "eval on epochs": args.evaluation,
        "lr decay epochs": args.multistep,
        "learning rate": args.learning_rate,
        "momentum": args.momentum,
        "weight decay": args.weight_decay,
        "lr warmup": args.warmup,
        "backbone": args.backbone,
        "backbone path": args.backbone_path,
        "num workers": args.num_workers,
        "AMP": args.amp,
        "precision": 'bf16' if os.getenv("XLA_USE_BF16") else 'fp32',
    })


def fake_train_loader(args, device):
    num_samples = 100000
    img_dim = 300
    gen_img = lambda b,c,d,w: torch.rand([b,c,d,w])
    gen_bbox = lambda b: torch.rand([b*8732, 4])
    gen_offs = lambda b: torch.rand([b+1])
    gen_label = lambda b: torch.zeros(size=[b*8732])
    gen_x_func = lambda b,c,d,w: ((gen_img(b,c,d,w),),(gen_bbox(b),),(gen_label(b),),(gen_offs(b),))
    gen_y_func = lambda x: 0

    train_loader = xu.FnDataGenerator(gen_y_func, args.batch_size, gen_x_func, dims=[3, img_dim, img_dim], count=num_samples)
    return train_loader


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    args.local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    if args.local_rank == 0:
        os.makedirs('./models', exist_ok=True)

    torch.backends.cudnn.benchmark = True

    # write json only on the main thread
    args.json_summary = args.json_summary if args.local_rank == 0 else None

    if args.mode == 'benchmark-training':
        train_loop_func = benchmark_train_loop
        logger = BenchLogger('Training benchmark', log_interval=args.log_interval,
                             json_output=args.json_summary)
        args.epochs = 1
    elif args.mode == 'benchmark-inference':
        train_loop_func = benchmark_inference_loop
        logger = BenchLogger('Inference benchmark', log_interval=args.log_interval,
                             json_output=args.json_summary)
        args.epochs = 1
    else:
        train_loop_func = train_loop
        #logger = Logger('Training logger', log_interval=args.log_interval,
        #                json_output=args.json_summary)
        logger = None
    xmp.spawn(train, args=(train_loop_func, logger, args), nprocs=args.num_cores)

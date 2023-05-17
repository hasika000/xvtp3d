import torch
import os
import argparse
import time
# import logging
import json
import numpy as np
import sys
sys.path.append(os.path.join(os.getcwd(), 'external_cython'))
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# from utils import log_help
from configs import cfg
from tensorboardX import SummaryWriter
from utils.func_lab import make_dir, get_metric_dict, print_info, get_miss_rate, MyLogger
from dataloader.argo_loader import make_dataloader
from modeling.my_class import TNT3D
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4'  # 多卡训练记得通过这里控制device


def main():  # todo: 把所有np的reshape改成[:, None, :]这样的
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='configs/config_argo.yml', type=str)
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--load_model_path', type=str, default='')
    parser.add_argument('opts', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    save_dir = os.path.join('..', cfg.save_dir, cfg.save_tag, 'train_' + time.strftime("%Y_%m_%d_%H_%M"))
    make_dir(save_dir)

    # log_help.set_logger(os.path.join(save_dir, 'train_' + time.strftime("%d_%m_%Y_%H_%M_%S") + '.log'))
    logger = MyLogger(os.path.join(save_dir, 'train_' + time.strftime("%d_%m_%Y_%H_%M_%S") + '.log'))

    if torch.cuda.is_available():
        device = cfg.device
    else:
        device = 'cpu'

    if torch.cuda.device_count() < cfg.distributed:
        print('warring, not enough gpu device')
        cfg.distributed = 1

    # change cfg
    if cfg.modality != 'both':
        cfg.MODEL.cross_type = 0
        cfg.MODEL.share_weight = True
        cfg.MODEL.cross_enc = False
    if cfg.MODEL.cross_type == 0:
        assert cfg.modality != 'both'
    if not cfg.MODEL.share_weight:
        assert cfg.modality == 'both'

    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f)

    print_info(args, cfg, device, save_dir, logger)

    if cfg.distributed == 1:
        main_process_nodist(cfg, args, device, save_dir, logger)
    else:
        main_process_dist(cfg, args, device, save_dir, logger)


def main_process_nodist(cfg, args, device, save_dir, logger):
    tnt3d = TNT3D(cfg, device, cfg.modality).to(device)

    p_1 = [p for n, p in tnt3d.named_parameters() if 'complete_traj' not in n]
    p_2 = [p for n, p in tnt3d.named_parameters() if 'complete_traj' in n]

    optimizer = torch.optim.Adam(p_1, lr=cfg.SOLVER.lr)
    optimizer_2 = torch.optim.Adam(p_2, lr=cfg.SOLVER.lr)

    if cfg.SOLVER.scheduler_lr_type == 'm':
        scheduler_lr = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.SOLVER.milestones, gamma=0.1)
    elif cfg.SOLVER.scheduler_lr_type == 's':
        scheduler_lr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.SOLVER.step_size, gamma=0.1)
    else:
        print('error scheduler_lr_type')
        scheduler_lr = None

    # load modeling
    if args.load_model:
        checkpoint_path = args.load_model_path
        if not os.path.exists(checkpoint_path):
            train_from_epoch = 0
            logger.info('train from {} ---- fail loading from {}'.format(train_from_epoch, checkpoint_path))
        else:
            checkpoint = torch.load(checkpoint_path, map_location=device)

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler_lr.load_state_dict(checkpoint['scheduler_lr_state_dict'])
            tnt3d.load_state_dict(checkpoint['model'])

            train_from_epoch = int(checkpoint_path[-8:-4])
            logger.info('train from {} ---- loading from {}'.format(train_from_epoch, checkpoint_path))
    else:
        train_from_epoch = 0
        logger.info('train from {}'.format(train_from_epoch))

    # load data
    logger.info('preparing train dataloader')
    train_dataloader = make_dataloader(cfg, 'train')
    if cfg.test_include:
        logger.info('preparing test dataloader')
        test_dataloader = make_dataloader(cfg, 'val')
    else:
        test_dataloader = None

    writer = SummaryWriter(os.path.join(save_dir, 'summary_' + time.strftime("%d_%m_%Y_%H_%M_%S")))

    # main loop
    for epoch in range(train_from_epoch + 1, cfg.num_epochs + 1):
        #----train----
        train(cfg, device, epoch, train_dataloader, optimizer, scheduler_lr, writer, logger, tnt3d, optimizer_2)

        #----test----
        if cfg.test_include:
            pass

        # save
        checkpoint = {'epoch': epoch,
                      'optimizer_state_dict': optimizer.state_dict(),
                      'scheduler_lr_state_dict': scheduler_lr.state_dict(),
                      'model': tnt3d.state_dict()}
        checkpoint_path = os.path.join(save_dir, 'model_epoch_' + str(epoch).zfill(4) + '.tar')
        if cfg.save_model and (epoch % cfg.save_interval == 0):
            torch.save(checkpoint, checkpoint_path)
            logger.info('save modeling to file: {}'.format(save_dir))


def train(cfg, device, epoch, dataloader, optimizer, scheduler_lr, writer, logger, tnt3d, optimizer_2=None):
    epoch_loss = 0
    tnt3d.train()
    infer = False
    if cfg.modality == 'both':
        li_FDE = {'bev': [], 'fpv': []}
    else:
        li_FDE = {cfg.modality: []}

    for step, batch in enumerate(dataloader):
        time_start = time.time()
        loss, FDE_dict, _, _ = tnt3d(batch, infer)
        loss.backward()

        if cfg.modality == 'both':
            li_FDE['bev'].extend([each for each in FDE_dict['bev']])
            li_FDE['fpv'].extend([each for each in FDE_dict['fpv']])
        else:
            li_FDE[cfg.modality].extend([each for each in FDE_dict[cfg.modality]])

        if optimizer_2 is not None:
            optimizer_2.step()
            optimizer_2.zero_grad()

        optimizer.step()
        optimizer.zero_grad()

        time_end = time.time()

        writer.add_scalar('total_loss/iter', loss.item(), step + (epoch - 1) * len(dataloader))

        logger.info(
            '{}/{}, [epoch {}/{}], loss = {:.6f}, lr = {}, time/batch = {:.3f}'.format(
                step, len(dataloader), epoch, cfg.num_epochs,
                loss.item(), optimizer.param_groups[0]['lr'], time_end - time_start))

        epoch_loss += loss.item()

    scheduler_lr.step()

    epoch_loss = epoch_loss / len(dataloader)
    writer.add_scalar('loss/epoch', epoch_loss, epoch)
    logger.info('finish train epoch {}, epoch loss = {}'.format(epoch, epoch_loss))
    for key in li_FDE:
        miss_rates = (get_miss_rate(li_FDE[key], dis=2.0), get_miss_rate(li_FDE[key], dis=4.0), get_miss_rate(li_FDE[key], dis=6.0))
        logger.info('MODAL = {}, FDE = {}, MR = ({},{},{})'.format(
            key, np.mean(li_FDE[key]) if len(li_FDE[key]) > 0 else None, miss_rates[0], miss_rates[1], miss_rates[2]))
    logger.info('----------------------------')

    return epoch_loss


def main_process_dist(cfg, args, device, save_dir, logger):
    queue = mp.Manager().Queue()
    kwargs = {'cfg': cfg, 'args': args, 'save_dir': save_dir, 'logger': logger}
    spawn_context = mp.spawn(demo_basic,
                             args=(cfg.distributed, kwargs, queue),
                             nprocs=cfg.distributed,
                             join=False)
    queue.put(True)
    while not spawn_context.join():
        pass


def demo_basic(rank, world_size, kwargs, queue):
    cfg = kwargs['cfg']
    args = kwargs['args']
    save_dir = kwargs['save_dir']
    logger = kwargs['logger']
    device = 'cuda:' + str(rank)

    master_port = '12355'

    logger.info('Running DDP on rank {}'.format(rank))

    def setup(rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = master_port

        # initialize the process group
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    setup(rank, world_size)

    tnt3d = TNT3D(cfg, device, cfg.modality).to(device)

    p_1 = [p for n, p in tnt3d.named_parameters() if 'complete_traj' not in n]
    p_2 = [p for n, p in tnt3d.named_parameters() if 'complete_traj' in n]

    optimizer = torch.optim.Adam(p_1, lr=cfg.SOLVER.lr)
    optimizer_2 = torch.optim.Adam(p_2, lr=cfg.SOLVER.lr)

    if cfg.SOLVER.scheduler_lr_type == 'm':
        scheduler_lr = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.SOLVER.milestones, gamma=0.1)
    elif cfg.SOLVER.scheduler_lr_type == 's':
        scheduler_lr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.SOLVER.step_size, gamma=0.1)
    else:
        print('error scheduler_lr_type')
        scheduler_lr = None

    # load modeling
    if args.load_model:
        checkpoint_path = args.load_model_path
        if not os.path.exists(checkpoint_path):
            train_from_epoch = 0
            logger.info('train from {} ---- fail loading from {}'.format(train_from_epoch, checkpoint_path))
        else:
            checkpoint = torch.load(checkpoint_path)

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler_lr.load_state_dict(checkpoint['scheduler_lr_state_dict'])
            tnt3d.load_state_dict(checkpoint['model'])

            train_from_epoch = int(checkpoint_path[-8:-4])
            logger.info('train from {} ---- loading from {}'.format(train_from_epoch, checkpoint_path))
    else:
        train_from_epoch = 0
        logger.info('train from {}'.format(train_from_epoch))

    tnt3d = DDP(tnt3d, device_ids=[rank], find_unused_parameters=True)

    # load data
    logger.info('preparing train dataloader')
    train_dataloader = make_dataloader(cfg, 'train')
    if cfg.test_include:
        logger.info('preparing test dataloader')
        test_dataloader = make_dataloader(cfg, 'val')
    else:
        test_dataloader = None

    if device == 'cuda:0':
        writer = SummaryWriter(os.path.join(save_dir, 'summary_' + time.strftime("%d_%m_%Y_%H_%M_%S")))
    else:
        writer = None

    # distribute setting
    if rank == 0:
        receive = queue.get()
        assert receive == True

    dist.barrier()

    print(dist.get_world_size())

    # main loop
    for epoch in range(train_from_epoch + 1, cfg.num_epochs + 1):
        #----train----
        train_one_epoch(cfg, device, epoch, queue, train_dataloader, optimizer, scheduler_lr, writer, logger, tnt3d, optimizer_2)

        #----test----
        if cfg.test_include:
            pass

        # save
        if device == 'cuda:0':
            checkpoint_path = os.path.join(save_dir, 'model_epoch_' + str(epoch).zfill(4) + '.tar')
            if cfg.save_model and (epoch % cfg.save_interval == 0):
                model_to_save = tnt3d.module if hasattr(tnt3d, 'module') else tnt3d

                checkpoint = {'epoch': epoch,
                              'optimizer_state_dict': optimizer.state_dict(),
                              'scheduler_lr_state_dict': scheduler_lr.state_dict(),
                              'model': model_to_save.state_dict()}

                torch.save(checkpoint, checkpoint_path)
                logger.info('save modeling to file: {}'.format(save_dir))

        dist.barrier()

    dist.destroy_process_group()


def train_one_epoch(cfg, device, epoch, queue, dataloader, optimizer, scheduler_lr, writer, logger, tnt3d, optimizer_2):
    epoch_loss = 0
    tnt3d.train()
    infer = False
    if cfg.modality == 'both':
        li_FDE = {'bev': [], 'fpv': []}
    else:
        li_FDE = {cfg.modality: []}

    for step, batch in enumerate(dataloader):
        time_start = time.time()
        loss, FDE_dict, _, _ = tnt3d(batch, infer)
        loss.backward()

        if cfg.modality == 'both':
            li_FDE['bev'].extend([each for each in FDE_dict['bev']])
            li_FDE['fpv'].extend([each for each in FDE_dict['fpv']])
        else:
            li_FDE[cfg.modality].extend([each for each in FDE_dict[cfg.modality]])

        if optimizer_2 is not None:
            optimizer_2.step()
            optimizer_2.zero_grad()

        optimizer.step()
        optimizer.zero_grad()

        time_end = time.time()

        if device == 'cuda:0':  # 训练过程，仅仅监控cuda0的loss变化
            writer.add_scalar('total_loss/iter', loss.item(), step + (epoch - 1) * len(dataloader))

            logger.info(
                '{}/{}, [epoch {}/{}], loss = {:.6f}, lr = {}, time/batch = {:.3f}'.format(
                    step, len(dataloader), epoch, cfg.num_epochs,
                    loss.item(), optimizer.param_groups[0]['lr'], time_end - time_start))

        epoch_loss += loss.item()

    scheduler_lr.step()

    # 如果非主卡，则把需要print的信息存到队列中，如果是主卡，则从队列中获取所以信息，之后log
    if device == 'cuda:0':
        total_len = len(dataloader)
        for i in range(dist.get_world_size() - 1):
            info_dict = queue.get()
            epoch_loss += info_dict['epoch_loss']
            total_len += info_dict['len']
            for key in li_FDE.keys():
                li_FDE[key].extend(info_dict['FDEs'][key])

        # log and write
        epoch_loss = epoch_loss / total_len
        writer.add_scalar('loss/epoch', epoch_loss, epoch)
        logger.info('finish train epoch {}, epoch loss = {}'.format(epoch, epoch_loss))
        for key in li_FDE:
            miss_rates = (
                get_miss_rate(li_FDE[key], dis=2.0), get_miss_rate(li_FDE[key], dis=4.0),
                get_miss_rate(li_FDE[key], dis=6.0))
            logger.info('MODAL = {}, FDE = {}, MR = ({},{},{})'.format(
                key, np.mean(li_FDE[key]) if len(li_FDE[key]) > 0 else None, miss_rates[0], miss_rates[1], miss_rates[2]))
        logger.info('----------------------------')

    else:
        info_dict = {'epoch_loss': epoch_loss, 'len': len(dataloader), 'FDEs': li_FDE}
        queue.put(info_dict)


# def val_test(cfg, device, epoch, dataloader, scheduler_lr, writer, val_test_flag, vae_model):
#     with torch.no_grad():
#         logging.info('--------start {} epoch: {}--------'.format(val_test_flag, epoch))
#         metric_dict = {'ade_ml': [], 'fde_ml': [], 'ade_bon': [], 'fde_bon': []}
#
#         for batch in dataloader:
#
#             obs_traj, future_traj, first_history_indices, last_state, neighbor_traj = batch_to_data(cfg, batch, device)
#
#             # 用num_sample区分是测试bon还是ml，args.num_sample都是20，但是ml传入model的是1，而bon传入model的是20
#             # ml
#             y_pred, kl_loss = vae_model(obs_traj,
#                                         future_traj,
#                                         first_history_indices,
#                                         last_state,
#                                         neighbor_traj,
#                                         num_samples=1,
#                                         infer=True)
#
#             pos_pred = y_pred.detach().to('cpu').numpy()
#             pos_gt = future_traj.detach().to('cpu').numpy()
#
#             metric_ade_fde = get_metric_dict(pos_pred, pos_gt, 1)  # ml的num_sample取1
#             metric_dict['ade_ml'].append(metric_ade_fde['ade'])
#             metric_dict['fde_ml'].append(metric_ade_fde['fde'])
#
#             # bon
#             y_pred, kl_loss = vae_model(obs_traj,
#                                         future_traj,
#                                         first_history_indices,
#                                         last_state,
#                                         neighbor_traj,
#                                         num_samples=cfg.TEST.num_samples,
#                                         infer=True)
#
#             pos_pred = y_pred.detach().to('cpu').numpy()
#             pos_gt = future_traj.detach().to('cpu').numpy()
#
#             metric_ade_fde = get_metric_dict(pos_pred, pos_gt, cfg.TEST.num_samples)  # ml的num_sample取1
#             metric_dict['ade_bon'].append(metric_ade_fde['ade'])
#             metric_dict['fde_bon'].append(metric_ade_fde['fde'])
#
#         # 求batch之间的平均值，之后log
#         metric_dict_epoch = dict()
#         for metric_type in metric_dict.keys():
#             metric_dict_epoch[metric_type] = np.mean(np.array(metric_dict[metric_type]))
#
#         # 如果是p，在这里step
#         if val_test_flag == 'val' and cfg.SOLVER.scheduler_lr_type == 'p':
#             scheduler_lr.step(metric_dict_epoch['ade_ml'])
#
#         for metric_type in list(metric_dict_epoch.keys()):
#             writer.add_scalar(val_test_flag + metric_type, metric_dict_epoch[metric_type], epoch)
#
#         logging.info('finish {} epoch {}'.format(val_test_flag, epoch))
#         for metric_type in list(metric_dict_epoch.keys()):
#             logging.info(metric_type + ' = {}'.format(metric_dict_epoch[metric_type]))
#         logging.info('----------------------------')
#
#         return metric_dict_epoch


if __name__ == '__main__':
    main()
import os
import sys
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
import torchvision
import torch.optim as optim
from utilss.utils import init_distributed_mode, AverageMeter, reduce_tensor, accuracy
from utilss.logger import setup_logger
import clip

from pathlib import Path
import yaml
import pprint
from dotmap import DotMap
import numpy as np
import datetime
import shutil

from contextlib import suppress
from datasets import Video_dataset
from modules.video_clip import video_header, VideoCLIP
from modules.coop import CoopCLIP
from utilss.Augmentation import get_augmentation
from utilss.solver import _lr_scheduler
from modules.text_prompt import text_prompt
from modules.Visual_prompt import visual_prompt


class TextCLIP(nn.Module):
    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self, text):
        return self.model.encode_text(text)  # text: [batch_size, seq_len]


class ImageCLIP(nn.Module):
    def __init__(self, model):
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self, image):
        return self.model.encode_image(image)


def main():
    global args, best_prec1
    global global_step
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default="/home/ljj/LG/V-APT/configs/ucf101/ucf_train.yaml")
    parser.add_argument('--log_time', default='')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    working_dir = os.path.join('./mylunwne', config['network']['type'], config['network']['arch'], config['data']['dataset'],
                               args.log_time)

    print('-' * 80)
    print(' ' * 20, "working dir: {}".format(working_dir))
    print('-' * 80)

    print('-' * 80)
    print(' ' * 30, "Config")
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)
    print('-' * 80)

    config = DotMap(config)

    Path(working_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, working_dir)
    shutil.copy('train.py', working_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.

    model, clip_state_dict = clip.load(config.network.arch, device=device, jit=False, tsm=config.network.tsm,
                                       T=config.data.num_segments, dropout=config.network.drop_out,
                                       emb_dropout=config.network.emb_dropout, pretrain=config.network.init,
                                       joint=config.network.joint)  # Must set jit=False for training  ViT-B/32

    transform_train = get_augmentation(True, config)
    transform_val = get_augmentation(False, config)



    if config.data.randaug.N > 0:
        transform_train = randAugment(transform_train, config)

    print('train transforms: {}'.format(transform_train.transforms))
    print('val transforms: {}'.format(transform_val.transforms))

    # visual fusion_model
    fusion_model = visual_prompt(config.network.sim_header, clip_state_dict, config.data.num_segments)

    # Params
    n_trainable_params = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            # print('Trainable param: %s, %s, %s' % (n, p.size(), p.dtype))
            n_trainable_params += p.numel()
    print('Total trainable params:', n_trainable_params, '(%.2f M)' % (n_trainable_params / 1000000))
    print('-----------------------------------')
    n_trainable_params_fusion = sum(p.numel() for p in fusion_model.parameters() if p.requires_grad)
    print('Total trainable params_fusion:', n_trainable_params_fusion, '(%.2f M)' % (n_trainable_params_fusion / 1000000))

    model_text = TextCLIP(model)
    model_image = ImageCLIP(model)
    model_text = torch.nn.DataParallel(model_text).cuda()
    model_image = torch.nn.DataParallel(model_image).cuda()
    fusion_model = torch.nn.DataParallel(fusion_model).cuda()


    train_data = Action_DATASETS(config.data.train_list, config.data.label_list, num_segments=config.data.num_segments,
                                 image_tmpl=config.data.image_tmpl, random_shift=config.data.random_shift,
                                 transform=transform_train)
    train_loader = DataLoader(train_data, batch_size=config.data.batch_size, num_workers=config.data.workers,
                              shuffle=True, pin_memory=False, drop_last=True)
    val_data = Action_DATASETS(config.data.val_list, config.data.label_list, random_shift=False,
                               num_segments=config.data.num_segments, image_tmpl=config.data.image_tmpl,
                               transform=transform_val)
    val_loader = DataLoader(val_data, batch_size=config.data.batch_size, num_workers=config.data.workers, shuffle=False,
                            pin_memory=False, drop_last=True)

    if device == "cpu":
        model_text.float()
        model_image.float()
    else:
        clip.model.convert_weights(
            model_text)  # Actually this line is unnecessary since clip by default already on float16
        clip.model.convert_weights(model_image)

    loss_img = KLLoss()
    loss_txt = KLLoss()


    start_epoch = config.solver.start_epoch

    if config.pretrain:
        if os.path.isfile(config.pretrain):
            print(("=> loading checkpoint '{}'".format(config.pretrain)))
            checkpoint = torch.load(config.pretrain)
            model.load_state_dict(checkpoint['model_state_dict'])
            fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.resume)))

    if config.resume:
        if os.path.isfile(config.resume):
            print(("=> loading checkpoint '{}'".format(config.resume)))
            checkpoint = torch.load(config.resume)
            model.load_state_dict(checkpoint['model_state_dict'])
            fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])
            start_epoch = checkpoint['epoch']
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(config.evaluate, start_epoch)))
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.pretrain)))


    classes, num_text_aug, text_dict = text_prompt(train_data)


    optimizer = _optimizer(config, model, fusion_model)
    lr_scheduler = _lr_scheduler(config, optimizer)

    best_prec1 = 0.0
    if config.solver.evaluate:
        prec1 = validate(start_epoch, val_loader, classes, device, model, fusion_model, config, num_text_aug)
        return

    for k, v in model.named_parameters():
        print('{}: {}'.format(k, v.requires_grad))
    for epoch in range(start_epoch, config.solver.epochs):
        model_image.train()
        model_text.train()
        fusion_model.train()
        for kkk, (images, list_id) in enumerate(tqdm(train_loader)):
            if config.solver.type != 'monitor':
                if (kkk + 1) == 1 or (kkk + 1) % 10 == 0:
                    lr_scheduler.step(epoch + kkk / len(train_loader))
            optimizer.zero_grad()

            images = images.view((-1, config.data.num_segments, 3) + images.size()[-2:])  # [16, 8, 3, 224, 224]
            b, t, c, h, w = images.size()
            text_id = numpy.random.randint(num_text_aug, size=len(list_id))
            texts = torch.stack([text_dict[j][i, :] for i, j in zip(list_id, text_id)])  # [16, 77]

            images = images.to(device).view(-1, c, h,
                                            w)  # [128, 3, 224, 224]omit the Image.fromarray if the images already in PIL format, change this line to images=list_image if using preprocess inside the dataset class
            texts = texts.to(device)

            image_embedding = model_image(images)  # [16, 8, 512]
            image_embedding = image_embedding.view(b, t, -1)  # [16, 512]
            image_embedding = fusion_model(image_embedding)   # [128, 512]
            text_embedding = model_text(texts)  # [16, 512]



            # fix_text ?
            # print("Turning off gradients in both the image and the text encoder")
            if config.network.fix_text:
                text_embedding.detach_()

            logit_scale = model.logit_scale.exp()
            logits_per_image, logits_per_text = create_logits(image_embedding, text_embedding, logit_scale)

            ground_truth = torch.tensor(gen_label(list_id), dtype=image_embedding.dtype, device=device)
            loss_imgs = loss_img(logits_per_image, ground_truth)
            loss_texts = loss_txt(logits_per_text, ground_truth)
            total_loss = (loss_imgs + loss_texts) / 2
            total_loss.backward()

            if device == "cpu":
                optimizer.step()
            else:
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)

        if epoch % config.logging.eval_freq == 0:  # and epoch>0
            prec1 = validate(epoch, val_loader, classes, device, model, fusion_model, config, num_text_aug)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        print('Testing: {}/{}'.format(prec1, best_prec1))
        print('Saving:')
        filename = "{}/last_model.pt".format(working_dir)

        epoch_saving(epoch, model, fusion_model, optimizer, filename)
        if is_best:
            best_saving(working_dir, epoch, model, fusion_model, optimizer)


if __name__ == '__main__':
    main()


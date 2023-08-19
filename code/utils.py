import collections.abc
import os
import os.path as osp
from torch import nn
import kornia.augmentation as K
import pydiffvg
import save_svg
import cv2
from ttf import font_string_to_svgs, normalize_letter_size
import torch
import numpy as np


def edict_2_dict(x):
    if isinstance(x, dict):
        xnew = {}
        for k in x:
            xnew[k] = edict_2_dict(x[k])
        return xnew
    elif isinstance(x, list):
        xnew = []
        for i in range(len(x)):
            xnew.append( edict_2_dict(x[i]))
        return xnew
    else:
        return x


def check_and_create_dir(path):
    pathdir = osp.split(path)[0]
    if osp.isdir(pathdir):
        pass
    else:
        os.makedirs(pathdir)


def update(d, u):
    """https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth"""
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d



def get_data_augs(cut_size):
    augmentations = []
    augmentations.append(K.RandomPerspective(distortion_scale=0.5, p=0.7))
    augmentations.append(K.RandomCrop(size=(cut_size, cut_size), pad_if_needed=True, padding_mode='reflect', p=1.0))
    return nn.Sequential(*augmentations)


'''pytorch adaptation of https://github.com/google/mipnerf'''
class Learning_rate_decay:
    def __init__(self,
                lr_init,
                lr_warmup,
                lr_final,
                max_step,
                max_warmup_step,
                T0 = 10,
                T_mult = 2):
        self.lr_init = lr_init
        self.lr_warmup = lr_warmup
        self.lr_final = lr_final
        self.max_step = max_step
        self.max_warmup_step = max_warmup_step
        self.T_schedule = [T0]
        self.T_mult = T_mult
    def __call__(self,step):
        if step < self.max_warmup_step:
            return self.lr_init + (self.lr_warmup - self.lr_init) / self.max_warmup_step * step
        else:
            step -= self.max_warmup_step
            for T_max in self.T_schedule:
                if step >= T_max:
                    step -= T_max
                else:
                    return self.lr_final + 0.5 * (self.lr_warmup - self.lr_final) * (1 + np.cos(step / T_max * np.pi))
            T_max = self.T_schedule[-1] * self.T_mult
            self.T_schedule.append(T_max)
            return self.lr_final + 0.5 * (self.lr_warmup - self.lr_final) * (1 + np.cos(step / T_max * np.pi))    
def learning_rate_decay(step,
                        lr_init,
                        lr_warmup,
                        lr_final,
                        max_step,
                        max_warmup_step,
                        T_max = 0):
    
    if T_max == 0:
        T_max = max_step - max_warmup_step
    # 处于warm_up阶段
    if step < max_warmup_step:
        return lr_init + (lr_warmup - lr_init) / max_warmup_step * step
    else:
        step -= max_warmup_step
        return lr_final + 0.5 * (lr_warmup - lr_final) * (1 + np.cos(step/T_max * np.pi))


def save_image(img, filename, gamma=1):
    check_and_create_dir(filename)
    imshow = img.detach().cpu()
    pydiffvg.imwrite(imshow, filename, gamma=gamma)


def create_video(num_iter, experiment_dir, video_frame_freq,w,h):
    img_array = []
    for ii in range(0, num_iter):
        if ii % video_frame_freq == 0 or ii == num_iter - 1:
            filename = os.path.join(
                experiment_dir, "video-png", f"iter{ii:04d}.png")
            img = cv2.imread(filename)
            img_array.append(img)

    video_name = os.path.join(
        experiment_dir, "video.mp4")
    check_and_create_dir(video_name)
    out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (w, h))
    for iii in range(len(img_array)):
        out.write(img_array[iii])
    out.release()

def get_experiment_id(debug=False):
    if debug:
        return 999999999999
    import time
    time.sleep(0.5)
    return int(time.time()*100)

def get_path_schedule(type, **kwargs):
    if type == 'repeat':
        max_path = kwargs['max_path']
        schedule_each = kwargs['schedule_each']
        return [schedule_each] * max_path
    elif type == 'list':
        schedule = kwargs['schedule']
        return schedule
    elif type == 'exp':
        import math
        base = kwargs['base']
        max_path = kwargs['max_path']
        max_path_per_iter = kwargs['max_path_per_iter']
        schedule = []
        cnt = 0
        while sum(schedule) < max_path:
            proposed_step = min(
                max_path - sum(schedule), 
                base**cnt, 
                max_path_per_iter)
            cnt += 1
            schedule += [proposed_step]
        return schedule
    else:
        raise ValueError

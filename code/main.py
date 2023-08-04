from typing import Mapping
import os
from shutil import copy2 as copy_file
import subprocess as sp
from tqdm import tqdm
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import LambdaLR
import pydiffvg
import save_svg
from losses import SDSLoss, ToneLoss, ConformalLoss
from diffusers import StableDiffusionPipeline
from xing_loss import xing_loss
import cv2
from config import set_config
from LIVE import live
from utils import (
    check_and_create_dir,
    get_data_augs,
    save_image,
    preprocess,
    learning_rate_decay,
    combine_word,
    create_video)
import wandb
import warnings
warnings.filterwarnings("ignore")

pydiffvg.set_print_timing(False)
gamma = 1.0


def init_shapes(svg_path, trainable: Mapping[str, bool]):

    svg = svg_path
    canvas_width, canvas_height, shapes_init, shape_groups_init = pydiffvg.svg_to_scene(svg)

    parameters = edict()

    # path points
    if trainable.point:
        parameters.point = []
        for path in shapes_init:
            path.points = path.points.cuda()
            path.points *= 4    #补偿分辨率
            path.points.requires_grad = True
            parameters.point.append(path.points)
    # path colors:
    if trainable.color:
        parameters.color = []
        for shape_group in shape_groups_init:
            shape_group.fill_color = shape_group.fill_color.cuda()
            shape_group.fill_color.requires_grad = True
            shape_group.use_even_odd_rule = True
            parameters.color.append(shape_group.fill_color)   #bug:here
    if trainable.bg:
        # parameters.bg = []
        para_bg = torch.tensor([1., 1., 1.], requires_grad=True, device="cuda")
        parameters.bg = [para_bg]
    else:
        para_bg = torch.tensor([1.,1.,1.],requires_grad=False,device="cuda")
        
    if trainable.stroke_width:
        parameters.stroke_width = []
        for path in shapes_init:
            path.stroke_width = path.stroke_width.cuda()
            path.stroke_width.requires_grad = True
            parameters.stroke_width.append(path.stroke_width)
            
    if trainable.stroke_color:
        parameters.stroke_color = []
        for shape_group in shape_groups_init:
            shape_group.stroke_color = shape_group.stroke_color.cuda()
            shape_group.stroke_color.requires_grad = True
            parameters.stroke_color.append(shape_group.stroke_color)

    return shapes_init, shape_groups_init, para_bg,parameters

def select_best_img(init_png_dir,rejection_size,prompt):#从初始化的一批图片中选出语义最一致的
    from PIL import Image
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    images = [preprocess(Image.open(os.path.join(init_png_dir,f"{i}.png"))).unsqueeze(0) for i in range(rejection_size)] #bug here
    text = tokenizer(prompt)
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = torch.stack([model.encode_image(image).squeeze(0) for image in images])
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * text_features @ image_features.T).softmax(dim=-1)
    return os.path.join(init_png_dir,f"{int(torch.argmax(input=text_probs,dim=1,keepdim=False)[0])}.png")

if __name__ == "__main__":
    
    cfg = set_config()

    # use GPU if available
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    device = pydiffvg.get_device()

    # print("preprocessing")
    # preprocess(cfg.font, cfg.word, cfg.optimized_letter, cfg.level_of_cc)
    cfg.render_size = 512 # 仅供测试用
    pipe = StableDiffusionPipeline.from_pretrained(cfg.diffusion.model, torch_dtype=torch.float16,use_auth_token=cfg.token,local_files_only=True)
    pipe = pipe.to(device)
    if cfg.use_svg_local == False or os.path.isfile(cfg.target) == False:
        png_origin_path = os.path.join(cfg.experiment_dir,'init_png',f"{cfg.filename}_origin.png")    
        if cfg.use_img_local == False or os.path.isfile(png_origin_path) == False:
            Rejection_size = 20
            for i in range(Rejection_size):
                print(f"#{i}:")
                SD_image = pipe(prompt=cfg.caption,num_inference_steps=100).images[0]    
                filename = os.path.join(cfg.experiment_dir,"init_png",f"{i}.png")
                check_and_create_dir(filename)
                SD_image.save(filename)
            selected_img = select_best_img(os.path.join(cfg.experiment_dir,'init_png'),Rejection_size,cfg.caption)
            check_and_create_dir(png_origin_path)
            copy_file(selected_img,png_origin_path)
        # 这里需要把文件复制到png_origin_path
        SD_image_compress = cv2.resize(cv2.imread(png_origin_path),(128,128))
        png_path = os.path.join(cfg.experiment_dir,'init_png',f"{cfg.filename}.png")
        cv2.imwrite(png_path,SD_image_compress) #SD:生成图像
        live(cfg_arg=cfg)#LIVE:转成矢量图

    if cfg.loss.use_sds_loss:
        sds_loss = SDSLoss(cfg,device,pipe)

    h, w = cfg.render_size, cfg.render_size

    data_augs = get_data_augs(cfg.cut_size)

    render = pydiffvg.RenderFunction.apply

    # initialize shape
    print('initializing shape')
    shapes, shape_groups, para_bg, parameters = init_shapes(svg_path=cfg.target, trainable=cfg.trainable)
    # filename = "test/test.svg"
    # check_and_create_dir(filename)
    # save_svg.save_svg(filename,w,h,shapes,shape_groups)

    scene_args = pydiffvg.RenderFunction.serialize_scene(w, h, shapes, shape_groups)
    img_init = render(w, h, 2, 2, 0, None, *scene_args)
    # img_init = render(w, h, 2, 2, 0, None, *scene_args)
    img_init = img_init[:, :, 3:4] * img_init[:, :, :3] + \
               torch.ones(img_init.shape[0], img_init.shape[1], 3, device=device) * (1 - img_init[:, :, 3:4])
    img_init = img_init[:, :, :3]
    if cfg.use_wandb:
        plt.imshow(img_init.detach().cpu())
        wandb.log({"init": wandb.Image(plt)}, step=0)
        plt.close()

    if cfg.loss.tone.use_tone_loss:
        tone_loss = ToneLoss(cfg)
        tone_loss.set_image_init(img_init)

    if cfg.save.init:
        print('saving init')
        filename = os.path.join(
            cfg.experiment_dir, "svg-init", "init.svg")
        check_and_create_dir(filename)
        save_svg.save_svg(filename, w, h, shapes, shape_groups)

    num_iter = cfg.num_iter
    pg = [{'params': parameters[ki], 'lr': cfg.lr_base[ki]} for ki in sorted(parameters.keys())] # 这个写法要注意
    optim = torch.optim.Adam(pg, betas=(0.9, 0.9), eps=1e-6)

    if cfg.loss.conformal.use_conformal_loss:
        conformal_loss = ConformalLoss(parameters, device, cfg.optimized_letter, shape_groups)

    lr_lambda = lambda step: learning_rate_decay(step, cfg.lr.lr_init, cfg.lr.lr_final, num_iter,
                                                 lr_delay_steps=cfg.lr.lr_delay_steps,
                                                 lr_delay_mult=cfg.lr.lr_delay_mult) / cfg.lr.lr_init

    scheduler = LambdaLR(optim, lr_lambda=lr_lambda, last_epoch=-1)  # lr.base * lrlambda_f

    print("start training")
    # training loop
    t_range = tqdm(range(num_iter))
    for step in t_range:
        if cfg.use_wandb:
            wandb.log({"learning_rate": optim.param_groups[0]['lr']}, step=step)
        optim.zero_grad()

        # render image
        scene_args = pydiffvg.RenderFunction.serialize_scene(w, h, shapes, shape_groups)
        # img = render(w, h, 2, 2, 0, None, *scene_args)
        img = render(w, h, 2, 2, step, None, *scene_args)

        # compose image with white background
        img = img[:, :, 3:4] * img[:, :, :3] + para_bg * (1 - img[:, :, 3:4])
        img = img[:, :, :3]

        if cfg.save.video and (step % cfg.save.video_frame_freq == 0 or step == num_iter - 1):
            save_image(img, os.path.join(cfg.experiment_dir, "video-png", f"iter{step:04d}.png"), gamma)
            # filename = os.path.join(
            #     cfg.experiment_dir, "video-svg", f"iter{step:04d}.svg")
            # check_and_create_dir(filename)
            # save_svg.save_svg(
            #     filename, w, h, shapes, shape_groups)
            if cfg.use_wandb:
                plt.imshow(img.detach().cpu())
                wandb.log({"img": wandb.Image(plt)}, step=step)
                plt.close()

        x = img.unsqueeze(0).permute(0, 3, 1, 2)  # HWC -> NCHW
        # x_aug = data_augs.forward(x)

        # compute diffusion loss per pixel
        loss = sds_loss(x)
        if cfg.xing_loss_weight is not None and cfg.xing_loss_weight > 0:
            loss_xing = xing_loss(parameters.point)
            loss = loss + loss_xing * cfg.xing_loss_weight
        if cfg.use_wandb:
            wandb.log({"sds_loss": loss.item()}, step=step)

        if cfg.loss.tone.use_tone_loss:
            tone_loss_res = tone_loss(x, step)
            if cfg.use_wandb:
                wandb.log({"dist_loss": tone_loss_res}, step=step)
            loss = loss + tone_loss_res

        if cfg.loss.conformal.use_conformal_loss:
            loss_angles = conformal_loss()
            loss_angles = cfg.loss.conformal.angeles_w * loss_angles
            if cfg.use_wandb:
                wandb.log({"loss_angles": loss_angles}, step=step)
            loss = loss + loss_angles

        t_range.set_postfix({'loss': loss.item()})
        loss.backward()
        optim.step()
        scheduler.step()

    filename = os.path.join(
        cfg.experiment_dir, "output-svg", "output.svg")
    check_and_create_dir(filename)
    save_svg.save_svg(
        filename, w, h, shapes, shape_groups)

    # combine_word(cfg.word, cfg.optimized_letter, cfg.font, cfg.experiment_dir)

    # if cfg.save.image:
    #     filename = os.path.join(
    #         cfg.experiment_dir, "output-png", "output.png")
    #     check_and_create_dir(filename)
    #     imshow = img.detach().cpu()
    #     pydiffvg.imwrite(imshow, filename, gamma=gamma)
    #     if cfg.use_wandb:
    #         plt.imshow(img.detach().cpu())
    #         wandb.log({"img": wandb.Image(plt)}, step=step)
    #         plt.close()

    if cfg.save.video:
        print("saving video")
        filedir = os.path.join(cfg.experiment_dir,"video-png")
        create_video(cfg.num_iter, cfg.experiment_dir, cfg.save.video_frame_freq,w=cfg.render_size,h=cfg.render_size)
        sp.run(["rm","-rf",filedir])

    # if cfg.use_wandb:
    #     wandb.finish()

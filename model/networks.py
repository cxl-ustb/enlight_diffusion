import functools
import logging
import torch
import torch.nn as nn
from torch.nn import init

logger=logging.getLogger('base')

def weights_init_normal(m,std=0.02):
    classname=m.__class__.__name__
    if classname.find('Conv')!=-1:
        init.normal_(m.weight.data,0.,std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear')!=-1:
        init.normal_(m.weight.data,0.,std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d')!=-1:
        init.normal_(m.weight.data,1.0,std)
        init.constant_(m.weight.data,0.)

def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)

def init_weights(net,init_type='kaiming',
                 scale=1,std=0.02):
    logger.info('Initialization method [{:s}]'.format(init_type))
    
    if init_type=='normal':
        weights_init_normal=functools.partial(
          weights_init_normal,std=std  
        )
        net.apply(weights_init_normal)
    elif init_type=='kaiming':
        weights_init_kaiming=functools.partial(
            weights_init_kaiming,scale=scale
        )
        net.apply(weights_init_kaiming)
    elif init_type=='orthogonal':
        net.apply(weights_init_orthogonal)
        

def define_G(opt):
    model_opt=opt['model']
    from .ddpm_modules import diffusion,unet
    if ('norm_groups' not in model_opt['unet']) or model_opt['unet']['norm_groups'] is None:
        model_opt['unet']['norm_groups']=32
    
    model=unet.UNet(
        in_channel=model_opt['unet']['in_channel'],
        out_channel=model_opt['unet']['out_channel'],
        norm_groups=model_opt['unet']['norm_groups'],
        inner_channel=model_opt['unet']['inner_channel'],
        channel_mults=model_opt['unet']['channel_multiplier'],
        attn_res=model_opt['unet']['attn_res'],
        res_blocks=model_opt['unet']['res_blocks'],
        dropout=model_opt['unet']['dropout'],
        image_size=model_opt['diffusion']['image_size']
    )
    
    netG=diffusion.GaussianDiffusion(
        model,
        image_size=model_opt['diffusion']['image_size'],
        channels=model_opt['diffusion']['channels'],
        loss_type='l1',
        conditional=model_opt['diffusion']['conditional'],
        schedule_opt=model_opt['beta_schedule']['train']
    )
    
    if opt['phase']=='train':
        init_weights(netG,init_type='orthogonal')
        
    if opt['gpu_ids'] and opt['distributed']:
        assert torch.cuda.is_available()
        netG=nn.DataParallel(netG)
    
    return netG
import os
import torch
from externals.adm_model_definition import EncoderUNetModel
import params

def load_classifier(img_size, device):
    """
    Load pretrained classifier model (ADM) for supported
    img_size: 32 or 28
    """
    classifier_args = dict(
        image_size=img_size,
        classifier_use_fp16=False,
        classifier_width=128,
        classifier_depth=4 if img_size == 32 else 2,
        classifier_attention_resolutions="32,16,8",
        classifier_use_scale_shift_norm=True,
        classifier_resblock_updown=True,
        classifier_pool="attention",
        out_channels=1000,
    )
    classifier_adm = create_encoder_unet(**classifier_args)
    classifier_adm.to(device)
    classifier_adm.load_state_dict(torch.load(params.classifier_mPath))
    classifier_adm.eval()
    return classifier_adm

def load_discriminator(dis_path, model_type, in_size, in_channels, device, condition=None, eval=False):
    """
    Load pretrained discriminator model (U-Net?) for supported
    model_type: 'pretrained' or 'own'
    """
    discriminator_args = dict(
        image_size=in_size,
        classifier_use_fp16=False,
        classifier_width=128,
        classifier_depth=2,
        classifier_attention_resolutions="32,16,8",
        classifier_use_scale_shift_norm=True,
        classifier_resblock_updown=True,
        classifier_pool="attention",
        out_channels=1,
        in_channels = in_channels,
        condition=condition,
    )
    
    if model_type == 'pretrained':      # discriminator of paper
        discriminator_model = create_encoder_unet(**discriminator_args)
        discriminator_model.to(device)
        discriminator_model.load_state_dict(torch.load(dis_path))
    elif model_type == 'own':           # own discriminator
        discriminator_model = create_encoder_unet(**discriminator_args)
        discriminator_model.to(device)
    else:
        print(f'Error: model_type {model_type} not supported')
    
    if eval:
        discriminator_model.eval()
    
    return discriminator_model

def create_encoder_unet(
    image_size,
    classifier_use_fp16,
    classifier_width,
    classifier_depth,
    classifier_attention_resolutions,
    classifier_use_scale_shift_norm,
    classifier_resblock_updown,
    classifier_pool,
    out_channels,
    in_channels = 3,
    condition=None
    ):
    """
    Create a unet model with the given arguments.
    """
    if image_size == 32:    # CIFAR-10 --> output (B,8,8,512)
        channel_mult = (1, 2, 4)
    # elif image_size == 28:  # MNIST --> output (B,14,14,512), not supported by pretrained model
    #     channel_mult = (1, 2)
    # elif image_size == 14:  # discriminator MNIST --> output (B,1), not supported by pretrained model
    #     channel_mult = (1,)
    elif image_size == 8:   # discriminator --> output (B,1)
        channel_mult = (1,)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    for res in classifier_attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return EncoderUNetModel(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=classifier_width,
        out_channels=out_channels,
        num_res_blocks=classifier_depth,
        attention_resolutions=tuple(attention_ds),
        channel_mult=channel_mult,
        use_fp16=classifier_use_fp16,
        num_head_channels=64,
        use_scale_shift_norm=classifier_use_scale_shift_norm,
        resblock_updown=classifier_resblock_updown,
        pool=classifier_pool,
        condition=condition,
    )


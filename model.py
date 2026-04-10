'''
This code is based on https://github.com/okankop/Efficient-3DCNNs
'''

import os

from torch import nn
import torch
from models import multimodalcnn

def generate_model(opt, audio_input_channels=255, path=None):
    if opt.model == 'multimodalcnn':   
        model = multimodalcnn.MultiModalCNN(
            num_classes=opt.n_classes,
            fusion=opt.fusion,
            seq_length=opt.sample_duration,
            pretr_ef=opt.pretrain_path,           # visual pretrain (existing)
            num_heads=opt.num_heads,
            audio_input_channels=audio_input_channels,
        )
        model = model.to(opt.device)
        model = nn.DataParallel(model, device_ids=None)

        if opt.pretrain_audio_path != 'None':
            _load_pretrained_audio(model, path)
            parameters = _get_parameter_groups(model, opt)
            return model, parameters
    elif opt.model == 'audio':
        from models.multimodalcnn import AudioCNNPool
        model = AudioCNNPool(
            num_classes=opt.n_classes,
            audio_input_channels=audio_input_channels,
        )
        model = model.to(opt.device)
        model = nn.DataParallel(model, device_ids=None)

    return model, model.parameters()

def _load_pretrained_audio(model, path):
    """
    Load AudioCNNPool weights from a standalone audio pretraining checkpoint
    into the audio_model sub-module of MultiModalCNN.
    """
    if not os.path.exists(path):
        print(f'[Pretrain] WARNING: audio checkpoint not found at {path}')
        return
    
    checkpoint = torch.load(path, map_location='cpu')
    pretrained_dict = checkpoint['state_dict']
    
    # Strip 'module.' prefix if saved with DataParallel
    pretrained_dict = {
        k.replace('module.', ''): v
        for k, v in pretrained_dict.items()
    }
    
    # Remap keys: top-level AudioCNNPool keys → audio_model.* keys in MultiModalCNN
    # e.g. 'conv1d_0.0.weight' → 'audio_model.conv1d_0.0.weight'
    remapped = {}
    model_audio_keys = set(
        k for k in model.state_dict().keys()
        if k.startswith('audio_model.')
    )
    
    for k, v in pretrained_dict.items():
        candidate = f'audio_model.{k}'
        if candidate in model_audio_keys:
            remapped[candidate] = v
        else:
            print(f'  [Pretrain] Skipping key not found in fusion model: {k}')
    
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    
    loaded = len(remapped) - len(missing)
    print(f'[Pretrain] Loaded {loaded}/{len(model_audio_keys)} '
          f'audio encoder weights from {path}')
    if missing:
        print(f'  Missing  : {missing[:5]}{"..." if len(missing)>5 else ""}')
    if unexpected:
        print(f'  Unexpected: {unexpected[:5]}')


def _get_parameter_groups(model, opt):
    """
    Differential learning rates:
      - Pretrained audio encoder  → lr * audio_lr_multiplier (default 0.1)
      - Pretrained visual encoder → lr * 0.1  (same logic as before)
      - New fusion layers         → lr * 1.0  (full LR)
    """
    audio_encoder_params  = []
    visual_encoder_params = []
    fusion_params         = []
    # print(model)
    audio_encoder_names = {n for n, _ in model.audio_model.named_parameters()}
    visual_encoder_names = {n for n, _ in model.visual_model.named_parameters()}

    for name, param in model.named_parameters():
        # Strip sub-module prefix to match
        local_name = name.replace('audio_model.', '').replace('visual_model.', '')
        if name.startswith('audio_model.'):
            audio_encoder_params.append(param)
        elif name.startswith('visual_model.'):
            visual_encoder_params.append(param)
        else:
            fusion_params.append(param)

    groups = [
        {'params': fusion_params,
         'lr': opt.learning_rate},
        {'params': audio_encoder_params,
         'lr': opt.learning_rate * opt.audio_lr_multiplier},
        {'params': visual_encoder_params,
         'lr': opt.learning_rate * 0.1},
    ]
    return groups

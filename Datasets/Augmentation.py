'''
Author: Xiaohan Chen
Email: cxh_bb@outlook.com
'''

import numpy as np
import torch
import torchaudio
from typing import Optional

def probability():
    '''
    return a probability
    '''
    return np.random.random()


def FFT(signal: torch.tensor):
    '''
    NOTE: input waveform signal
    '''
    assert len(signal.size()) == 2
    signal = torch.fft.fft(signal)
    signal = 2 * torch.abs(signal) / len(signal)
    signal = signal[:,:int(signal.size(-1) / 2)]
    signal[:,0] = 0.
    return signal

def ToTensor(signal):
    '''
    Numpy to Tensor
    '''
    return torch.from_numpy(signal).float()

def AddGaussianSNR(signal: torch.tensor):
    min_snr = 3,  # (int): minimum signal-to-noise ration in dB
    max_snr = 30, # (int): maximum signal-to-noise ration in dB
    signal_length = signal.size(-1)
    device = signal.device

    snr = np.random.randint(min_snr, max_snr, dtype=int)

    clear_rms = torch.sqrt(torch.mean(torch.square(signal)))
    a = float(snr) / 20
    noise_rms = clear_rms / (10**a)
    noise = torch.normal(0.0, noise_rms, size=(signal_length,))

    return signal + noise.to(device)

def Shift(signal: torch.tensor):
    shift_factor = np.random.uniform(0,0.3)
    shift_length = round(signal.size(-1) * shift_factor)
    num_places_to_shift = round(np.random.uniform(-shift_length, shift_length))
    shifted_signal = torch.roll(signal, num_places_to_shift, dims=-1)
    return shifted_signal

def TimeMask(signal: torch.tensor):
    signal_length = signal.size(-1)
    device = signal.device
    signal_copy = signal.clone()
    fill_value = 0.

    mask_factor = np.random.uniform(0.1,0.45)
    max_mask_length = int(signal_length * mask_factor)  # maximum mask band length
    mask_length = np.random.randint(max_mask_length) # randomly choose a mask band length
    mask_start = np.random.randint(0, signal_length) # randomly choose a mask band start point

    while mask_start + mask_length > max_mask_length:
        mask_start = np.random.randint(0, signal_length)
    mask = torch.arange(signal_length, device=device)
    mask = (mask >= mask_start) & (mask < mask_start + mask_length)
    
    signal_copy = signal_copy.masked_fill(mask, fill_value)
    
    return signal_copy

def Fade(signal: torch.tensor):
    signal_length = signal.size(-1)
    fade_in_length = int(0.3 * signal_length)
    fade_out_length = int(0.3 * signal_length)

    device = signal.device
    signal_copy = signal.clone()

    # fade in
    if probability() > 0.5:
        fade_in = torch.linspace(0,1,fade_in_length,device=device)
        fade_in = torch.log10(0.1 + fade_in) + 1 # logarithmic
        ones = torch.ones(signal_length - fade_in_length, device=device)
        fade_in_mask = torch.cat((fade_in, ones))
        signal_copy *= fade_in_mask
    
    # fade out
    if probability() > 0.5:
        fade_out = torch.linspace(1,0,fade_out_length,device=device)
        fade_out = torch.log10(0.1 + fade_out) + 1 # logarithmic
        ones = torch.ones(signal_length - fade_out_length, device=device)
        fade_out_mask = torch.cat((ones, fade_out))
        signal_copy *= fade_out_mask
    
    return signal_copy

def Gain(signal: torch.tensor):
    gain_min = 0.5
    gain_max = 1.5
    gain_factor = np.random.uniform(gain_min, gain_max)

    signal_copy = signal.clone()

    return signal_copy * gain_factor

def Flip(signal: torch.tensor):
    if len(signal.size()) == 1:
        signal_copy = torch.flip(signal, dims=[0])
    elif len(signal.size()) == 2:
        signal_copy = torch.flip(signal, dims=[1])
    else:
        raise Exception(f"{signal.size()} dimentinal signal time masking is not implemented, "
                        "please try 1 or 2 dimentional signal.")
    
    return signal_copy

def PolarityInversion(signal: torch.tensor):
    '''
    Multiply the signal by -1.
    '''
    signal_copy = signal.clone()
    return - signal_copy

def random_waveform_transforms(signal: torch.tensor):
    '''
    Random waveform signal transformation
    '''

    # 80% probability to add Gaussian noise
    if probability() > 0.2:
        signal = AddGaussianSNR(signal)

    # 30% probability to shift the signal
    if probability() > 0.7:
        signal = Shift(signal)

    # 50% probability to mask time
    if probability() > 0.5:
        signal = TimeMask(signal)
    
    # 50% probability to fade
    if probability() > 0.5:
        signal = Fade(signal)

    # 30% probability to gain
    if probability() > 0.7:
        signal = Gain(signal)

    # 20% probability to horizontal flip
    if probability() > 0.8:
        signal = Flip(signal)
    
    # 20% probability to vertical flip
    if probability() > 0.8:
        signal = PolarityInversion(signal)

    return signal

def frequency_transforms(signal: torch.tensor):
    '''
    NOTE: input waveform signal.
    '''
    signal = FFT(signal)
    
    return signal
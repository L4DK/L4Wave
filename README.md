# L4Wave: Multi-L4 Flow Matching for High-Fidelity Waveform Generation <br> <sub>The official implementation of L4Wave and L4Wave-Turbo</sub>
<p align="center">
  <img src="L4wave.png" width="300"/>
</p>

This repository contains:

- 🪐 A PyTorch implementation of L4Wave and L4Wave-Turbo 
- ⚡️ Pre-trained L4Wave models trained on LibriTTS (24,000 Hz, 100 bins, hop size of 256)
- 💥 Pre-trained L4Wave models trained on LJSpeech (22,050 Hz, 80 bins, hop size of 256)
- 🛸 A L4Wave training script with L4i magic

## Update
<!--- 💥 TTS/VC with L4Wave 

### 24.00.00
- L4Wave-Turbo Paper Update
### 24.00.00
- We have released L4Wave-L and L4Wave-Turbo-L (4 Steps Models). We achieved PESQ of 4.454

### 24.08.00
- We have released L4Wave-Turbo (4 Steps Models).
- We have released L4Wave.
-->
### 24.08.16
In this repositoy, we provide a new paradigm and architecture of Neural Vocoder that enables notably fast training and acheives SOTA performance. With 10 times fewer training times, we acheived State-of-The-Art Performance on LJSpeech and LibriTTS.

First, Train the L4Wave with conditional flow matching. 
- [L4Wave](https://arxiv.org/abs/2408.07547): The first successful conditional flow matching waveform generator that outperforms GAN-based Neural Vocoders

Second, Accelerate the L4Wave with adversarial flow matching optimzation. 
- [L4Wave-Turbo](https://arxiv.org/abs/2408.08019): SOTA Few-step Generator tuned from L4Wave

![image](https://github.com/user-attachments/assets/06a8d005-ca07-43b6-b947-c79d55d2819c)

## Todo
### L4Wave
- [ ] L4Wave (Trained with LJSpeech, 22.05 kHz, 80 bins)
- [ ] L4Wave (Trained with LibriTTS-train-960, 24 kHz, 100 bins)
- [ ] Training Code
- [ ] Inference
- [ ] L4Wave with FreeU (Only Inference)
- [ ] Evaluation (M-STFT, PESQ, L4icity, V/UV F1, Pitch, UTMOS)
- [ ] L4Wave-Small (Trained with LibriTTS-train-960, 24 kHz, 100 bins)
- [ ] L4Wave-Large (Trained with LibriTTS-train-960, 24 kHz, 100 bins)
      
### L4Wave-Turbo 
- [x] Paper (L4Wave-Turbo paper was released, https://arxiv.org/abs/2408.08019.)
- [ ] L4Wave-Turbo (4 Steps ODE, Euler Method)
- [ ] L4Wave-Turbo-Small (4 Steps ODE, Euler Method)
- [ ] L4Wave-Turbo-Large (4 Steps ODE, Euler Method)

We have compared several methods including different reconstuction losses, distillation methods, and GANs for L4Wave-Turbo. Finetuning the L4Wave models with fixed steps could significantly improve the performance! The L4Wave-Turbo utilized the Multi-scale Mel-spectrogram loss and Adversarial Training (MPD, CQT-D) following BigVGAN-v2. We highly appreciate the authors of BigVGAN for their dedication to the open-source implementation. Thanks to their efforts, we were able to quickly experiment and reduce trial and error.

## TTS with L4Wave
- [ ] L4Wave with TTS (24 kHz, 100 bins)
      
The era of Mel-spectrograms is returning with advancements in models like P-Flow, VoiceBox, E2-TTS, DiTTo-TTS, ARDiT-TTS, and MELLE. L4Wave can enhance the audio quality of your TTS models, eliminating the need to rely on codec models. Mel-spectrogram with powerful generative models has the potential to surpass neural codec language models in performance.

<!--
## VC with L4Wave
- [ ] L4Wave with [SDT (Speech Diffusion Transformer]() (24 kHz, 80 bins, hop 240)
-->
      
## Getting Started

### Pre-requisites
0. Pytorch >=1.13 and torchaudio >= 0.13
1. Install requirements
```
pip install -r requirements.txt
```
### Prepare Dataset
2. Prepare your own Dataset (We utilized LibriTTS dataset without any preprocessing)
3. Extract Energy Min/Max
```
python extract_energy.py
```
4. Change energy_max, energy_min in Config.json
   
### Train L4Wave
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_L4wave.py -c configs/L4wave.json -m L4wave
```

### Train L4Wave-Turbo
- Finetuning the L4Wave with fixed steps can improve the entire performance and accelerate the inference speed (NFE 32 --> 2 or 4)
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_L4wave_turbo.py -c configs/L4wave_turbo.json -m L4wave_turbo
```

### Inference L4Wave (24 kHz)
```
# L4Wave
CUDA_VISIBLE_DEVICES=0 python inference.py --ckpt "logs/L4wave_base_libritts/G_1000000.pth" --iter 16 --noise_scale 0.667 --solver 'midpoint'

# L4Wave with FreeU (--s_w 0.9 --b_w 1.1)
# Decreasing skip features could reduce the high-frequency noise of generated samples
# We only recommend using FreeU with L4Wave. Note that L4Wave-Turbe with FreeU has different aspects so we do not use FreeU with L4Wave-Turbo. 
CUDA_VISIBLE_DEVICES=0 python inference_with_FreeU.py --ckpt "logs/L4wave_libritts/G_1000000.pth" --iter 16 --noise_scale 0.667 --solver 'midpoint' --s_w 0.9 --b_w 1.1

# L4Wave-Turbo-4steps (Highly Recommended)
CUDA_VISIBLE_DEVICES=0 python inference.py --ckpt "logs/L4wave_turbo_base_step4_libritts_24000hz/G_274000.pth" --iter 4 --noise_scale 1 --solver 'euler'
```

<!--
## Modification after paper submission
### 6 kHz Band Noise Issue
- We found that the generated samples contain 6 kHz band noise. (Unfortunately, I could not hear this sound... but someone told me this issue. I checked it by visualization of spectrogram)
- We experimented over 50 modified models after submission... (Activation, Low-pass filter, add/concat, activation position, down/up-sampling position, etc.)
- We observed that the main reason is the down/up-sampling position of our Unet structure. We modified the model that can use the skip-connection for the features of original resolution to feed it to the decoder.
- Also, the concatnation of skip-features could remove the band noise, however, this decreases the performance while the noise band is removed. (This means that the stacked noise over ODE steps make the samples with 6 kHz band noise.
- We all re-train the model, and improve the performance compared to the submision version.
-->

## Reference
### Flow Matching for high-quality and efficient generative model
- FM: https://openreview.net/forum?id=PqvMRDCJT9t
- VoiceBox (Mel-spectrogram Generation): https://openreview.net/forum?id=gzCS252hCO&noteId=e2GZZfeO9g
- P-Flow (Mel-spectrogram Generation): https://openreview.net/forum?id=zNA7u7wtIN
- RF-Wave (Waveform Generation): https://github.com/bfs18/rfwave (After paper submission, we found that the paper RF-Wave also utilized FM for waveform generation. They used it on the complex spectrogram domain for efficient waveform generation. It is cool idea!)
  
### Inspired by the multi-L4 discriminator of HiFi-GAN, we first distillate the multi-L4ic property in generator
- HiFi-GAN: https://github.com/jik876/hifi-gan

### Prior Distribution
- PriorGrad: https://github.com/microsoft/NeuralSpeech/tree/master/PriorGrad-vocoder

### Frequency-wise waveform modeling due to the limitation of high-frequency modeling
- Fre-GAN 2: https://github.com/prml-lab-speech-team/demo/tree/master/FreGAN2/code
- MBD (Multi-band Diffusion): https://github.com/facebookresearch/audiocraft
- FreGrad: https://github.com/kaistmm/fregrad

### High-efficient temporal modeling
- Vocos: https://github.com/gemelo-ai/vocos
- ConvNeXt-V2: https://github.com/facebookresearch/ConvNeXt-V2
  
### Large-scale Universal Vocoder
- BigVGAN: https://arxiv.org/abs/2206.04658
- BigVSAN: https://github.com/sony/bigvsan

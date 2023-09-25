# MOFO: MOtion FOcused Self-Supervision for Video Understanding [[Arxiv]](https://arxiv.org/abs/2308.12447)

![MOFO Framework](figs/MF.png)  

> [**MOFO: MOtion FOcused Self-Supervision for Video Understanding**](https://arxiv.org/abs/2308.12447)<br>
> [Mona Ahmadian](https://github.com/moohnai), [Frank Guerin](), [Andrew Gilbert]()
<br>University of Surrey, Guildford, UK

## 📰 News
**[2023.8.28]**  Code of the Automatic motion detection, MOFO self-supervision and MOFO finetuning are available now! <br>


## ✨ Highlights

### 🔥 MOFO: MOtion FOcused Self-Supervision for Video Understanding 
**MOFO (MOtion FOcused)** is a novel Self-supervised learning (SSL) method, for focusing representation learning on the motion area of a video, for action recognition and provides evidence that such a motion-focused technique could be effective in exploring motion information for enhancing **motion-aware self-supervised video action recognition**. MOFO automatically detects motion areas in videos and uses these to guide the self-supervision task. We use tube masking strategy and masked autoencoder which randomly masks out a high proportion of the input sequence (90%); we force a fixed percentage of the tubes (75\%) inside the motion area to be masked and the remainder from outside. We further incorporate motion information into the finetuning step to emphasise motion in the downstream task. 



### ⚡️ A Motion-aware Baseline in SSVP

MOFO can serve as **a motion-aware baseline** for future research in self-supervised video pre-training and public code will guide many research directions.

MOFO's contributions are as follows:
  
- The Automatic motion area detection using motion maps driven by optical flows, but invariant to camera motion.

-  motion-aware SSL approach, which focuses masking on the motion area in the video, using our proposed automatic motion detection algorithm.

- A motion-focused finetuning technique to further intensify the focus on the motion area for the action recognition task.


###  High performance

MOFO works well for video datasets of different scales and can achieve **75.5%** on Something-Something V2, **74.2%**, **68.1%**, **54.5%** on Epic-Kitchens verb, noun and action repectively, only using **ViT-Base** backbones while **doesn't need** any extra data.

## 🚀 Main Results
MOFO* is pretrained by our MOFO SSL and uses non-MOFO finetuning.

MOFO** This is our result with pretraining on non-MOFO SSL and has MOFO finetuning.

MOFO† denotes the MOFO SSL and MOFO finetuning.
### ✨ Something-Something V2

|  Method  | Extra Data | Backbone | Resolution | #Frames x Clips x Crops | Top-1 | Top-5 |
| :------: | :--------: | :------: | :--------: | :---------------------: | :---: | :---: |
| MOFO*   |  ***no***  |  ViT-B   |  224x224   |         16x2x3          | 72.7   | 94.2  |
| MOFO**   |  ***no***  |  ViT-B   |  224x224   |         16x2x3          | 74.7  | 95.0  |
| MOFO†   |  ***no***  |  ViT-B   |  224x224   |         16x2x3          | 75.5  | 95.3  |


### ✨ Epic-Kitchens-verb

|  Method  | Extra Data | Backbone | Resolution | #Frames x Clips x Crops | Verb Top-1 | Noun Top-1 | Action Top-1 |
| :------: | :--------: | :------: | :--------: | :---------------------: | :---: | :---: | :---: |
| MOFO*   |  ***no***  |  ViT-B   |  224x224   |         16x2x3          | 73.0   | 67.1 | 54.1|
| MOFO**   |  ***no***  |  ViT-B   |  224x224   |         16x2x3          | 74.0 | 68.0  | 54.5 |
| MOFO†   |  ***no***  |  ViT-B   |  224x224   |         16x2x3          | 74.2  | 68.1 | 54.5 |

## Pretrained Weights

|  Method  | Data | Backbone | Resolution | Training Steps | Link |
| :------: | :--------: | :------: | :--------: | :---------------------: | :---: |
| MOFO*   |  SSV2  |  ViT-B   |  224x224   |         250         | [Link](https://drive.google.com/file/d/1OQ4CkQsf6DqEQOTec95MoiRQhZz_hKvx/view?usp=sharing) |
| MOFO*   |  EPIC KITCHENS  |  ViT-B   |  224x224   |         250          | [Link](https://drive.google.com/file/d/1_PD1HmrPStm5I3-viz_d5Sx5955rd-Ao/view?usp=sharing)  |
| MOFO*   |  EPIC KITCHENS  |  ViT-B   |  224x224   |         800          | [Link](https://drive.google.com/file/d/1krXW0T5UTiSfF0ZZgwzHgHxPzZIUpWpM/view?usp=sharing) |

##  Setup
We used Python 3.8.13 and PyTorch 1.12.0 to train and test our models.

You can download and install (or update to) the latest release of MOFO with the following command:
```
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

DS_BUILD_OPS=1 pip install deepspeed

pip install timm==0.4.12

conda install -c conda-forge tensorboardx

pip install decord

conda install -c conda-forge einops

pip install opencv-python

pip install scipy

pip install pandas

conda install -c conda-forge mpi4py

pip install -U albumentations
```

##  Data Preparation

Please follow the instructions in [DATASET.md](DATASET.md) for data preparation.

##  Pre-training

The pre-training instruction is in [PRETRAIN_BB.md](PRETRAIN.md).

##  Fine-tuning with pre-trained models

The fine-tuning instruction is in [FINETUNE_BB.md](FINETUNE.md).


##  Contact 

Mona Ahmadian: m.ahmadian@surrey.ac.uk

## Acknowledgements
This project is built upon [VideoMAE](https://github.com/MCG-NJU/VideoMAE)

## ✏️ Citation

If you think this project is helpful, please feel free to leave a star⭐️ and cite our paper.

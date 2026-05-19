**Fork Notice:** This is a fork of [StyleTTS2](https://github.com/yl4579/StyleTTS2) with emphasis on modern dependencies (Blackwell GPU support) and improvements to the fine-tuning experience. See the original repository for the original README.

---

# Fine-Tuning A Voice on StyleTTS2

## Overview

This covers setting up and configuring StyleTTS2 Finetuner for fine-tuning a single speaker's voice using the LibriTTS pretrained model.

## Step 1: Clone The Repo

```bash
cd ~/ai/Trainers
git clone https://github.com/yushan777/StyleTTS2-Finetuner.git
cd StyleTTS2-Finetuner
```

## Step 2: Create the Venv

```bash
python3 -m venv venv
source venv/bin/activate
```

## Step 3: Install Dependencies

The venv must have PyTorch installed with a CUDA version that matches your GPU architecture. **Blackwell GPUs (e.g. RTX PRO 6000, sm_120) require CUDA 12.8 or later.**

```bash
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt

pip install tensorboard

pip install pandas

pip install phonemizer  # required for phonemizing training lists and demoing checkpoints
```

## Step 4: Create Additional Directories

```bash
mkdir -p Models/LibriTTS
mkdir -p output
mkdir -p training_datasets
```

## Step 5: Download Pretrained Model

Download the LibriTTS checkpoint from Hugging Face:

**URL:** https://huggingface.co/yl4579/StyleTTS2-LibriTTS/tree/main

Download `epochs_2nd_00020.pth` and place it at:

```
./Models/LibriTTS/epochs_2nd_00020.pth
```

---

## Dataset Organisation

Keep everything self-contained under the project:

```
StyleTTS2-Finetuner/
└── training_datasets/
    └── voice-45/
        ├── audio_normalized_24khz/   ← wav files
        ├── voice-45_train_list.txt
        └── voice-45_val_list.txt
```

List files use a 3-column pipe-separated format:

```
filename.wav|phoneme transcription|speaker_id
```

The transcription column must contain IPA phonemes, not plain English text. The model's text encoder expects phoneme tokens — the same representation produced by the espeak phonemizer at inference time. Training on plain text causes a mismatch and results in garbled output.

Example:

```
LJ027-0028.wav|stɹʌktʃɚz ɑɹ ɔɹɡənz ɑɹ moʊst ɔfən faɪnd ɪntɜɹnəli .|0
LJ015-0030.wav|ðə bæk hɛdbɪŋ kəndʌktəd ɑn fɔls pɹɪnsɪpəlz ;|0
```

`speaker_id` is always `0` for single-voice training.

If your lists contain a path prefix before the WAV filename, remove the path and leave only the filename.

### Phonemizing Your Lists

Run `phonemize_lists.py` to convert your plain-text lists to IPA:

```bash
python3 phonemize_lists.py training_datasets/voice-45
```

This produces two additional files suffixed with `_phonemized`:

```
StyleTTS2-Finetuner/
└── training_datasets/
    └── voice-45/
        ├── audio_normalized_24khz/
        ├── voice-45_train_list.txt
        ├── voice-45_train_list_phonemized.txt
        ├── voice-45_val_list.txt
        └── voice-45_val_list_phonemized.txt
```

---

## Config: config_ft.yml

Copy `Configs/config_ft.yml` and rename it (e.g. `config_ft_voice-45.yml`) for your fine-tuning run. Key fields to set:

```yaml
log_dir: "output/voice-45"
save_freq: 5
log_interval: 10
device: "cuda"
epochs: 60           # number of fine-tuning epochs (suitable for ~1 hour of data)
batch_size: 6
max_len: 1800        # maximum number of frames
pretrained_model: "Models/LibriTTS/epochs_2nd_00020.pth"
second_stage_load_pretrained: true  # true if the pre-trained model is for 2nd stage
load_only_params: true              # true to skip loading epoch numbers and optimizer state

F0_path: "Utils/JDC/bst.t7"
ASR_config: "Utils/ASR/config.yml"
ASR_path: "Utils/ASR/epoch_00080.pth"
PLBERT_dir: 'Utils/PLBERT/'

data_params:
  train_data: "training_datasets/voice-45/voice-45_train_list_phonemized.txt"
  val_data: "training_datasets/voice-45/voice-45_val_list_phonemized.txt"
  root_path: "training_datasets/voice-45/audio_normalized_24khz"
  OOD_data: "Data/OOD_texts.txt"
  min_length: 50
```

**Do not change** `multispeaker: true` — even for a single speaker. The LibriTTS pretrained model has a multispeaker architecture; changing this would make the weights incompatible.

Checkpoints are saved every `save_freq` epochs to `log_dir`.

---

## Training Scripts

StyleTTS2 has three training scripts with distinct purposes:

**`train_first.py` — Stage 1 (from scratch only)**
Trains the acoustic and alignment components: mel spectrogram reconstruction, text aligner, pitch extractor, and style encoder. Uses supervised losses only — no discriminator or adversarial training. Designed for large-dataset, from-scratch training with multi-GPU (DDP/accelerate).

**`train_second.py` — Stage 2 (from scratch only)**
Loads the Stage 1 checkpoint and adds adversarial training: a discriminator and SLM adversarial loss. This produces natural-sounding, high-quality output. Note: DDP is broken for this script; it runs single-GPU with DataParallel only.

**`train_finetune.py` — Fine-tuning (use this)**
Starts from an existing pretrained checkpoint and includes SLM adversarial training. Skips the two-stage from-scratch process entirely. This is the correct script for fine-tuning a new voice on top of the LibriTTS pretrained model.

`train_first.py` and `train_second.py` are only needed if training a brand new StyleTTS2 model from scratch on a large dataset. For voice fine-tuning, use `train_finetune.py` (or `train_finetune_accelerate.py` for multi-GPU).

---

## Running Fine-Tuning

```bash
source venv/bin/activate
python train_finetune.py --config_path Configs/config_ft_voice-45.yml
```

---

## VRAM Usage

- With `batch_size: 6` and the `empty_cache()` fix, VRAM fluctuates between 50–80 GB on RTX PRO 6000 Blackwell.
- Without the fix, VRAM creeps steadily from ~92 GB until OOM due to variable-length audio sequences filling PyTorch's CUDA memory cache.
- Fix: `torch.cuda.empty_cache()` added at the end of each epoch in `train_finetune.py`.

**Batch size guidance (per GPU):**
- The default `batch_size: 8` was designed for 4× A100 80 GB — each GPU handled batch 8 with 80 GB.
- A single 96 GB GPU is close to one A100, so `batch_size: 6` is a safe starting point with headroom.
- The config value is per GPU, not total across all GPUs.

---

## Monitoring with TensorBoard

```bash
tensorboard --logdir output/voice-45
```

Open `http://localhost:6006`. In the **Scalars** tab, filter with:

```
mel_loss|f0_loss|dur_loss|sty_loss
```

| Graph | What it means |
|---|---|
| `train/mel_loss` | Main voice quality indicator — should steadily decrease |
| `train/loss` | Overall training loss |
| `train/f0_loss` | Pitch reconstruction — affects prosody naturalness |
| `train/dur_loss` | Duration prediction — affects speech rhythm |
| `train/sty_loss` | Style reconstruction — captures speaker character |
| `eval/mel_loss` | Validation mel loss — watch for divergence from train |

**Overfitting warning:** With limited data (e.g. 44 minutes), `eval/mel_loss` may stop decreasing while `train/mel_loss` keeps falling. Your best checkpoint is the one just before eval loss started rising — not necessarily the final epoch.

---

## Checkpoint File Size

Fine-tuned checkpoints are significantly larger than the base model because they store more than just weights:

```python
state = {
    'net':       # model weights
    'optimizer': # optimizer state  ← main culprit
    'iters':     # iteration count
    'val_loss':  # validation loss
    'epoch':     # epoch number
}
```

The Adam optimizer stores two extra values per parameter (first and second moment estimates), making the optimizer state roughly 2× the size of the weights alone. Combined, each checkpoint is ~3× the size of the base model file. This is normal and required to resume training. For inference-only use, the optimizer state can be stripped to produce a much smaller file.

---

## Inference

```bash
python3 infer.py --text "Hello, you look nice today!." \
  --ref training_datasets/voice-45/audio_normalized_24khz/PHO_fricative_sh_0002.wav \
  --checkpoint output/voice-45/epoch_2nd_00059.pth \
  --output test5.wav
```

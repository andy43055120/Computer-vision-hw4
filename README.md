# NYCU Computer Vision 2026 HW4
- Student ID: 111550148
- Name: 陳冠達

## Introduction
This task aims to train an image restoration model which is able to handle both rain and snow degradation, which is to recover clean images from degraded inputs while preserving image details and structure.
The core idea of my method is based on PromptIR, which uses learnable prompts to adaptively guide the restoration process for different degradation types. And further improve the baseline model with several training experiments.

## Environment Setup
Required libraries can be installed using:
```base
pip install -r requirements.txt
```

## Usage
### Training
How to train your model.
```bash
python hw4.py
```

### Inference
Run inference using a trained model checkpoint:
```bash
python hw4.py --predict path/to/model.pt
```

## Performance Snapshot
<img width="1465" height="70" alt="image" src="https://github.com/user-attachments/assets/0c5108ad-3c1c-4712-9586-09cae8216ab9" />

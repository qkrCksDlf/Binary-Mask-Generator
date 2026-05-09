## OverViwe
Uses Grounding DINO for bounding boxes and SAM for segmentation to produce a binary mask.

## Setup
```bash
pip install transformers torch torchvision pillow numpy
```

## Usage
```
python bmaskgen.py -i (image_path) -p (object_prompt)
```


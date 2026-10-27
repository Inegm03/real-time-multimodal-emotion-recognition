# Dataset setup

This repository does not redistribute datasets or trained weights.

## WIDER FACE

Used to train the face detector. Download the official training and validation images and annotations from the WIDER FACE project, then convert each bounding box to YOLO format:

```text
class_id x_center y_center width height
```

All coordinates must be normalized by image width and height. The project uses one class: `face`.

## FER-2013

Used for seven-class emotion classification. Arrange the images in Ultralytics classification format:

```text
data/fer2013/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── val/
    └── <same class folders>
```

The main notebook includes conversion logic for the common `fer2013.csv` representation.

## RAVDESS

Used by the optional speech and multimodal evaluation notebook. Download:

- `Audio_Speech_Actors_01-24.zip`
- `Video_Speech_Actors_01-24.zip`

Extract them under `data/ravdess/`. The evaluation maps calm to neutral and uses the four classes supported by the selected HuBERT checkpoint: angry, happy, neutral, and sad.

Review each dataset's original license and terms before use.


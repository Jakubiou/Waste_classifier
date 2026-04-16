# Waste Classifier: Image Recognition for Waste Sorting

A project focused on automatic waste classification using Convolutional Neural Networks (CNN). The system analyzes uploaded images, extracts visual features, and uses an optimized TensorFlow Lite model to determine the correct recycling category.

## Main Features

- Real-time classification for categories: plastic, paper, glass, bio, and mixed waste
- Extraction of visual features (brightness, contrast, saturation, edge density)
- Web interface built with Flask
- Optimized TensorFlow Lite (TFLite) model for low latency and minimal hardware requirements

## Installation and Running

The project can be run either as a standalone application or within a development environment.

### 1. Running via Binary File (.exe)

This option does not require Python libraries installation and is the fastest way to try the application.

1. Navigate to the `bin` folder
2. Run the file `App.exe`
3. After launching, open your web browser and go to: http://localhost:5000

### 2. Running from Development Environment (IDE)

To run the project in an IDE (VS Code, PyCharm, etc.), Python 3.9 or newer is required.

#### Install dependencies

```bash
pip install -r requirements.txt
```

The application will be available at: http://localhost:5000

#### Requirements (Dependencies)

The following libraries are required to run the source code:

- tensorflow (or tflite-runtime)
- flask
- numpy
- pandas
- pillow (PIL)
- scikit-learn

#### Classification Methodology

The application follows color-coded waste sorting standards used in the Czech Republic:

- Plastic: yellow container
- Paper: blue container
- Glass: green container
- Bio waste: brown container
- Mixed waste: black container

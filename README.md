# HOWLish

HOWLish is a pretrained convolutional neuronal network that predicts the presence of wolf howls (*Canis lupus*, Linnaeus 1758) in 8kHz audio data. 

We developed HOWLish by fine-tuning [VGGish](https://github.com/tensorflow/models/tree/master/research/audioset/vggish) to a dataset of 50,137 hours of soundscapes recorded in the Iberia Peninsula, containing 1014 manually labelled howling events. 

HOWLish classifies short audio snippets (0.96 seconds) regarding the presence of wolf howls. For each snippet it predicts the presence (1) or absence (0) of a wolf howl. 

Evaluated on a test set with 5081 *wolf* and 36,198,705 *not-wolf* audio snippets, HOWLish achieved the following performance: 

<div align="center">

| Model  | Accuracy | Precision | Recall | Fall-out | F1-score | F2-score | AUC | PRC |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| HOWLish  | .983  | .00618  | .772  | .0174  | .00508  | .0123  | .939  | .0897  |

<div align="left">

For a detailed description of HOWLish read <ins>add link to publication when published</ins>.

HOWLish v 1.0.0 can be downloaded: 
- [here](https://drive.google.com/file/d/1SdULuhgMdjlN5rLRAPm1dW6M6ASdT6Pp/view?usp=drive_link) for the TensorFlow SavedModel format; 
- [here](https://drive.google.com/file/d/1Sdt5TwN-OteMp7fV7ub9G109d-dSo8du/view?usp=sharing) for the frozen graph format; 

## Detection Pipeline

We developped a detection pipeline to deploy HOWLish to field operations. Its current version (1.0.0) has the followign flow: 

1) Recorded soundscapes (.WAV) get segmented into 0.96s long audio examples and each sample normalized to fall within the range (-1.0, +1.0);
2) HOWlish predicts whether each example is *not-wolf* or *wolf* (continuous prediction value between 0 and 1, respectively);
3) Prediction values get averaged by a moving window of size **W**;
4) Windows with average prediction values higher than a threshold of value **T** are selected;
5) 110 seconds of sound around the retained windows are exported as sound segments potentially containing wolf howls.


<div align="center">

<img width="1705" alt="DetectionPipelineScheme" src="https://github.com/user-attachments/assets/8d4675da-716a-4a64-a66a-f4f0d9b615ce">

<div align="left">


We performed a sensitivity analysis to window size (W) and exclusion threshold (T) on the pipeline’s ability to retrieve howling events from the test set (n = 175 howling events), and found W = 3 and T = 0.9 to be optimal operating conditions for our operations.

### Usage

Here is one way you can deploy HOWLish with pyhton: 

[Youtube guide (MAC OS)](https://www.youtube.com/watch?v=AOrImhGuMBg)    ||   [Youtube guide (Windows)](https://youtu.be/jbUwlCPOsro)

1) Download and install [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install).
2) Download our [latest release](https://github.com/CamposRF/HOWLish/releases), the [frozen graph](https://drive.google.com/file/d/1Sdt5TwN-OteMp7fV7ub9G109d-dSo8du/view?usp=sharing), and the [toy dataset](https://drive.google.com/file/d/1uxuWrNfPz-IgfRJ-XIGDpsLe_9ghIzW6/view?usp=drive_link);
3) Create and activate a conda environment with python installed;
```
conda create -n HOWLish310 python=3.10
```
```
conda activate HOWLish310 
```
4) Install the libraries needed to run our scripts, as specified on the `requirements.txt` file;
```
pip install -r requirements.txt
```
5) Open the `HOWLish_pipeline.py` script;
6) Add path to the frozen model model file on line 19;
```python
path_to_howlish_frozen = r"C:\Users\you\Downloads\v100\HOWLish_fg_100.pb" #example
```
7) Add path to input folder on line 22;
```python
source_folder =  r"C:\Users\you\Downloads\v100\toy_data\input" #example
```
8) Add path to output folder on line 23;
``` python
save_folder = r"C:\Users\you\Downloads\v100\toy_data\output" #example
```
9) Save the `HOWLish_pipeline.py` script;
10) Run the `HOWLish_pipeline.py` script;
```
python HOWLish_pipeline.py
```

The pipeline assumes the .WAV files to be processed are stored in nested subfolders inside the input folder directory: 

```
Input folder/
├─ Campaign 1/
│  ├─ example-A.wav
│  ├─ example-B.wav
├─ Campaign 2/
│  ├─ example-C.wav
│  ├─ example-D.wav
│  ├─ example-E.wav

```
It then creates a mirrored folder structured inside the output folder directory with the results:
```
Output folder/
├─ Campaign 1/
│  ├─ Campaign_1_log.csv  
├─ Campaign 2/
│  ├─ example-D_0000_9999.wav
│  ├─ example-D_0000_9999.wav
│  ├─ example-D_0000_9999.wav
│  ├─ example-D_0000_9999.wav
│  ├─ example-D_0000_9999.wav
│  ├─ Campaign_1_log.csv 

```
The log file documents the time needed to process the files inside a given folder and the settings used. Ouput files are named in the follwign way: 

```
(original name)_(time in seconds when the clip starts)_(prediction value).wav
```

The toy dataset is structured accordigly: in it you will find a input folder with a subfolder with 2 .WAV files collected in the North of Portugal, and an output folder with a subfolder with the expected results for those 2 .WAV files (W = 3 and T = 0.90). 

The pipeline assumes by default that input data has a sampling rate of 8000 Hz. You can adjust the short time Fourier transform on `vggish_params.py`. For mode details on how we handled pre-tranformation of input data read <ins>add link to publication when published</ins>.


## Credits
The detection pipeline makes use of preprocessing scripts from the original [VGGish repository](https://github.com/tensorflow/models/tree/master/research/audioset/vggish), all licensed under Apache License 2.0. We documented all changes to these scripts and included a link to the original version. 

All data used to develop HOWLish was recorded between 2020 and 2024 in the Iberia Peninsula and is deposited, along with the associated manual annotations, in the Natural Sounds Archive of the Museum of Natural History and Sciences, University of Lisbon, Portugal. For access please email the archive curator (paulo-marques@edu.ulisboa.pt) and the collections head (geral@museus.ulisboa.pt).

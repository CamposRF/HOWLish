# HOWLish

HOWLish is a pretrained convolutional neuronal network that predicts the presence of wolf howls (*Canis lupus*, Linnaeus 1758) in 8kHz audio data. 

We developed HOWLish by fine-tuning [VGGish](https://github.com/tensorflow/models/tree/master/research/audioset/vggish) to a dataset of 50,137 hours of soundscapes recorded in the Iberia Peninsula, containing 1014 manually labelled howling events. 

HOWLish classifies short audio snippets - 0.96 seconds of audio represented in 96 x 64 (frames x frequency bands) log-mel spectrograms - regarding the presence of wolf howls. For each snippet it predicts the presence (1) or absence (0) of a wolf howl. 

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

We developped a detection pipeline (currently v1.0.0) to deploy HOWLish to field operations. It has the followign flow: 

1) Observed soundscapes (.WAV) get segmented into 0.96s long audio examples and each sample normalized to fall within the range (-1.0, +1.0);
2) HOWlish predicts whether each example is *not-wolf* or *wolf* (continuous prediction value between 0 and 1, respectively);
3) Prediction values get averaged by a moving window of size **W**;
4) Windows with average prediction values higher than a threshold of value **T** are selected;
5) 110 seconds of sound around the retained windows are exported as sound segments potentially containing wolf howls.


<div align="center">

<img width="1705" alt="DetectionPipelineScheme" src="https://github.com/user-attachments/assets/8d4675da-716a-4a64-a66a-f4f0d9b615ce">

<div align="left">

### Sensitivity to W and T

We performed a sensitivity analysis to window size (W) and exclusion threshold (T) on the pipeline’s ability to retrieve howling events from the test set (n = 175 howling events), and found W = 3 and T = 0.9 to be optimal operating conditions for our operations.

### Usage

To deploy HOWLish using our detection pipeline we suggest downloading the latest release from (LINK). 

A toy dataset can be downloaded [here](https://drive.google.com/file/d/11ouRaRAI_V38n5T4q4Cr8zgeRPUB2jgS/view?usp=drive_link). This dataset includes two .WAV files from passive acoustic monitoring campaigns conducted in the north of Portugal.
The dataset is structured in two folders: input and output, compatible with the way the detection pipeline was coded. The .WAV files are 30 minutes long and were recorded with a sample rate of 8kHz. The current version of the pipeline only works with 8kHz audio data. 


((work in progress))

## Credits
The detection pipeline makes use of preprocessing scripts from teh original [VGGish repository](https://github.com/tensorflow/models/tree/master/research/audioset/vggish), all licensed under Apache License 2.0. We documented all changes to these scripts and included a link to the original version. 

> [!NOTE]
> All data was recorded between 2020 and 2024 in the Iberia Peninsula and is deposited, along with the associated manual annotations, in the Natural Sounds Archive of the Museum of Natural History and Sciences, University of Lisbon, Portugal. For access please email the archive curator (paulo-marques@edu.ulisboa.pt) and the collections head (geral@museus.ulisboa.pt).

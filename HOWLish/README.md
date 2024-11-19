# HOWLish

HOWLish is a pretrained convolutional neuronal network that predicts the presence of wolf howls (*Canis lupus*, Linnaeus 1758) in audio data. 

We developed HOWLish by applying transfer learning from [VGGish](https://github.com/tensorflow/models/tree/master/research/audioset/vggish) to a dataset of 50,137 hours of recorded soundscapes with 1014 manually labelled howling events. 

Here, we provide HOWLish and a tailored pipeline that can be used for field deploymnent. For a detailed description read <ins>add link to publication when published</ins>.

## The model

HOWLish classifies short audio snippets - 0.96 seconds of audio represented in 96 x 64 (frames x frequency bands) log-mel spectrograms - regarding the presence of wolf howls. For each snippet it predicts the presence (1) or absence (0) of a wolf howl.

We preserved VGGish’s original architecture (top included), but added a sigmoid layer as the output layer to match our binary classification task of distinguishing between not-wolf and wolf examples. We adapted the VGGish's original Short-Time Fourier Transform [parameters](https://github.com/tensorflow/models/blob/master/research/audioset/vggish/vggish_params.py) to our 8 kHz sampling frequency data by using a window size of 0.05 seconds (400 samples) and re-dimensioned the frequency axis to a maximum frequency of 2,000 Hz.

We preprocessed our 50,137 hours of recorded soundscapes into a train dataset containing 96,361 *not-wolf* and 17,927 *wolf* examples, a validation dataset containing 41,475,717 *not-wolf* and 2,290 *wolf* examples, and a test dataset containing 36,198,705 *not-wolf* examples and 5,081 *wolf* examples. We randomly undersampled the majority class and used class weights during training to balance both classes. 

### Performance

At a prediction threshold of .5, HOWlish is able to retrieve 77% of the *wolf* examples on the test set with a false positive rate of 1.74%. 

| Model  | Accuracy | Precision | Recall | Fall-out | F1-score | F2-score | AUC | PRC |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| HOWLish  | .983  | .00618  | .772  | .0174  | .00508  | .0123  | .939  | .0897  |

HOWLish v 1.0.0 can be downloaded: 
- [here](https://drive.google.com/file/d/1SdULuhgMdjlN5rLRAPm1dW6M6ASdT6Pp/view?usp=drive_link) for the TensorFlow SavedModel format; 
- [here](https://drive.google.com/file/d/1Sdt5TwN-OteMp7fV7ub9G109d-dSo8du/view?usp=sharing) for the frozen graph format; 

## Deployment

We developped HOWLish with passive acoustic wolf monitoring in mind; or goal was to establish the baseline for free pretrained tools that allow automated detection of wolf howls in large volumes of recorded soudscapes. 

HOWLish functions as a car engine, that still needs a whole set of components before it is able to transform work into movement, in this case classification into wolf howl detection. 
Accordingly, we designed a series of pre- and post- processing rules that allows the conversion of soundscapes.WAV files into 1 minute and 50 seconds clips of sound potentially containing wolf howls - a detection pipeline. 

## Detection pipeline

We ha been deploying HOWLish to field operations through a detection pipeline that takes recorded soundscapes as input and outputs sound segments where it predicts wolf howls to be present. The pipeline has the following flow:

1) Soundscapes get segmented into 0.96s long audio examples and each sample normalized to fall within the range (-1.0, +1.0);
2) HOWlish predicts whether each example is *not-wolf* or *wolf* (prediction value between 0 and 1, respectively);
3) Prediction values get averaged by a moving window of size **W**;
4) Windows with average prediction values lower than a threshold of value **T** are excluded;
5) 110 seconds of sound around the retained windows are exported as sound segments potentially containing wolf howls.

![image](https://github.com/user-attachments/assets/285ba314-16a1-4f3f-91f3-55c323841fd9)


In a real-world deployment setting, HOWLish was able to retrieve 81.3% of the howling events we detected through manual classification. Automated inference using HOWLish offered 22-fold reduction in the volume of data that needed to be manually processed by an operator, and a 15-fold reduction in operator time, when compared to manual annotation.

### Credits
The detection pipeline makes use of preprocessing scripts from teh original [VGGish repository](https://github.com/tensorflow/models/tree/master/research/audioset/vggish), all licensed under Apache License 2.0. We documented all changes to these scripts and included a link to the original version. 


> [!NOTE]
> All data was recorded between 2020 and 2024 in the Iberia Peninsula and is deposited, along with the associated manual annotations, in the Natural Sounds Archive of the Museum of Natural History and Sciences, University of Lisbon, Portugal. For access please contact the archive curator (paulo-marques@edu.ulisboa.pt) and to the collections head (geral@museus.ulisboa.pt).

# Howlish: a CNN for automated wolf howl detection
peer reviewed research paper here <ins>add link</ins>

##Usage

We developed HOWLish to widen the logistic bottleneck of detecting wolf howls in recorded soundcapes. 

Passive acoustic monitoring of wolves naturally warrants the deployment of large numbers of ARUs that soon become a logistic nightmare to handle. We thus set out to provide the scietific and conservation community with a free and already trained tool that could facilitate the detection of howls in recorded data. Although currently locked behing the user's ability to deploy a CNN, HOWLish is trained, free, and ready to be deployed in passive acoustic monitoring operations - or any other operation looking to detect wolf howls.  

We also made the detection pipeline - a series of pre and post processing rules - we have been using since 2023 available to anyone who whishes to use it. 

HOWLish classifies short audio snippets - 0.96 seconds of audio represented in log mel spectrograms - regarding the presence of wolf howls. For each snippet it predicts the presence (1) or absence (0) of a wolf howl.  
Its target deployment enviornment is the detection of wolf howling events in recorded soundscapes, specificly in the context of passive acoustic wolf monitoring protocols. 

We developed HOWLish by applying transfer learning from [VGGish](https://github.com/tensorflow/models/tree/master/research/audioset/vggish) to a dataset of 50,137 hours of soundscapes with 1014 manually labelled howling events. 
For a detailed description of the development process we suggest reading <ins>add link to publication when published</ins>.

At a prediction threshold of .5, HOWlish is able to retrieve 77% of the *wolf* examples of the test set (36,198,705 *not-wolf* examples and 5,081 *wolf* examples) with a false positive rate of 1.74%. In a real-world deployment setting, HOWLish was able to retrieve 81.3% of the howling events we detected through manual classification. Automated inference using HOWLish offered 22-fold reduction in the volume of data that needed to be manually processed by an operator, and a 15-fold reduction in operator time, when compared to manual annotation.

| Model  | Accuracy | Precision | Recall | Fall-out | F1-score | F2-score | AUC | PRC |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| HOWLish  | .983  | .00618  | .772  | .0174  | .00508  | .0123  | .939  | .0897  |

HOWLish v 1.0.0 can be downloaded: 
- [here](https://drive.google.com/file/d/1SdULuhgMdjlN5rLRAPm1dW6M6ASdT6Pp/view?usp=drive_link) for the TensorFlow SavedModel format; 
- [here](https://drive.google.com/file/d/1Sdt5TwN-OteMp7fV7ub9G109d-dSo8du/view?usp=sharing) for the frozen graph version 

## Detection pipeline

### Credits
The detection pipeline makes use of preprocessing scripts from teh original [VGGish repository](https://github.com/tensorflow/models/tree/master/research/audioset/vggish), all licensed under Apache License 2.0. We documented all changes to these scripts and included a link to the original version. 


> [!NOTE]
> All data was recorded between 2020 and 2024 in the Iberia Peninsula and is deposited, along with the associated manual annotations, in the Natural Sounds Archive of the Museum of Natural History and Sciences, University of Lisbon, Portugal. For access please contact the archive curator (paulo-marques@edu.ulisboa.pt) and to the collections head (geral@museus.ulisboa.pt).

# Howlish: a CNN for automated wolf howl detection
published in <ins>add link to publication when published</ins>

We developed HOWLish by applying transfer learning from [VGGish](https://github.com/tensorflow/models/tree/master/research/audioset/vggish) to a manually labelled dataset of 50,137 hours of soundscapes and 841 unique howling events recorded in the Iberia Peninsula. HOWLish was effectively trained on a train set containing 96,361 not-wolf and 17,927 wolf examples. Each example corresponds to 0.96 seconds of sound data represented by 96x64 log mel spectrogram.

HOWlish is able to retrieve 77% of the wolf examples of the test set (36,198,705 not-wolf examples and 5,081 wolf examples) at a prediction threshold of .5, with a false positive rate of 1.74%. In a real-world deployment setting, HOWLish was able to retrieve 81.3% of the howling events we detected through manual classification. Automated inference offered 22-fold reduction in the volume of data that needed to be manually processed by an operator, and a 15-fold reduction in operator time, when compared to manual annotation.

| Model  | Accuracy | Precision | Recall | Fall-out | F1-score | F2-score | AUC | PRC |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| HOWLish  | .983  | .00618  | .772  | .0174  | .00508  | .0123  | .939  | .0897  |

For a detailed description of the development process we suggest reading <ins>add link to publication when published</ins>.

> [!NOTE]
> All data was collected between 2020 and 2024 and is deposited, along with the associated manual annotations, in the Natural Sounds Archive of the Museum of Natural History and Sciences, University of Lisbon, Portugal. 

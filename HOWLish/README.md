# Howlish: a CNN for automated wolf howl detection
published in <ins>add link to publication when published</ins>

We developed HOWLish by applying transfer learning from [VGGish](https://github.com/tensorflow/models/tree/master/research/audioset/vggish) to a dataset of 50,137 hours of soundscapes with 1014 manually labelled howling events. 
For a detailed description of the development process we suggest reading <ins>add link to publication when published</ins>.

At a prediction threshold of .5, HOWlish is able to retrieve 77% of the wolf examples of the test set (36,198,705 not-wolf examples and 5,081 wolf examples) with a false positive rate of 1.74%. In a real-world deployment setting, HOWLish was able to retrieve 81.3% of the howling events we detected through manual classification. Automated inference using HOWLish offered 22-fold reduction in the volume of data that needed to be manually processed by an operator, and a 15-fold reduction in operator time, when compared to manual annotation.

| Model  | Accuracy | Precision | Recall | Fall-out | F1-score | F2-score | AUC | PRC |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| HOWLish  | .983  | .00618  | .772  | .0174  | .00508  | .0123  | .939  | .0897  |

HOWLish can be downloaded: 
- here for the TensorFlow checkpoint format; 
- here for the frozen graph version 

> [!NOTE]
> All data was recorded between 2020 and 2024 in the Iberia Peninsula and is deposited, along with the associated manual annotations, in the Natural Sounds Archive of the Museum of Natural History and Sciences, University of Lisbon, Portugal. For access please contact the archive curator (paulo-marques@edu.ulisboa.pt) and to the collections head (geral@museus.ulisboa.pt).

## Usage


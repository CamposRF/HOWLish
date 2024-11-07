# Howlish
published in <ins>add link to publication when published</ins>

HOWLish is a convolutional neural network trained for automated wolf howl detection. 

We developed HOWLish by applying transfer learning from [VGGish](https://github.com/tensorflow/models/tree/master/research/audioset/vggish) to a manually labeled dataset of 50,137 hours of soundscapes and 841 unique howling events recorded in the Iberia Peninsula. VGGish is based on the simple VGG’s (configuration A) architecture (Simonyan & Zisserman, 2015) and has 14 layers and 72.1M parameters, and was trained on Audioset (Gemmeke et al., 2017), a dataset, compiled by Google, of over 2 million human-labeled 10-second YouTube video soundtracks.

We preserved VGGish’s default input dimensions but adapted the Short-Time Fourier Transform to our 8 kHz sampling frequency data by using a window size of 0.05 seconds (400 samples). We also re-dimensioned the frequency axis to a maximum frequency of 2,000 Hz, which matches our labeling process and encompasses the bulk of the wolf howling bandwidth. We balanced our dataset by randomly undersampling our majority class, stratified by soundscape file. We preserved VGGish’s original architecture (top included) and added a sigmoid layer and batch normalization, before activation.

HOWlish was able to reteieve 77% of the wolf examples of the test set at a prediction thresholf of .5, with a false positive rate of 1.74%. 

| Model  | Accuracy | Precision | Recall | Fall-out | F1-score | F2-score | AUC | PRC |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| HOWLish  | .983  | .00618  | .772  | .0174  | .00508  | .0123  | .939  | .0897  |


All data was collected between 2020 and 2024 and is deposited, along with the associated manual annotations, in the Natural Sounds Archive of the Museum of Natural History and Sciences, University of Lisbon, Portugal. You can find detailed information on HOWLish in <ins>add link to publication when published</ins>.


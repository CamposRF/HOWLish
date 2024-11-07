# Howlish
published in <ins>add link to publication when published</ins>

HOWLish is a convolutional neural network trained for automated wolf howl detection. 

We developed HOWLish by applying transfer learning from [VGGish](https://github.com/tensorflow/models/tree/master/research/audioset/vggish) to a manually labeled dataset of 50,137 hours of soundscapes and 841 unique howling events recorded in the Iberia Peninsula. We preserved VGGish’s default input dimensions but adapted the Short-Time Fourier Transform step to match our data and labeling approach. Following VGGish's 0.96s temporal input dimention, our dataset contained 173,771,879 examples on the majority class (not-wolf) and 25,298 examples on the minority class (wolf), which we balanced by randomly undersampling our majority class, stratified by soundscape file. HOWLish was effectly trained on a train set containing 96,361 not-wolf and 17,927 wolf examples. 

HOWlish was able to reteieve 77% of the wolf examples of the test set (36,198,705 not-wolf examples and 5,081 wolf examples) at a prediction thresholf of .5, with a false positive rate of 1.74%. 

| Model  | Accuracy | Precision | Recall | Fall-out | F1-score | F2-score | AUC | PRC |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| HOWLish  | .983  | .00618  | .772  | .0174  | .00508  | .0123  | .939  | .0897  |


All data was collected between 2020 and 2024 and is deposited, along with the associated manual annotations, in the Natural Sounds Archive of the Museum of Natural History and Sciences, University of Lisbon, Portugal. You can find detailed information on HOWLish in <ins>add link to publication when published</ins>.


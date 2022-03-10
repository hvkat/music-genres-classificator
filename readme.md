# PyTorch Music Genres Classifier
Simple classifier based on neural network consisting of fully connected layers.
Project build in order to grow own skills, for personal, no commercial use. 

Project operates on audio features.

# Dataset
Dataset used is
[GTZAN Dataset - Music Genre Classification](http://marsyas.info/downloads/datasets.html). The collection consists of samples of 10 music genres, 100 samples each. 


# Preprocessing
Data was preprocessed with use of [librosa](https://librosa.org/), a tool for audio and music processing and analysis in Python (*version librosa 0.9.0, DOI: 10.5281/zenodo.5996429*).
Specific audio features were retrieved.

# Results
## Model 1 (Fully Connected with 4 layers)

Training accuracy: 81.62. Validation accuracy: 35.5.
![](https://github.com/hvkat/music-genres-classificator/blob/main/output/model1-curves.jpg?raw=true)

## Model 2 (Fully Connected with 7 layers)

Training accuracy: 98.88. Validation accuracy: 56.5.
![](https://github.com/hvkat/music-genres-classificator/blob/main/output/model2-curves.jpg?raw=true)


# References
Tzanetakis, George and Cook, Perry. (2002). Musical Genre Classification of Audio Signals. IEEE Transactions on Speech and Audio Processing. 10. 293 - 302. 10.1109/TSA.2002.800560.

McFee, Brian, Colin Raffel, Dawen Liang, Daniel PW Ellis, Matt McVicar, Eric Battenberg, and Oriol Nieto. “librosa: Audio and music signal analysis in python.” In Proceedings of the 14th python in science conference, pp. 18-25. 2015.


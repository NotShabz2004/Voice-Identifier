# Voice-Identifier
Able to learn the voices of people and then identify when given a clip. Accepts files (.wav,mp3.ogg)
You can edit the code as required and choose whose voices need to be identified. Currently I have chosen Rayyan, Swen, Riyaad and Adithya as my dataset.
Also by adding more voices I was able to solve the overfitting issue, where the machines learns too well and is unable find differences. This can be avoided by adding more variance in voice.
The voice is processed by the librosa library and tensorflow is used to train the model. Math plotlib and Numpy for plotting the learning curve graphs and numerical calculations for the X and Y values.
Unable to upload dataset because of the size since it requires many voices to work efficiently with 95.6% accuration as of now.

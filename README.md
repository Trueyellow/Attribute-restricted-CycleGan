# Pattern Recognition Course Project #

This is our final project for course Pattern Recognition, and here is our abstract, and in the pr-final-project.pdf

## Abastraction
*In this project, we come up with an idea that combines three neural network models to achieve a speech-emotion to facial-expression translation. We propose a CNN network for audio signal processing and recognition, a GAN for generating the final result with the emotion and facial expression, a pre-trained CNN for our GAN's enhancement. Our pilot approach is to use cycleGAN combining the result of CNN as the restricting-attributes for cycleGAN. It restricts the condition of GAN such that we can get expected results from affecting GAN by not only the data distribution that generator learned but also other kinds of attributes from the multi-modal feature. We will use different datasets to evaluate and test our model and make sure the robustness and accuracy of our combined network. We achieve a purpose of "machine imagination" which can be interpreted as the speech-image translation.*

## Methdology
$L(G_{(X, Z)\rightarrow Y}, D_Y)=\min\limits_{\Theta_g}\max\limits_{\Theta_d}\lbrace E_{y, z}[logD_Y(y,z)] + E_{x,z}[log(1-D_Y(G_{(X,Z)\rightarrow Y}(x,z), z))]\rbrace$
## Test result
We used two datasets to train and test our model. 
![demo1](/assets/demo1.PNG)

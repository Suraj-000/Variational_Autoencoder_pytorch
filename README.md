# Variational_Autoencoder_pytorch

## Implementing a Variational Autoencoder using pytorch for point cloud generation and Interpolation.

#### Variational Autoencoder

A variational autoencoder (VAE) is a type of artificial neural network used in the field of unsupervised learning and generative modeling. It is a probabilistic graphical model that combines deep learning with variational inference. VAEs are designed to learn a low-dimensional representation of input data in an unsupervised manner and generate new data points similar to the training data.

The encoder network takes point cloud data and maps it to a distribution in the latent space. This distribution is typically Gaussian with a mean and variance. The latent space is a lower-dimensional space where each point represents a different encoding of the point cloud data. It captures the underlying structure of the data in a compact form. From the learned distribution in the latent space, random samples are drawn. These samples are then decoded to reconstruct the point cloud data. The decoder network takes samples from the latent space and reconstructs the original point cloud.

#### Functional Block Diagram

Input point cloud is normalised and is given to VAE network, the VAE network learns geometric patterns from input, the encoder learns a latent representation and decoder decodes the latent vector to reconstruct the original point cloud. The learnt representation of input point cloud is then used for classification of point clouds.

![Image](https://github.com/user-attachments/assets/d44b8669-e022-46af-9efa-3e089f297aed)

#### Decoder

The decoder consists of a sequence of fully connected layers followed by ReLU activations, with dimensions specified as 246, 512, and 1024 respectively. In the forward pass, the input is passed through the layers and reshaped to match the desired output shape. Overall, the Decoder transforms latent representations into multi-dimensional output tensors as shown in Figure. This decoder is used to reconstruct point clouds from latent representations in generative models like variational autoencoders.

<p align="center">
    <img src="https://github.com/user-attachments/assets/b18d1f2f-fae2-42dd-904c-cd5a7c061b76" width="50%">
</p>

#### Classifier Network

It takes a latent input of dimension 128 and outputs a log probability distribution over classes. The architecture consists of linear layers with dimensions specified as 128,64,32 with batch normalization, ReLU activations, and dropout for regularization, enabling it to learn discriminative features from input point cloud data and effectively classify them into predefined categories.

<p align="center">
    <img src="https://github.com/user-attachments/assets/b4122cbb-2762-4ec3-bd59-231678748fb8" width="50%">
</p>

## Reconstructed samples

#### Visualisation of 3D reconstructed models and ground truth of point clouds ModelNet40 dataset. Grey represents input point cloud and Green represents the reconstructed point cloud.
<img width="956" alt="Image" src="https://github.com/user-attachments/assets/6638b0db-d846-4313-b304-2103fde28b01" />


## Intepolation between chair 1 and chair 2

| ![Image](https://github.com/user-attachments/assets/ea7c6d34-9daf-44ad-a68b-55afe45e2e9d) | ![Image](https://github.com/user-attachments/assets/e0dee3f3-149d-42a7-bbe1-a906455b9524) |  ![Image](https://github.com/user-attachments/assets/6e0e5be4-572a-43d3-ad35-240f73381ade) |  ![Image](https://github.com/user-attachments/assets/4e5a8fa8-8afb-417f-915f-7a01ebbdadb2)  |
|-|-|-|-|

| ![Image](https://github.com/user-attachments/assets/a958a313-096d-482a-92e0-0d3b60689044) | ![Image](https://github.com/user-attachments/assets/84e61c26-0186-4c78-be7e-c32a9e5eabd6) |  ![Image](https://github.com/user-attachments/assets/34369abc-cd77-4b10-89f1-1364146d2fd2)  | ![Image](https://github.com/user-attachments/assets/8a69e9c8-ea38-4116-a7e4-ea50480ef2ba)|
|-|-|-|-|

## Reconstructed samples

| ![Figure_1](https://github.com/Suraj-000/Variational_Autoencoder_pytorch/assets/83648833/afe5502f-f37d-4245-9d6d-012c4ccda950) |
|-|

| ![Figure_2](https://github.com/Suraj-000/Variational_Autoencoder_pytorch/assets/83648833/c478a115-67bf-403e-93af-b1f420383555) |
|-|

| ![Figure_3](https://github.com/Suraj-000/Variational_Autoencoder_pytorch/assets/83648833/c3c54590-9dba-4fe8-891c-306d3fb060d8) |
|-|

## Intepolation between chair 1 and chair 2

![Image](https://github.com/user-attachments/assets/f21ce5ec-a358-40f6-8515-d26711bf77a3)
![Image](https://github.com/user-attachments/assets/3a8ba602-7c7c-479e-83dd-d82c5b11f2c1)
![Image](https://github.com/user-attachments/assets/db528283-30d0-4515-af51-0751dcc97a6c)
![Image](https://github.com/user-attachments/assets/1c073037-99ec-4fab-b17b-c50ffb91af15)
![Image](https://github.com/user-attachments/assets/c4e5b16e-891a-45cf-a044-f7f6321ce61c)
![Image](https://github.com/user-attachments/assets/3b939a25-a3d5-4955-afd5-173cba6234be)
![Image](https://github.com/user-attachments/assets/0f0e8884-f633-41b8-88a4-433f2954542e)


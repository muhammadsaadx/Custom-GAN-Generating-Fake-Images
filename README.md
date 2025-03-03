# Cat Image Generator with GANs

## Architecture

### Generator
The generator creates synthetic images from random noise vectors. It uses a series of transposed convolution layers to progressively upsample from the latent space to a 32x32 RGB image.

- **Input**: Random noise vector of size 100
- **Architecture**: 
  - Linear layer to reshape noise (100 → 4×4×512)
  - Four upsampling blocks with transposed convolutions
  - Final convolutional layer with tanh activation to produce normalized RGB output
- **Output**: 32×32×3 RGB image with pixel values normalized between -1 and 1

### Discriminator
The discriminator evaluates whether an image is real or generated. It uses convolutional layers to progressively downsample the image to a single scalar output.

- **Input**: 32×32×3 RGB image
- **Architecture**: 
  - Three convolutional blocks with LeakyReLU activations
  - Flatten layer followed by linear layer to produce a single value
  - Sigmoid activation to output probability
- **Output**: Scalar value between 0 and 1 (1 = real, 0 = fake)

## Training Process

1. **Data Preparation**:
   - The CIFAR-10 dataset is loaded and filtered to extract only cat images (class 3)
   - Images are normalized to range [-1, 1]

2. **Adversarial Training**:
   - For each epoch, batches of real images are sampled
   - The discriminator is trained to classify real images as real (1) and generated images as fake (0)
   - The generator is trained to produce images that the discriminator classifies as real
   - Loss functions: Binary Cross Entropy for both networks

3. **Training Parameters**:
   - Batch size: 64
   - Learning rate: 0.0002
   - Adam optimizer with betas=(0.5, 0.999)
   - Epochs: 100
   - Latent dimension: 100

## Results

The training shows characteristic GAN behavior:
- Initially both networks have high losses as they learn the data distribution
- The discriminator loss gradually decreases as it gets better at distinguishing real from fake
- The generator loss initially increases then fluctuates as it tries to fool the increasingly effective discriminator

The generated images progressively improve in quality throughout training. The final model produces plausible cat-like images with recognizable features.

## Usage

### Requirements
- PyTorch
- torchvision
- matplotlib

### Training
```python
# Train the model
python train_gan.py
```

### Generate Images
```python
# Generate new cat images using the trained model
python generate_images.py
```

## Future Improvements

1. **Architecture Enhancements**:
   - Add residual connections for more stable training
   - Implement progressive growing for higher resolution images

2. **Training Stability**:
   - Implement spectral normalization or gradient penalty for improved Wasserstein GAN performance
   - Experiment with different learning rates and batch sizes

3. **Image Quality**:
   - Train on a larger, higher-resolution dataset of cat images
   - Implement conditional GAN for more controlled generation

## References

- [Original GAN Paper by Goodfellow et al.](https://arxiv.org/abs/1406.2661)
- [DCGAN Paper by Radford et al.](https://arxiv.org/abs/1511.06434)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

# Introduction

This package describes the implementation and usage of a deep recurrent encoder used to encode complex EEG signals into an embedded representation with much lesser data quantity.

Let’s explore each aspect of this in detail.  

---

## What is EEG?

Electroencephalography (EEG) signals represent the electrical activity of the brain, captured through electrodes placed on the scalp. These signals arise from the synchronized firing of neurons, particularly the postsynaptic potentials of cortical pyramidal cells. EEG is a valuable tool for studying brain function, diagnosing neurological conditions, and exploring cognitive processes.

### Characteristics of EEG Signals

- **Amplitude and Frequency**: EEG signals are typically low in amplitude, ranging from a few microvolts to hundreds of microvolts. Their frequency is categorized into bands such as:
  - **Delta (0.5–4 Hz)**: Associated with deep sleep and unconscious states.
  - **Theta (4–8 Hz)**: Linked to light sleep, relaxation, and creativity.
  - **Alpha (8–12 Hz)**: Indicates calmness and relaxed wakefulness.
  - **Beta (13–30 Hz)**: Reflects active thinking and alertness.
  - **Gamma (30+ Hz)**: Related to high-level cognitive functioning, such as problem-solving and memory.

- **Temporal and Spatial Resolution**: EEG provides high temporal resolution, capturing rapid brain activity in milliseconds. However, its spatial resolution is limited, as the signals reflect activity across broader cortical areas rather than specific deep brain regions.

### Applications of EEG

#### Medical Diagnosis:
- Commonly used to diagnose conditions like epilepsy, where abnormal brain wave patterns are evident.
- Assists in detecting sleep disorders, such as insomnia or sleep apnea.
- Supports the evaluation of coma states and brain death.

#### Neuroscientific Research:
- Studies cognition, sensory processing, and emotional responses.
- Used in brain-computer interface (BCI) systems to enable control of devices through thought patterns.

#### Psychological and Cognitive Applications:
- Helps analyze stress, mental workload, and relaxation levels.
- Investigates disorders like ADHD and autism through distinct EEG patterns.

---

## What is an Autoencoder?

An autoencoder is a type of artificial neural network designed for unsupervised learning, primarily used for dimensionality reduction, feature learning, and data denoising. It works by compressing input data into a smaller, latent representation and then reconstructing it to be as close to the original as possible.

### Key Components:
- **Encoder**: Maps the input data to a lower-dimensional latent space.
- **Latent Space**: A compressed representation of the input, capturing its most significant features.
- **Decoder**: Reconstructs the input data from the latent representation.

### Applications:
- **Dimensionality Reduction**: Reduces complex data to fewer dimensions for visualization or preprocessing.
- **Denoising**: Removes noise from data, such as images or signals.
- **Anomaly Detection**: Identifies unusual patterns by comparing input to reconstructed output.

The diagram above illustrates how an autoencoder works: input data is passed through the encoder to the bottleneck (latent space) and then reconstructed by the decoder.

---

## Why do we need an Autoencoder?

EEG signals in real life are presented in channels, each channel corresponding to a specific electrode placed on the scalp. More frequently, these may be in the order of more than 150-200 channels. This amount of data is very large and highly complex to process for any application. Hence, we use an autoencoder to learn a smaller latent space representation of the data, reducing computational complexity in later stages.

### Architecture:
1. The **200 channels** of EEG signals are fed into a **deep recurrent network**. This is a recurrent network since we need to encode not only the data in the current timestamp but also previous states.
2. The **neural network** compresses the input data into a **latent space** of a size pre-specified by the designer. The higher the latent space dimension, the better the accuracy but at the cost of increased space complexity.
3. Using this latent space, the **decoder** attempts to reconstruct the original signal. The model learns the most accurate latent space representation, maximizing captured information.

---

## About the DRE Package

The current implementation of the **Deep Recurrent Encoder (DRE)** model is as follows:

```julia
function DRE(in_chs::Int, hidden_chs::Int, out_chs::Int; kernel_size=4, stride=2)

    return DRE(
        
        LSTMCell(hidden_chs => hidden_chs),

        Conv((kernel_size,), (in_chs => hidden_chs), identity, stride=(stride,), use_bias=true, pad=SamePad()),

        ConvTranspose((kernel_size,), (hidden_chs => out_chs), identity, stride=(stride,), use_bias=true, pad=SamePad())

    )

end
```
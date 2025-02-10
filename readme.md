# Health Assistance through Fine-Tuned LoRA-Based Language Models

## Author

**Aayush Adhikari**\
Email: [220426@softwarica.edu.np](mailto:220426@softwarica.edu.np)\
Coventry ID: 13898296

## Overview

This project leverages **Gemma**, a family of lightweight, state-of-the-art open models, to develop a **healthcare chatbot**. It fine-tunes **Gemma** using **Low-Rank Adaptation (LoRA)** to provide improved healthcare assistance for users in **Kathmandu**.

Large Language Models (LLMs) are powerful for various **NLP tasks** but require fine-tuning for domain-specific applications. **LoRA fine-tuning** allows efficient customization while reducing computational overhead.

## Features

- Fine-tunes **Gemma 2B** model using **LoRA** for efficient adaptation.
- Uses **Google Colab** for training with **T4 GPU**.
- Loads and preprocesses healthcare-related dataset.
- Implements **AdamW optimizer** and **Sparse Categorical Crossentropy loss**.
- Visualizes **accuracy** and **loss trends** across epochs.
- Saves and loads fine-tuned models.

## Setup

### 1. Get Access to Gemma

Follow [Gemma Setup Guide](https://ai.google.dev/gemma/docs/setup) to:

- Access **Gemma** on [Kaggle](https://kaggle.com).
- Configure a Kaggle API key.
- Select a Colab runtime with a **T4 GPU**.

### 2. Select Runtime in Google Colab

1. Click **Runtime** > **Change runtime type**.
2. Select **T4 GPU** under "Hardware Accelerator".

### 3. Configure API Key

Generate a Kaggle API key and store it in **Colab Secrets**:

```python
import os
from google.colab import userdata
os.environ["KAGGLE_USERNAME"] = userdata.get('KAGGLE_USERNAME')
os.environ["KAGGLE_KEY"] = userdata.get('KAGGLE_KEY')
```

### 4. Install Dependencies

Run the following:

```sh
pip install -q -U keras-nlp keras tensorflow-text tensorflow-hub keras-hub
```

### 5. Select Backend

Set **JAX** as the backend:

```python
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="1.00"
```

## Dataset Preparation

Mount Google Drive and load dataset:

```python
from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
df = pd.read_json('/content/drive/MyDrive/llm/dataset/newdata.json')
```

Process JSON data:

```python
import json
data = []
with open("/content/drive/MyDrive/llm/dataset/newdata.json", 'r') as file:
    for line in file:
        features = json.loads(line)
        for i in features:
            data.append(f"Instruction:\n{i['prompt']}\n\nResponse:\n{i['response']}")
```

## Load and Fine-Tune the Model

Instantiate **Gemma 2B** model:

```python
import keras_nlp
gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma2_2b_en")
```

Enable **LoRA Fine-Tuning**:

```python
gemma_lm.backbone.enable_lora(rank=4)
```

Compile Model:

```python
from keras.optimizers import AdamW
gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=AdamW(learning_rate=5e-5, weight_decay=0.01),
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
```

Train Model:

```python
gemma_lm.fit(data, epochs=2, batch_size=1)
```

Save Fine-Tuned Model:

```python
my_model_name = "my_gemma2_pt1"
gemma_lm.save_to_preset(f"/content/drive/MyDrive/llm/{my_model_name}")
```

## Inference (Generating Responses)

```python
template = "Instruction:\n{prompt}\n\nResponse:\n{response}"
prompt = template.format(prompt="In Kathmandu there is huge air pollution. What should I do?", response="")
print(gemma_lm.generate(prompt, max_length=256))
```

## Plot Training Metrics

```python
from matplotlib import pyplot as plt
history = gemma_lm.history.history
history['epochs'] = [i for i in range(1,6)]

plt.plot(history['epochs'], history['loss'])
plt.title('Loss across epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

plt.plot(history['epochs'], history['sparse_categorical_accuracy'])
plt.title('Accuracy across epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
```

## Notes

- **Use full precision** for fine-tuning, and consider mixed precision for faster inference.
- Fine-tuning with **LoRA rank=4** keeps model memory-efficient while improving response quality.
- The chatbot is optimized for **health-related queries in Kathmandu**.

## Acknowledgment

- **Softwarica College of E-commerce and IT** & **Coventry University** for academic support.
- Kaggle for dataset hosting.
- Google Colab & TensorFlow for providing accessible model training resources.


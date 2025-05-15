# HuggingFace LLM Course

Hugging Face is a company and open-source platform that has become a central hub for AI and machine learning, especially in natural language processing (NLP). It provides tools, libraries, and repositories that make it easier to work with state-of-the-art AI models, particularly transformers (like BERT, GPT, and others).

## Key Offerings from Hugging Face:
1. 🤗 Transformers Library
    - A popular Python library that provides thousands of pre-trained AI models (like GPT, BERT, T5, etc.) for tasks like text generation, translation, summarization, and more.
    - Works with PyTorch, TensorFlow, and JAX.
      
2. 🤗 Hub (Model & Dataset Repository)
    - A platform where researchers and developers share pre-trained models (like Llama 2, Stable Diffusion, Mistral) and datasets.
    - Think of it as the "GitHub for AI models."

3. Inference API & Hosting
    - Allows users to run AI models in the cloud without setting up infrastructure.
    - Useful for deploying models quickly.

4. Spaces (AI App Demos)
    - Lets users create and share web demos of AI models (like chatbots, image generators).
    - Example: [https://huggingface.co/spaces](https://huggingface.co/spaces) (many free AI tools here!)

5. Collaborations with Big Tech
    - Hugging Face works with companies like Microsoft, Google, and Meta to distribute models (e.g., Meta’s Llama 2 is available on Hugging Face).
## Why is Hugging Face So Popular?
- Open-source friendly – Many free models and tools.
- Easy to use – Simplifies working with complex AI models.
- Community-driven – Thousands of contributors share models and datasets.

## Use Cases
- Running LLMs (like GPT, Llama, Mistral).
- Fine-tuning models for custom tasks (e.g., chatbots, sentiment analysis).
- Deploying AI models without deep learning expertise.

## File List
- `00_setup.ipynb`:
    - Development environment setup
- `01_introduction.ipynb`:
    - Course Introduction
- `02_nlp_and_llm.ipynb`:
    - NLP Definition
    - The Rise of Large Language Models (LLMs)
- `03_transformers_do.ipynb`:
    - Introduction to HuggingFace Transformers `pipeline` library
    - Using HuggingFace Transformers `pipeline` library to do:
        -  Zero-shot classification
        -  Text generation
        -  Using any model from the Hub in a pipeline
        -  Mask filling
        -  Named entity recognition (NER)
        -  Question answering
        -  Summarization
        -  Translation
        -  Image classification
        -  Automatic speech recognition
- `04_how_transformer_works.ipynb`:
    - A bit of Transformer history
    - Transformers are language models (they have been trained on large amounts of raw text in a self-supervised fashion)
    - Transfer Learning / Fine Tuning (the benefits of using existing pre-trained models and fine tuned it)
    - General Transformer architecture:
        - `Encoder-only models`: Good for tasks that require understanding of the input, such as sentence classification and named entity recognition.
        - `Decoder-only models`: Good for generative tasks such as text generation.
        - `Encoder-decoder models` or `sequence-to-sequence models`: Good for generative tasks that require an input, such as translation or summarization.
    - Attention layers in a model based on a paper title `Attention Is All You Need!`
    - The original architecture: `The Transformer architecture was originally designed for translation`
    - Architectures vs. checkpoints:
        - **Architecture**: This is the skeleton of the model — the definition of each layer and each operation that happens within the model.
        - **Checkpoints**: These are the weights that will be loaded in a given architecture.
        - **Model**: This is an umbrella term that isn’t as precise as “architecture” or “checkpoint”: it can mean both. This course will specify architecture or checkpoint when it matters to reduce ambiguity.

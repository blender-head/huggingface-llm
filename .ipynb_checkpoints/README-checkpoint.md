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

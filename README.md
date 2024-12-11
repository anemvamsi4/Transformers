# Training a Lightweight Transfomer Model

This project demonstrates the training and evaluation of a lightweight Transformer-based language model to generate bilingual (Telugu-English) lyrics. The model, inspired by GPT-2, uses a decoder-only architecture and was trained on a custom dataset sourced from Kaggle. It explores efficient tokenization, lightweight architecture, and GPU resource utilization.

## Features
- **Transformer Architecture:** A decoder-only model with 6.58M parameters, embedding size of 256, and 8 decoder layers.
- **Tokenization:** Byte Pair Encoding (BPE) reduces sequence length while maintaining vocabulary size 512.
- **Bilingual Dataset:** Telugu-English lyrics dataset optimized for model training.
- **Optimized Training:** Conducted on an Nvidia P100 GPU with efficient use of resources.



## Dataset
The dataset consists of Telugu lyrics interspersed with English words, sourced from Kaggle. It initially contained over 600,000 Python code points (characters). After applying Byte Pair Encoding (BPE) tokenization with a vocabulary size of 512, the token count was reduced to approximately 300,000+.



## Tokenization
Two approaches were considered:
1. **Character-Level Tokenization:** Resulted in longer sequences (600k+ tokens) with a smaller vocabulary size of 165.
2. **BPE Tokenization:** Reduced sequence length to ~300k tokens with a vocabulary size of 512. This approach was chosen for computational efficiency.



## Model Architecture
- **Type:** Decoder-only Transformer (inspired by GPT-2).
- **Embedding Dimension:** 256.
- **Decoder Layers:** 8.
- **Vocabulary Size:** 512.
- **Normalization:** RMSNorm replaces LayerNorm for improved stability.
- **Total Parameters:** 6.58M.



## Training Details
- **Hardware:** Nvidia P100 GPU (16GB memory).
- **Optimizer:** AdamW with weight decay.
- **Learning Rate:** 3e-4
- **Epochs:** 10.
- **Time per Epoch:** ~53 minutes.
- **Total Training Time:** ~9 hours 30 minutes.

---

## Evaluation
- **Metrics:**
  - Cross-Entropy Loss: Reduced from 4.5 to 0.02 over 10 epochs.
  - Perplexity: Evaluated for predictive power.
- **Challenges:** Overfitting risk due to the small dataset size (300k tokens).



## Results

! [training loss curve](lyricGPT_training.png)
- **Training Performance:** Efficient use of GPU memory and resources with a consistent loss reduction.
- **Text Generation:** The model generates coherent bilingual lyrics but shows some repetition, likely due to dataset limitations.
- **Key Insights:** Demonstrates feasibility of training Transformer models for low-resource languages.



## Setup and Usage

### Requirements
- Python 3.8+
- Required Libraries (install via `requirements.txt`):
  ```bash
  pip install -r requirements.txt
  ```

### Steps to Run
1. **Preprocess the Dataset:**
   ```bash
   python dataset/preprocessing.py
   ```
2. **Train the Model:**
   ```bash
   python src/train.py
   ```
3. **Generate Text:**
   ```bash
   python src/inference.py
   ```



## Future Work
- Expand the dataset to improve model generalization.
- Experiment with additional regularization techniques.
- Optimize the model further for real-world applications.



## References
1. Vaswani, A. et al. (2017). *Attention Is All You Need.*
2. Radford, A. et al. (2019). *Language Models Are Unsupervised Multi-Task Learners.*
3. Touvron, H. et al. (2023). *LLaMA: Open and Efficient Foundation Language Models.*



## Contact
Developed by [Anem Vamsi](mailto:anemvamsi4@gmail.com). 

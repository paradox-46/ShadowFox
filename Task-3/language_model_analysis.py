# language_model_analysis.py
#!pip install transformers torch datasets matplotlib seaborn

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification, pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from datasets import load_dataset

print("Starting Language Model Analysis with DistilBERT...")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# 1. LM Selection: DistilBERT
model_name = "distilbert-base-uncased"
print(f"Loading model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

print("Model and tokenizer loaded successfully!")

# 2. Basic Functionality Demonstration
print("\n" + "="*50)
print("EXPERIMENT 1: Basic Fill-Mask Functionality")
print("="*50)

# Create fill-mask pipeline
fill_mask = pipeline("fill-mask", model=model_name, tokenizer=model_name)

# Test sentences
test_sentences = [
    "The capital of France is [MASK].",
    "I want to [MASK] a book about machine learning.",
    "The weather today is very [MASK]."
]

for i, sentence in enumerate(test_sentences, 1):
    print(f"\nTest {i}: {sentence}")
    results = fill_mask(sentence, top_k=3)
    for j, result in enumerate(results, 1):
        print(f"  {j}. {result['token_str']} (score: {result['score']:.4f})")

# 3. Contextual Understanding Analysis
print("\n" + "="*50)
print("EXPERIMENT 2: Contextual Understanding")
print("="*50)

contextual_sentences = [
    "I deposited my money in the [MASK].",   # Bank (financial)
    "The fisherman sat on the [MASK].",      # Bank (river)
    "The battery in my phone is [MASK].",    # Dead/low
    "The soldier was charged with [MASK]."   # Murder/negligence
]

for sentence in contextual_sentences:
    print(f"\nSentence: {sentence}")
    results = fill_mask(sentence, top_k=3)
    for i, res in enumerate(results, 1):
        print(f"  {i}. {res['token_str']} (score: {res['score']:.4f})")

# 4. Bias Analysis
print("\n" + "="*50)
print("EXPERIMENT 3: Bias Analysis")
print("="*50)

bias_test_sentences = [
    "The [MASK] cooked a delicious meal.",
    "The [MASK] fixed the car engine.",
    "The [MASK] wrote a new novel.",
    "The [MASK] took care of the patients in the hospital.",
    "The [MASK] wrote the code for the new software."
]

bias_results = []

for sentence in bias_test_sentences:
    results = fill_mask(sentence, top_k=5)
    sentence_results = []
    for res in results:
        sentence_results.append({
            'token': res['token_str'],
            'score': res['score']
        })
    bias_results.append({
        'sentence': sentence,
        'predictions': sentence_results
    })
    
    print(f"\n{sentence}")
    for i, res in enumerate(results, 1):
        print(f"  {i}. {res['token_str']} (score: {res['score']:.4f})")

# 5. Visualization of Bias Results
print("\nVisualizing bias analysis results...")
# Prepare data for visualization
roles = ['cook', 'mechanic', 'author', 'nurse', 'programmer']
gender_data = {'male': [], 'female': []}

for result in bias_results:
    male_score = 0
    female_score = 0
    
    for prediction in result['predictions']:
        token = prediction['token'].lower()
        score = prediction['score']
        
        if token in ['he', 'man', 'male', 'boy', 'him']:
            male_score += score
        elif token in ['she', 'woman', 'female', 'girl', 'her']:
            female_score += score
    
    gender_data['male'].append(male_score)
    gender_data['female'].append(female_score)

# Create visualization
x = np.arange(len(roles))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width/2, gender_data['male'], width, label='Male', color='blue')
rects2 = ax.bar(x + width/2, gender_data['female'], width, label='Female', color='pink')

ax.set_ylabel('Probability Score')
ax.set_title('Gender Association with Different Professions')
ax.set_xticks(x)
ax.set_xticklabels(roles)
ax.legend()

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('gender_bias_analysis.png')
plt.show()

# 6. Sentiment Analysis with Visualization
print("\n" + "="*50)
print("EXPERIMENT 4: Sentiment Analysis with Attention Visualization")
print("="*50)

# Load a sentiment analysis model
sentiment_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)

# Function to get attention weights
def get_attention(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model(**inputs, output_attentions=True)
    attention = outputs.attentions[-1]  # Get attention from last layer
    return attention, inputs

# Test sentences for sentiment
sentiment_texts = [
    "This movie is fantastic and incredibly engaging!",
    "I absolutely hated the film, it was terrible.",
    "The product is okay, nothing special but not bad either."
]

for text in sentiment_texts:
    print(f"\nAnalyzing: '{text}'")
    
    # Get sentiment prediction
    sentiment_classifier = pipeline("sentiment-analysis", model=sentiment_model_name, tokenizer=sentiment_model_name)
    result = sentiment_classifier(text)[0]
    print(f"Sentiment: {result['label']} (confidence: {result['score']:.4f})")
    
    # Get attention weights
    attention, inputs = get_attention(text, sentiment_model, sentiment_tokenizer)
    tokens = sentiment_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Visualize attention
    plt.figure(figsize=(12, 8))
    
    # Average attention across all heads
    avg_attention = attention.mean(dim=1).squeeze().detach().numpy()
    
    # Create heatmap
    sns.heatmap(avg_attention, 
                xticklabels=tokens, 
                yticklabels=[f"Layer {i+1}" for i in range(avg_attention.shape[0])],
                cmap="viridis")
    plt.title(f"Attention Weights for: '{text}'")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"attention_{hash(text)}.png")
    plt.show()

# 7. Language Generation Experiment
print("\n" + "="*50)
print("EXPERIMENT 5: Text Generation with Sampling")
print("="*50)

# Manual text generation using the model
def generate_text(prompt, max_length=50, temperature=1.0):
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            top_k=50,
            top_p=0.95
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

prompts = [
    "The future of artificial intelligence",
    "In a world where machines can think",
    "Climate change is"
]

for prompt in prompts:
    generated = generate_text(prompt, max_length=50, temperature=0.9)
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated}")

# 8. Conclusion and Insights
print("\n" + "="*50)
print("CONCLUSIONS AND INSIGHTS")
print("="*50)

print("""
1. CAPABILITIES:
- DistilBERT demonstrates strong performance on fill-mask tasks, showing understanding of grammar and context.
- The model successfully handles polysemy (words with multiple meanings) based on context.
- It shows reasonable performance on sentiment analysis tasks.

2. LIMITATIONS AND BIASES:
- The model exhibits gender biases, associating certain professions more with one gender.
- Generated text can sometimes be inconsistent or nonsensical.
- The model lacks true understanding and operates based on patterns in training data.

3. PRACTICAL APPLICATIONS:
- Could be used for text completion features in writing assistants.
- Useful for sentiment analysis in customer feedback systems.
- Can serve as a base for more specialized NLP applications through fine-tuning.

4. ETHICAL CONSIDERATIONS:
- Bias mitigation techniques should be applied before deployment in production systems.
- Users should be aware of limitations in understanding and generation capabilities.
- Outputs should be monitored for potentially harmful or biased content.
""")

print("\nLanguage model analysis completed successfully!")
# README

## Overview
This project demonstrates the use of Hugging Face's `transformers` library in conjunction with TensorFlow to translate English news headlines into French. The translation process is executed using two different methods:

1. **Manual Tokenization and Translation using T5 Model**
2. **Using the Transformer Pipeline for Translation**

### Prerequisites
Before running the code, ensure that the necessary libraries are installed and properly configured. The key dependencies include:

- `transformers` library
- `tensorflow`
- Python 3.10 or above

### Installation
If you're running this in Google Colab, you may need to install the `transformers` library by uncommenting and running the following line:

```bash
!pip install transformers
```

### Translation Methods

#### 1. Manual Tokenization and Translation using T5 Model

In this method:
- **AutoTokenizer**: Tokenizes the input headlines into token IDs using the `t5-base` model.
- **TFAutoModelForSeq2SeqLM**: Converts token IDs back into natural language sentences (translated text).

```python
# Import the necessary classes from the transformers module.
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

# Initialize the tokenizer and translation model.
tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=256)
translation_model = TFAutoModelForSeq2SeqLM.from_pretrained("t5-base")

# Define the news headlines.
headlines = [
    'How To Spend More Time With Your Family While Working Remotely',
    'NCAA Football Playoffs Should be Like the NFL',
    # Other headlines...
]

# Tokenize the headlines and generate translations.
headline_input_ids = [tokenizer(f"translate English to French: {headline}", return_tensors="tf").input_ids for headline in headlines]
translated_headlines = [tokenizer.decode(translation_model.generate(input_id, max_new_tokens=100)[0], skip_special_tokens=True) for input_id in headline_input_ids]

# Print the translated headlines.
for translation in translated_headlines:
    print(translation)
```

#### 2. Using the Transformer Pipeline for Translation

In this method:
- **Pipeline**: A high-level abstraction that handles the tokenization, model processing, and decoding in a streamlined manner.

```python
# Import the pipeline class from the transformers module.
from transformers import pipeline

# Initialize the translation pipeline.
translator = pipeline("translation", model="t5-base")

# Translate the headlines.
translated_headlines = [translator(f"translate English to French: {headline}")[0]['translation_text'] for headline in headlines]

# Print the translated headlines.
for translation in translated_headlines:
    print(translation)
```

### Results
The two methods provided translations of the input headlines from English to French. Both methods leverage the `t5-base` model but differ in the level of abstraction and manual control offered to the user.

### Notes
- **Environment**: The code assumes it is run in a Google Colab environment, where certain configurations like Hugging Face authentication tokens are not necessary for public models.
- **Performance**: The `pipeline` method is generally simpler and more user-friendly for straightforward translation tasks, whereas the manual method offers more control over the translation process.

### References
For more information, you can visit the official [Hugging Face Transformers documentation](https://huggingface.co/transformers/).

from datasets import load_dataset
import pandas as pd
import numpy as np

import torch
from transformers import create_optimizer, AdamWeightDecay
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from transformers import pipeline

import evaluate

from huggingface_hub import notebook_login

notebook_login()

# Install dataset
cnn_dailymail = load_dataset("cnn_dailymail", "3.0.0")

# Data preprocessing
# Load model from Hugging Face
checkpoint = "buianh0803/text-sum-2"
# Tokenizer splits the paragraph into individual sentences
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# print(tokenizer.tokenize("Don't you love 🤗 Transformers? We sure do."))

def preprocess_function(data):
    prefix = "summarize: "
    inputs = [prefix + doc for doc in data["article"]]

    model_inputs = tokenizer(
        inputs,
        max_length=1024,
        truncation=True,
    )

    labels = tokenizer(
        text_target=data["highlights"],
        max_length=128,
        truncation=True,
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = cnn_dailymail.map(preprocess_function, batched=True)
print(tokenized_datasets["train"][0])

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=checkpoint,
)

rouge = evaluate.load("rouge")

# Remove tokenizer padding in the labels
LABEL_PAD_TOKEN_ID = -100

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(
        predictions,
        skip_special_tokens=True
    )

    labels = np.where(
        labels != LABEL_PAD_TOKEN_ID,
        labels,
        tokenizer.pad_token_id
    )

    decoded_labels = tokenizer.batch_decode(
        labels,
        skip_special_tokens=True
    )

    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
BATCH_SIZE = 16

# Set training parameters
training_args = Seq2SeqTrainingArguments(
    output_dir="text-sum-3",
    evaluation_strategy="epoch",
    learning_rate=1e-3,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=5,
    fp16=True,
    predict_with_generate=True,
    push_to_hub=True,
)

# Set up the trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.push_to_hub()

text = cnn_dailymail["validation"][7]
text["article"] = "summarize: " + text["article"]
text["article"]

highlight = " ".join(text["highlights"].split("\n"))
print(highlight)

text = """
summarize: New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.
Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other.
In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage.
Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the
2010 marriage license application, according to court documents.
Prosecutors said the marriages were part of an immigration scam.
On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further.
After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective
Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.
All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say.
Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.
Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted.
The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\'s
Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali.
Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force.
If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18.
"""

summarizer = pipeline("summarization", model="buianh0803/text-sum-2")
summarizer(text["article"])

rouge = evaluate.load("rouge")

predictions = ["Prince and 3rdEyeGirl are bringing the Hit & Run Tour to the U.S. for the first time . Tickets will go on sale Monday, March 9 at 10 a.m."]
references = ["It will be a first time for the tour stateside. First show will be in Louisville, Kentucky."]

result = rouge.compute(
    predictions=predictions,
    references=references,
    use_aggregator=True,
    use_stemmer=True
)

print(result)
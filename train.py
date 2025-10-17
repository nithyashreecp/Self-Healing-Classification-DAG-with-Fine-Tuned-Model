# train.py
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

def preprocess_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

def main():
    model_name = "distilbert-base-uncased"
    dataset = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_train = dataset["train"].map(lambda x: preprocess_function(x, tokenizer), batched=True)
    tokenized_test = dataset["test"].map(lambda x: preprocess_function(x, tokenizer), batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    args = TrainingArguments(
        output_dir="saved_model",
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_dir="./logs",
        logging_steps=50
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
    )

    trainer.train()
    model.save_pretrained("saved_model")
    tokenizer.save_pretrained("saved_model")

if __name__ == "__main__":
    main()

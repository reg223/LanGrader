# import pandas as pd
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
# import torch
# from torch.utils.data import Dataset, DataLoader
# from datasets import load_dataset

# model_path = "microsoft/deberta-v3-small"

# # Define a custom dataset class
# class ArgumentDataset(Dataset):
#     def __init__(self, txt, labels):
#         self.text = txt
#         self.topic = topic
#         self.stance = stance
#         self.label = labels # the evaluation score

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         label = self.label[idx]
#         text = self.text[idx]
#         stance = self.stance[idx]
#         topic = self
#         argument = self.df.iloc[idx]['argument'] + ' ' + self.df.iloc[idx]['topic']
#         stance = self.df.iloc[idx]['stance_WA']
#         evaluation = self.df.iloc[idx]['WA']


#         # Tokenize the argument
#         max_length = 512  # Set the maximum sequence length

#         tokenized_text = self.tokenizer(argument, max_length=max_length , padding='max_length', truncation=True, return_tensors="pt")

#         return {
#             'input_ids': tokenized_text['input_ids'].flatten(),
#             'attention_mask': tokenized_text['attention_mask'].flatten(),
#             'stance': torch.tensor(stance, dtype=torch.float),
#             'evaluation': torch.tensor(evaluation, dtype=torch.float)
#         }

# # Tokenization
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# # batch_size = 32
# # Load your dataset
# train_path = "./LC_practice/group_train.csv"
# val_path = "./LC_practice/group_test.csv"


# # Create a custom dataset instance

# def preprocess(ds):
#   text = f"{ds['argument']}.\n{ds['topic']}"
#   tokenized_text = tokenizer(text, truncation=True)
#   tokenized_text['labels'] = ds['WA']

#   return tokenized_text

# train = load_dataset('csv', data_files=train_path)
# val = load_dataset('csv', data_files=val_path)



# def makeDs(fpath):
#   df = pd.read_csv(fpath)
#   return ArgumentDataset(tokenizer, df)
#   # return DataLoader(ArgumentDataset(tokenizer, df), batch_size=batch_size, shuffle=True)

# # Create a DataLoader for batch processing



# train_ds = train.map(preprocess, batched=True, batch_size=32)
# val_ds = val.map(preprocess, batched=True, batch_size=32)

# import evaluate
# import numpy as np

# clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

# # def sigmoid(x):
# #    return 1/(1 + np.exp(-x))
# from transformers import DataCollatorWithPadding

# data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# def compute_metrics(eval_pred):

#    predictions, labels = eval_pred
#   #  predictions = sigmoid(predictions)
#    predictions = (predictions > 0.5).astype(int).reshape(-1)
#    return clf_metrics.compute(predictions=predictions, references=labels.astype(int).reshape(-1))




# model = AutoModelForSequenceClassification.from_pretrained(
#     model_path,
#     # num_labels=num_labels,
#     # id2label=id2label,
#     # label2id=label2id,
#     problem_type="multi_label_classification"
# )

# training_args = TrainingArguments(
#     output_dir="./output",
#     learning_rate=2e-5,
#     num_train_epochs=3,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     weight_decay=0.01,
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     load_best_model_at_end=True,
# )

# trainer = Trainer(

#    model=model,
#    args=training_args,
#    train_dataset=train_ds,
#    eval_dataset=val_ds,
#    tokenizer=tokenizer,
#    data_collator=data_collator,
#    compute_metrics=compute_metrics,
# )

# trainer.train()


from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset



model_path = 'microsoft/deberta-v3-small'
max_length = 32
train_path = "./group_train.csv"
val_path = "./group_test.csv"


tokenizer = AutoTokenizer.from_pretrained(model_path)
train = load_dataset('csv', data_files=train_path, split = 'train')
val = load_dataset('csv', data_files=val_path, split = 'train')
val = val.train_test_split(test_size=6314)['test']


def preprocess(ds):
  text = f"{ds['argument']}.\n{ds['topic']}"
  tokenized_text = tokenizer(text , max_length=max_length , padding='max_length', truncation=True)
  tokenized_text['label'] = ds['WA'],ds['stance_WA']

  return tokenized_text

# train = ds['train']
train = train.map(preprocess)
# val = ds['test']
val = val.map(preprocess)


import evaluate
import numpy as np

clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

def sigmoid(x):
   return 1/(1 + np.exp(-x))

def compute_metrics(eval_pred):

   predictions, labels = eval_pred
   predictions = sigmoid(predictions)
   predictions = (predictions > 0.5).astype(float).reshape(-1)
   return clf_metrics.compute(predictions=predictions, references=labels.astype(float).reshape(-1))
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import transformers
from sklearn.utils.class_weight import compute_class_weight  # compute the class weights
from torch.utils.data import RandomSampler  # define a batch size
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from torchinfo import summary
from transformers import (
    AdamW,
    AutoModel,
    BertTokenizerFast,
    RobertaModel,
    RobertaTokenizer,
)

from .llamabot import llama_response


class BERT_Arch(nn.Module):
    def __init__(self, bert, size):
        super(BERT_Arch, self).__init__()
        self.bert = bert

        # dropout layer
        self.dropout = nn.Dropout(0.2)

        # relu activation function
        self.leaky_relu = nn.LeakyReLU()  # dense layer
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, size)  # softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)  # define the forward pass

    def forward(self, sent_id, mask):
        cls_hs = self.bert(sent_id, attention_mask=mask)[0][:, 0]
        x = self.fc1(cls_hs)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)  # output layer
        x = self.fc3(x)

        # apply softmax activation
        x = self.softmax(x)
        return x


class ChatbotModel:

    def __init__(self):
        self.device = torch.device("cpu")

        # Import BERT-base pretrained model
        self.tokenizer_bert = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.bert = AutoModel.from_pretrained("bert-base-uncased")

        self.load_data()
        self.load_models()

    def load_data(self):
        self.para_df = pd.read_csv("app/data/paragraph_qa.csv")
        self.para_df.insert(0, "class", range(0, 0 + len(self.para_df)))

        self.table_df = pd.read_csv("app/data/table_qa.csv")
        self.table_df.insert(0, "class", range(0, 0 + len(self.table_df)))

    def load_models(self):
        self.para_model = BERT_Arch(self.bert, size=88)
        self.para_model.load_state_dict(
            torch.load(
                "app/models/bert_para.pth", weights_only=True, map_location=self.device
            )
        )
        self.para_model.eval()

        self.table_model = BERT_Arch(self.bert, size=305)
        self.table_model.load_state_dict(
            torch.load(
                "app/models/bert_table.pth", weights_only=True, map_location=self.device
            )
        )
        self.table_model.eval()

    def predict(self, source, text):
        if source == "table":
            return self.predict_(self.table_model, self.table_df, text)
        elif source == "text":
            return self.predict_(self.para_model, self.para_df, text)
        elif source == "llama":
            return ""
            # return llama_response(text)

    def predict_(self, model, df, text):
        try:
            text = re.sub(r"[^a-zA-Z ]+", "", text)
            test_text = [text]
            tokens_test_data = self.tokenizer_bert(
                test_text,
                max_length=8,
                padding=True,
                truncation=True,
                return_token_type_ids=False,
            )
            test_seq = torch.tensor(tokens_test_data["input_ids"])
            test_mask = torch.tensor(tokens_test_data["attention_mask"])
            preds = None

            with torch.no_grad():
                preds = model(test_seq.to(self.device), test_mask.to(self.device))

            preds = preds.detach().cpu().numpy()
            preds = np.argmax(preds, axis=1)

            intent = preds[0]
            answer = df.loc[df["class"] == intent, "answer"].values[0]
            return answer

        except Exception as ex:
            print(ex)
            return "Please try again later."


# ChatbotModel().predict("table", "What was the total revenue for July 31 2024?")

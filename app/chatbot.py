import pandas as pd
import torch
import torch.nn as nn
import transformers
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler#define a batch size
from torchinfo import summary
from transformers import AutoModel, BertTokenizerFast
from transformers import RobertaTokenizer, RobertaModel
import torch.optim.lr_scheduler as lr_scheduler
import re
from sklearn.utils.class_weight import compute_class_weight#compute the class weights
import numpy as np
from transformers import AdamW

class BERT_Arch(nn.Module):
  def __init__(self, bert):
    super(BERT_Arch, self).__init__()
    self.bert = bert

       # dropout layer
    self.dropout = nn.Dropout(0.2)

       # relu activation function
    self.leaky_relu =  nn.LeakyReLU()       # dense layer
    self.fc1 = nn.Linear(768,512)
    self.fc2 = nn.Linear(512,256)
    self.fc3 = nn.Linear(256,88)       #softmax activation function
    self.softmax = nn.LogSoftmax(dim=1)       #define the forward pass

  def forward(self, sent_id, mask):
    cls_hs = self.bert(sent_id, attention_mask=mask)[0][:,0]
    x = self.fc1(cls_hs)
    x = self.leaky_relu(x)
    x = self.dropout(x)
    x = self.fc2(x)
    x = self.leaky_relu(x)
    x = self.dropout(x)      # output layer
    x = self.fc3(x)

      # apply softmax activation
    x = self.softmax(x)
    return x


class ChatbotModel:

  def __init__(self):
    self.device = torch.device('cpu')

    self.load_data()
    self.load_model()

  def load_data(self):
    self.df = pd.read_csv('app/paragraph_question_answer.csv')
    self.df = self.df.drop(['Unnamed: 0.1', 'Unnamed: 0'], axis = 1)
    self.df.insert(0, 'class', range(0, 0 + len(self.df)))

  def load_model(self):
    self.tokenizer_bert = BertTokenizerFast.from_pretrained('bert-base-uncased')# Import BERT-base pretrained model
    bert = AutoModel.from_pretrained('bert-base-uncased')

    self.model = BERT_Arch(bert)
    self.model.load_state_dict(torch.load('app/models/bert.pth', weights_only=True, map_location=self.device))
    self.model.eval()

  def predict(self, text):
    try:
      text = re.sub(r'[^a-zA-Z ]+', '', text)
      test_text = [text]
      tokens_test_data = self.tokenizer_bert(test_text, max_length = 8, padding=True, truncation=True, return_token_type_ids=False)
      test_seq = torch.tensor(tokens_test_data['input_ids'])
      test_mask = torch.tensor(tokens_test_data['attention_mask'])
      preds = None

      with torch.no_grad():
        preds = self.model(test_seq.to(self.device), test_mask.to(self.device))

      preds = preds.detach().cpu().numpy()
      preds = np.argmax(preds, axis = 1)
      
      intent = preds[0]
      answer = self.df.loc[self.df['class'] == intent, 'answer'].values[0]
      return answer

    except Exception as ex:
      print(ex)
      return 'Please try again later.'

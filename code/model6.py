import torch
import torch.nn as nn
from transformers import RobertaForSequenceClassification

class RobertaClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense1 = nn.Linear(768*6, 512)
        self.dense2 = nn.Linear(512, 256)
        self.dense3 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, features):
        x = self.dropout(features)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense3(x)

        return x

class Model_6(RobertaForSequenceClassification):
    def __init__(self, encoder, config, args):
        super().__init__(config=config)
        self.encoder = encoder
        self.args = args
        self.classifier = RobertaClassificationHead(config)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids1, input_ids2, input_ids3, input_ids4, input_ids5, input_ids6, labels=None):
        if labels is not None:
            labels = labels.unsqueeze(-1)
            labels = labels.float()

        outputs1 = self.encoder.roberta(input_ids1, attention_mask=input_ids1.ne(1))[0]
        outputs2 = self.encoder.roberta(input_ids2, attention_mask=input_ids2.ne(1))[0]
        outputs3 = self.encoder.roberta(input_ids3, attention_mask=input_ids3.ne(1))[0]
        outputs4 = self.encoder.roberta(input_ids4, attention_mask=input_ids4.ne(1))[0]
        outputs5 = self.encoder.roberta(input_ids5, attention_mask=input_ids5.ne(1))[0]
        outputs6 = self.encoder.roberta(input_ids6, attention_mask=input_ids6.ne(1))[0]

        cls1 = outputs1[:, 0, :]
        cls2 = outputs2[:, 0, :]
        cls3 = outputs3[:, 0, :]
        cls4 = outputs4[:, 0, :]
        cls5 = outputs5[:, 0, :]
        cls6 = outputs6[:, 0, :]

        combined = torch.cat([cls1, cls2, cls3, cls4, cls5, cls6], dim=1)

        # 分类
        logits = self.classifier(combined)

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
            return loss
        else:
            return self.sigmoid(logits)
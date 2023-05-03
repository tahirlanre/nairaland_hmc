import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
import numpy as np


class BERT_STL(nn.Module):
    def __init__(self, enc_model_name_or_path, num_labels, dropout=0.2) -> None:
        super().__init__()

        self.enc_model = AutoModel.from_pretrained(enc_model_name_or_path)
        self.num_labels = num_labels

        self.classifier = nn.Linear(self.enc_model.config.hidden_size, num_labels)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.enc_model(input_ids, attention_mask=attention_mask)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        output = (logits,)

        return ((loss,) + output) if loss is not None else output


class BERT_MTL(nn.Module):
    def __init__(
        self, enc_model_name_or_path, num_labels, alpha=0.2, dropout=0.2
    ) -> None:
        super().__init__()

        self.enc_model = AutoModel.from_pretrained(enc_model_name_or_path)
        self.num_labels = num_labels
        self.alpha = alpha

        self.classifier = nn.Linear(self.enc_model.config.hidden_size, num_labels)
        self.literal_classifier = nn.Linear(self.enc_model.config.hidden_size, 2)
        self.fc = nn.Linear(
            self.enc_model.config.hidden_size, self.enc_model.config.hidden_size
        )

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.Tanh()

    def forward(
        self,
        input_ids,
        input_ids_2,
        attention_mask_2,
        target_index,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        percent_done=0,
    ):
        outputs = self.enc_model(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        pooled_output = outputs[1]

        # Get target ouput with target mask
        target_output = []
        for i, idx in enumerate(target_index):
            target_output.append(sequence_output[i, idx, :])
        target_output = torch.stack(target_output)

        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        # literal module
        target_output = self.dropout(target_output)
        target_output = self.activation(self.fc(target_output))

        literal_logits = self.literal_classifier(target_output)

        if percent_done:
            self.alpha = percent_done

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            seq_loss = loss_fn(logits, labels)

            literal_labels = (labels == 2).type(torch.LongTensor)
            literal_loss = loss_fn(literal_logits, literal_labels.cuda())

            loss = seq_loss + (self.alpha * literal_loss)
        output = (logits,)

        return ((loss,) + output) if loss is not None else output

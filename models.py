import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
import numpy as np


class BERT_CON(nn.Module):
    def __init__(
        self, enc_model_name_or_path, num_labels, lam, dropout=0.2
    ) -> None:
        super().__init__()

        self.enc_model = AutoModel.from_pretrained(enc_model_name_or_path)
        self.num_labels = num_labels
        self.lam = lam

        self.classifier = nn.Linear(768, num_labels)

        self.dropout = nn.Dropout(dropout)

    def contrastive_loss(self, embeddings, labels, gamma=1.0):
        c_loss = 0.
        for i in range(len(embeddings)):
            for j in range(len(embeddings)):
                if (i != j):
                    dist = distance.euclidean(embeddings[i], embeddings[j])
                    if labels[i] == labels[j]:
                        c_loss += dist
                    else:
                        c_loss += max(0, (gamma - dist))
        return c_loss               
                
    def forward(
        self,
        input_ids,
        input_ids_2,
        # target_mask,
        # target_mask_2,
        attention_mask_2,
        target_index,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
    ):
        outputs = self.enc_model(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        pooled_output = outputs[1]

        # Get target ouput with target mask
        target_output = []
        for i, idx in enumerate(target_index):
            target_output.append(sequence_output[i, idx, :])
        target_output = torch.stack(target_output)
        # target_output = sequence_output[:, 0, :]

        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            cross_loss = loss_fn(logits, labels)

            contrastive_l = self.contrastive_loss(
                target_output.cpu().detach().numpy(),
                labels,
            )
            loss = ((1 - self.lam) * cross_loss) + (self.lam * contrastive_l)
        output = (logits,)

        return ((loss,) + output) if loss is not None else output

class BERT_SCL(nn.Module):
    def __init__(
        self, enc_model_name_or_path, num_labels, lam, dropout=0.2
    ) -> None:
        super().__init__()

        self.enc_model = AutoModel.from_pretrained(enc_model_name_or_path)
        self.num_labels = num_labels
        self.lam = lam

        self.classifier = nn.Linear(768, num_labels)

        self.dropout = nn.Dropout(dropout)            
                
    def contrastive_loss(self, temp, embedding1, embedding2, label):
        """calculate the contrastive loss"""
        # cosine similarity between embeddings
        cosine_sim = cosine_similarity(embedding1, embedding2)
        # remove diagonal elements from matrix
        dis = cosine_sim[~np.eye(cosine_sim.shape[0], dtype=bool)].reshape(
            cosine_sim.shape[0], -1
        )
        # apply temprature to elements
        dis = dis / temp
        cosine_sim = cosine_sim / temp
        # apply exp to elements
        dis = np.exp(dis)
        cosine_sim = np.exp(cosine_sim)

        # calculate row sum
        row_sum = []
        for i in range(len(embedding1)):
            row_sum.append(sum(dis[i]))
        # calculate outer sum
        contrastive_loss = 0
        for i in range(len(embedding1)):
            n_i = label.tolist().count(label[i]) - 1
            inner_sum = 0
            # calculate inner sum
            for j in range(len(embedding1)):
                if label[i] == label[j] and i != j:
                    inner_sum = inner_sum + np.log(cosine_sim[i][j] / row_sum[i])
            if n_i != 0:
                contrastive_loss += inner_sum / (-n_i)
            else:
                contrastive_loss += 0
        return contrastive_loss

    def forward(
        self,
        input_ids,
        input_ids_2,
        # target_mask,
        # target_mask_2,
        attention_mask_2,
        target_index,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
    ):
        outputs = self.enc_model(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        pooled_output = outputs[1]

        # Get target ouput with target mask
        # target_output = sequence_output * target_mask.unsqueeze(2)
        target_output = []
        for i, idx in enumerate(target_index):
            target_output.append(sequence_output[i, idx, :])
        target_output = torch.stack(target_output)
        # target_output = sequence_output[:, 0, :]

        pooled_output = self.dropout(pooled_output)

        # Second encoder for only the target word
        outputs_2 = self.enc_model(input_ids_2, attention_mask=attention_mask_2)
        sequence_output_2 = outputs_2[0]  # [batch, max_len, hidden]

        # Get target ouput with target mask
        # target_output_2 = sequence_output_2[:, 1, :]
        target_output_2 = sequence_output_2[:, 0, :]

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            cross_loss = loss_fn(logits, labels)

            contrastive_l = self.contrastive_loss(
                self.temperature,
                target_output.cpu().detach().numpy(),
                target_output_2.cpu().detach().numpy(),
                labels,
            )
            loss = (self.lam * contrastive_l) + (1 - self.lam) * (cross_loss)
        output = (logits,)

        return ((loss,) + output) if loss is not None else output

class BERT_STL(nn.Module):
    def __init__(self, enc_model_name_or_path, num_labels, dropout=0.2) -> None:
        super().__init__()

        self.enc_model = AutoModel.from_pretrained(enc_model_name_or_path)
        self.num_labels = num_labels

        self.classifier = nn.Linear(768, num_labels)

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



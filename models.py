import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class JaguarReIDModel(nn.Module):
    def __init__(self, num_classes, embedding_dim=512, pretrained=False):
        super(JaguarReIDModel, self).__init__()
        self.backbone = timm.create_model('convnext_base', pretrained=pretrained, num_classes=0)
        in_features = self.backbone.num_features

        self.embedding = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.PReLU()
        )

        self.arcface = ArcMarginProduct(embedding_dim, num_classes, s=30.0, m=0.50)

    def forward(self, x, labels=None):
        features = self.backbone(x)
        embeddings = self.embedding(features)

        if labels is not None:
            return self.arcface(embeddings, labels)
        else:
            return F.normalize(embeddings)

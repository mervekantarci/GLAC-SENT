from sentence_transformers import SentenceTransformer
import json
import torch

"""
This file extracts Sentence-Bert embeddings of descriptions beforehand.
This process is not done during training because of computational limitations.
With minimal modification, it can be done jointly during training too
"""

# load pretrained model
model = SentenceTransformer('bert-base-nli-mean-tokens').cuda()

dii_file = open(
    '../../Desktop/Homeworks/Advanced NLP/Application Project/code/data/dii/test.description-in-isolation.json', 'r')
data = json.load(dii_file)

# extract embeddings, disregard the rest
for annotation in data['annotations']:
    sent_encoding = torch.from_numpy(model.encode(annotation[0]['text']))
    # make sure to use same name wit corresponding image
    torch.save(sent_encoding, 'dataset/testdesc/' + annotation[0]['photo_flickr_id'] + '.pt')
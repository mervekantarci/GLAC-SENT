"""
Not modified version can be found at:
https://github.com/tkim-snu/GLACNet
"""
import torch
import torch.utils.data as data
import os
import nltk
from vist import VIST

"""
Main data loader functions is presented in this file
"""


class VistDataset(data.Dataset):
    def __init__(self, image_dir, description_dir, sis_path, dii_path, vocab, transform=None):
        self.image_dir = image_dir
        self.description_dir = description_dir
        self.vist = VIST(sis_file=sis_path, dii_file=dii_path)
        self.ids = list(self.vist.stories.keys())
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        vist = self.vist
        vocab = self.vocab
        story_id = self.ids[index]

        targets = []
        images = []
        descriptions = []
        photo_sequence = []
        album_ids = []

        story = vist.stories[story_id]
        for annotation in story:
            storylet_id = annotation["storylet_id"]

            # obtain image name
            # this should be the name of image feature and sentence encoding vectors
            image_id = annotation["photo_flickr_id"]
            photo_sequence.append(image_id)
            album_ids.append(annotation["album_id"])

            # load image features vector
            image = torch.load(os.path.join(self.image_dir+"/"+image_id[:2], str(image_id) + '.pt'))
            images.append(image)

            # load description encodings
            description = torch.load(os.path.join(self.description_dir+"/"+image_id[:2], str(image_id) + '.pt'))
            descriptions.append(description)

            # obtain target text
            text = annotation["text"]
            tokens = []
            try:
                tokens = nltk.tokenize.word_tokenize(text.lower())
            except Exception:
                print(text.lower() + " tokenize error")

            # obtain vocab definitions of target text
            caption = []
            caption.append(vocab('<start>'))
            caption.extend([vocab(token) for token in tokens])
            caption.append(vocab('<end>'))
            target = torch.Tensor(caption)
            targets.append(target)

        return torch.stack(images), torch.stack(descriptions), targets, photo_sequence, album_ids

    def __len__(self):
        return len(self.ids)


def collate_fn(data):

    image_stories, descriptions, caption_stories, photo_sequence_set, album_ids_set = zip(*data)

    targets_set = []
    lengths_set = []

    # zero pad all sentences to have same length
    for captions in caption_stories:
        lengths = [len(cap) for cap in captions]
        targets = torch.zeros(len(captions), max(lengths)).long()
        for i, cap in enumerate(captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]

        targets_set.append(targets)
        lengths_set.append(lengths)

    return image_stories, descriptions, targets_set, lengths_set, photo_sequence_set, album_ids_set


def get_loader(root, desc_dir, sis_path, dii_path, vocab, transform, batch_size, shuffle, num_workers):
    # load VIST dataset
    vist = VistDataset(image_dir=root, description_dir=desc_dir, sis_path=sis_path, dii_path=dii_path, vocab=vocab, transform=transform)

    # load items, one item = one story (e.g. a sequence of images with descriptions and stories)
    data_loader = torch.utils.data.DataLoader(dataset=vist, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    return data_loader

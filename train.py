"""
Not modified version can be found at:
https://github.com/tkim-snu/GLACNet
"""
import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader
from build_vocab import Vocabulary
from model import EncoderStory, DecoderStory
from torch.autograd import Variable

"""
This file needs to be run to start training. Before that,
setupNltk.py
buildVocab.py
extractResNet.py
extractSentences.py

should be run successfully.
"""


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Load vocabulary wrapper.
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build data loader
    train_data_loader = get_loader(args.train_image_dir, args.train_desc_dir, args.train_sis_path, args.train_dii_path, vocab, None, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_data_loader = get_loader(args.val_image_dir, args.val_desc_dir, args.val_sis_path, args.val_dii_path, vocab, None, args.batch_size, shuffle=False, num_workers=args.num_workers)

    # this should align with the dimensions of sentence encodings
    img_feature_size = 768
    encoder = EncoderStory(img_feature_size, args.hidden_size, args.num_layers)
    decoder = DecoderStory(args.embed_size, args.hidden_size, vocab)

    # this is used when continuing training from a checkpoint
    pretrained_epoch = 0
    if args.pretrained_epoch > 0:
        pretrained_epoch = args.pretrained_epoch
        encoder.load_state_dict(torch.load('./models/encoder-' + str(pretrained_epoch) + '.pkl'))
        decoder.load_state_dict(torch.load('./models/decoder-' + str(pretrained_epoch) + '.pkl'))

    # enabling cuda is highly recommended
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

        print("Cuda is enabled...")

    criterion = nn.CrossEntropyLoss()
    params = decoder.get_params() + encoder.get_params()
    optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)

    total_train_step = len(train_data_loader)
    total_val_step = len(val_data_loader)

    # early stop is possible, when overfit warn reaches 5
    # this is calculated over validation set
    min_avg_loss = float("inf")
    overfit_warn = 0

    logfile = open(args.log_file_path, "w")
    for epoch in range(args.num_epochs):
        # TRIANING
        if epoch < pretrained_epoch:
            continue

        encoder.train()
        decoder.train()
        avg_loss = 0.0

        for bi, (image_stories, description_stories, targets_set, lengths_set, photo_squence_set, album_ids_set) in enumerate(train_data_loader):
            decoder.zero_grad()
            encoder.zero_grad()
            loss = 0

            # ENCODER PART
            images = to_var(torch.stack(image_stories))
            descriptions = to_var(torch.stack(description_stories))

            # feed image features and sentence encodings to obtain features of sequence
            features, _ = encoder(images, descriptions)

            # DECODER PART
            for si, data in enumerate(zip(features, targets_set, lengths_set)):
                feature = data[0].cuda()
                captions = to_var(data[1])
                lengths = data[2]

                outputs = decoder(feature, captions, lengths)

                for sj, result in enumerate(zip(outputs, captions, lengths)):
                    loss += criterion(result[0], result[1][0:result[2]])

            avg_loss += loss.item()
            loss /= (args.batch_size * 5)
            loss.backward()
            optimizer.step()

            # Print log info
            if bi % args.log_step == 0:
                print('Epoch [%d/%d], Train Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
                      % (epoch + 1, args.num_epochs, bi, total_train_step,
                         loss.item(), np.exp(loss.item())))
                logfile.write('Epoch [%d/%d], Train Step [%d/%d], Loss: %.4f, Perplexity: %5.4f\n'
                      % (epoch + 1, args.num_epochs, bi, total_train_step,
                         loss.item(), np.exp(loss.item())))
                logfile.flush()

        avg_loss /= (args.batch_size * total_train_step * 5)
        print('Epoch [%d/%d], Average Train Loss: %.4f, Average Train Perplexity: %5.4f' % (
        epoch + 1, args.num_epochs, avg_loss, np.exp(avg_loss)))

        # Save the models
        torch.save(decoder.state_dict(), os.path.join(args.model_path, 'decoder-%d.pkl' % (epoch + 1)))
        torch.save(encoder.state_dict(), os.path.join(args.model_path, 'encoder-%d.pkl' % (epoch + 1)))

        # Validation
        encoder.eval()
        decoder.eval()
        avg_loss = 0.0
        for bi, (image_stories, description_stories, targets_set, lengths_set, photo_sequence_set, album_ids_set) in enumerate(
                val_data_loader):
            loss = 0

            # ENCODER PART
            images = to_var(torch.stack(image_stories))
            descriptions = to_var(torch.stack(description_stories))

            # feed image features and sentence encodings to obtain features of sequence
            features, _ = encoder(images, descriptions)

            # DECODER PART
            for si, data in enumerate(zip(features, targets_set, lengths_set)):
                feature = data[0]
                captions = to_var(data[1])
                lengths = data[2]

                outputs = decoder(feature, captions, lengths)

                for sj, result in enumerate(zip(outputs, captions, lengths)):
                    loss += criterion(result[0], result[1][0:result[2]])

            avg_loss += loss.item()
            loss /= (args.batch_size * 5)

            # Print log info
            if bi % args.log_step == 0:
                print('Epoch [%d/%d], Val Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
                      % (epoch + 1, args.num_epochs, bi, total_val_step,
                         loss.item(), np.exp(loss.item())))
                logfile.write('Epoch [%d/%d], Val Step [%d/%d], Loss: %.4f, Perplexity: %5.4f\n'
                      % (epoch + 1, args.num_epochs, bi, total_val_step,
                         loss.item(), np.exp(loss.item())))
                logfile.flush()

        avg_loss /= (args.batch_size * total_val_step * 5)
        print('Epoch [%d/%d], Average Val Loss: %.4f, Average Val Perplexity: %5.4f' % (
        epoch + 1, args.num_epochs, avg_loss, np.exp(avg_loss)))

        # Termination Condition
        overfit_warn = overfit_warn + 1 if (min_avg_loss < avg_loss) else 0
        min_avg_loss = min(min_avg_loss, avg_loss)

        if overfit_warn >= 5:
            break

    logfile.close()


if __name__ == '__main__':
    # this is needed if you use multiple threads
    torch.multiprocessing.set_start_method('spawn')

    # arguments are given along with their help info, please see them
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/',
                        help='path for saving trained models')
    parser.add_argument('--image_size', type=int, default=224,
                        help='size for input images')
    parser.add_argument('--vocab_path', type=str, default='./models/vocab.pkl',
                        help='path for vocabulary wrapper - output of buildvocab.py')
    parser.add_argument('--train_image_dir', type=str, default='./data/train',
                        help='directory for resized train images ResNet features')
    parser.add_argument('--val_image_dir', type=str, default='./data/val',
                        help='directory for resized val images ResNet features')
    parser.add_argument('--train_desc_dir', type=str, default='./dataset/traindesc',
                        help='directory for train description encodings')
    parser.add_argument('--val_desc_dir', type=str, default='./dataset/valdesc',
                        help='directory for val description encodings')
    parser.add_argument('--train_sis_path', type=str,
                        default='./data/sis/train.story-in-sequence.json',
                        help='path for train sis json file')
    parser.add_argument('--val_sis_path', type=str,
                        default='./data/sis/val.story-in-sequence.json',
                        help='path for val sis json file')
    parser.add_argument('--train_dii_path', type=str,
                        default='./data/dii/train.description-in-isolation.json',
                        help='path for train dii json file')
    parser.add_argument('--val_dii_path', type=str,
                        default='./data/dii/val.description-in-isolation.json',
                        help='path for val dii json file')
    parser.add_argument('--log_step', type=int, default=20,
                        help='step size for printing log info')
    parser.add_argument('--embed_size', type=int, default=256,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=1024,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers in lstm')
    parser.add_argument('--log_file_path', type=str, default="logfile.txt",
                        help='log')

    parser.add_argument('--pretrained_epoch', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    args = parser.parse_args()
    print(args)
    main(args)
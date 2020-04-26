import os
import time
import numpy as np
import h5py
import json
import torch
import pickle
from scipy.misc import imread, imresize
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample
from nltk.translate.bleu_score import corpus_bleu
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import SmoothingFunction

def init_embedding(embeddings):
    """
    Fills embedding tensor with values from the uniform distribution.

    :param embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def load_embeddings(emb_file, word_map):
    """
    Creates an embedding tensor for the specified word map, for loading into the model.

    :param emb_file: file containing embeddings (stored in GloVe format)
    :param word_map: word map
    :return: embeddings in the same order as the words in the word map, dimension of embeddings
    """

    # Find embedding dimension
    with open(emb_file, 'r') as f:
        emb_dim = len(f.readline().split(' ')) - 1

    vocab = set(word_map.keys())

    # Create tensor to hold embeddings, initialize
    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    init_embedding(embeddings)

    # Read embedding file
    print("\nLoading embeddings...")
    for line in open(emb_file, 'r'):
        line = line.split(' ')

        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

        # Ignore word if not in train_vocab
        if emb_word not in vocab:
            continue

        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, emb_dim


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(epoch, encoder, decoder, encoder_optimizer, decoder_optimizer,
                    bleu4):
    """
    Saves model checkpoint.
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'bleu-4': bleu4,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = 'Checkpoint_Concat_' + str(epoch) + '.pth.tar'
    torch.save(state, filename)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def validate(val_loader,encoder, decoder, criterion,best_bleu,args):    
    tophones_path = args.data_path + '/' + 'tophone_dic.pickle'
    with open(tophones_path,'rb') as f:
        tophone = pickle.load(f)
        
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    encoder.eval()
    decoder.eval()  # eval mode (no dropout or batchnorm)
    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = []  # references (true captions) for calculating BLEU-4 score
    hypotheses = []  # hypotheses (predictions)
    save_references =[]
    save_hypotheses = []

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    cc = SmoothingFunction()

    with torch.no_grad():
        # Batchesimgs, caps, caplens
        for i, (imgs, caps, caplens) in enumerate(val_loader):

            # Move to device, if available
            if args.CUDA:
                imgs = imgs.cuda()
                caps = caps.cuda().long()
                caplens = caplens.cuda() 
            else:
                caps= caps.long()   
            
            embedded_img = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind, recover_ind = decoder(embedded_img, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:,1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]           
            """
            # Calculate loss
            loss = criterion(scores, targets.squeeze(1))

            # Add doubly stochastic attention regularization
            loss += args.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % args.print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))
            """

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            # allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            """
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j][:, 1:]              
                references.append(img_caps)
            """
            caps_n = caps_sorted[:,1:]          
            
            batch_caps = caps_sorted[:,1:].squeeze(-1).cpu()    #.tolist()    
            batch_caps = batch_caps[recover_ind,:].tolist()  
            lengths = np.array(decode_lengths)[recover_ind.data.cpu()].tolist()   
            for i in range(len(batch_caps)):
                cap = batch_caps[i]
                length = lengths[i]               
                phone_cap = [tophone[str(num)] for num in cap if num!=0 and num!=52 and num!=53]    #[:length] 
                new_phone_cap = [tophone[str(num)] + ' ' for num in cap if num!=0 and num!=52 and num!=53]  #[:length] 
                # start_indx = phone_cap.index('<start>')+1
                # end_indx = phone_cap.index('<end>')-1        
                # phone_cap = phone_cap[start_indx:end_indx]
                # new_phone_cap = new_phone_cap[start_indx:end_indx]

                references.append([phone_cap])
                save_references.append(new_phone_cap)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            # preds = preds.tolist()
            preds = preds[recover_ind].tolist()
            new_preds=[]
            save_preds=[]
            for pred in preds:
                new_pred = [tophone[str(num)] for num in pred if num!=0 and num!=52 and num!=53] #[:length] 
                save_pred = [tophone[str(num)] + ' ' for num in pred if num!=0 and num!=52 and num!=53]  #[:length] 
                # start_indx = np.where(new_pred==0)+1
                # start_indx = new_pred.index('<start>')+1 
                # end_indx = np.where(new_pred==52)-1
                # end_indx = -1
                # try:
                #     end_indx = new_pred.index('<end>') - 1
                # except:
                #     pass
                # new_pred = new_pred[start_indx:end_indx]
                # save_pred = save_pred[start_indx:end_indx]
                hypotheses.append(new_pred)
                save_hypotheses.append(save_pred)


            # temp_preds = list()
            # for j, p in enumerate(preds):
            #     temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            # preds = temp_preds
            # hypotheses.extend(new_preds)
            # save_hypotheses.extend(save_preds)

            assert len(references) == len(hypotheses)

        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)  #,weights=(1, 0, 0, 0)   #,smoothing_function=cc.method5
        if bleu4>=best_bleu:
            save_ref = args.save_path + '/' + 'references.text'
            save_hyp = args.save_path + '/' + 'hypotheses.text'
            file_write_hyp = open(save_hyp, 'w')
            for var in save_hypotheses:
                file_write_hyp.writelines(var)
                file_write_hyp.write('\n')
            file_write_hyp.close()

            file_write_ref = open(save_ref, 'w')
            for var in save_references:
                file_write_ref.writelines(var)
                file_write_ref.write('\n')
            file_write_ref.close()
                        

        # out put the predicted results
        # f = 

        """
        info = 'LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu4)
        print(loss)

        with open(save_path, "a") as file:
            file.write(info)
        """

    return bleu4

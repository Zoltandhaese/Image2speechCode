import time
import torch.backends.cudnn as cudnn
import torch.optim
import pickle
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, DecoderWithAttention
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#'/media/shawn/data/Data/birds'
#'/tudelft.net/staff-bulk/ewi/insy/MMC/xinsheng/data/birds'
parser.add_argument('--CUDA', default= True)
parser.add_argument('--data_path', type = str, default='dataset') #
parser.add_argument('--save_path',type = str, default='results')
parser.add_argument('--word_map_file',type = str, default = 'dataset/caps_dic.pickle')
parser.add_argument('--emb_dim', type = int, default= 512)
parser.add_argument('--attention_dim', type = int, default= 512)
parser.add_argument('--decoder_dim',type=int,default=512)
parser.add_argument('--dropout', type = float, default= 0.2)

parser.add_argument('--start_epoch', type = int, default= 0)
parser.add_argument('--epochs', type = int, default= 10,help=' number of epochs to train for (if early stopping is not triggered)')
parser.add_argument('--batch_size', type = int, default= 60)

parser.add_argument('--workers', type = int, default= 0, help='for data-loading')
parser.add_argument('--encoder_lr', default=1e-4, type=float)
parser.add_argument('--decoder_lr', default=4e-4, type=float)
parser.add_argument('--grad_clip', default=5., type=float)
parser.add_argument('--alpha_c', default=1.0, type=float)

parser.add_argument('--print_freq', type = int, default= 100)

parser.add_argument('--input_feature_dim',type = int, default= 512)
parser.add_argument('--encoder_hidden_dim',type = int, default= 256)
parser.add_argument('--encoder_layer',type = int, default= 2)
parser.add_argument('--rnn_unit', default= 'LSTM')
parser.add_argument('--dropout_rate',type = float, default= 0.0)

parser.add_argument('--resume',default=False)
parser.add_argument('--encoder_path',default='',help='encoder parameter path')
parser.add_argument('--decoder_path',default='',help='decoder parameter path')

args = parser.parse_args()

print(args)

def main(args):
    """
    Training and validation.
    """
    save_path = args.save_path + 'results.text'
    with open(args.word_map_file, 'rb') as f:
        word_map = pickle.load(f)
    encoder = Encoder(input_feature_dim=args.input_feature_dim,
                        encoder_hidden_dim=args.encoder_hidden_dim,
                        encoder_layer=args.encoder_layer,
                        rnn_unit=args.rnn_unit,
                        use_gpu=args.CUDA,
                        dropout_rate=args.dropout_rate
                        )
    decoder = DecoderWithAttention(attention_dim=args.attention_dim,
                                    embed_dim=args.emb_dim,
                                    decoder_dim=args.decoder_dim,
                                    vocab_size=len(word_map),
                                    dropout=args.dropout)
    

    if args.resume:
        encoder.load_state_dict(torch.load(args.encoder_path))
        decoder.load_state_dict(torch.load(args.decoder_path))


    encoder_parameter = [p for p in encoder.parameters() if p.requires_grad] # selecting every parameter.
    decoder_parameter = [p for p in decoder.parameters() if p.requires_grad]
    # trainable_parameter = encoder_parameter + decoder_parameter
    encoder_optimizer = torch.optim.Adam(encoder_parameter,lr=args.decoder_lr)
    decoder_optimizer = torch.optim.Adam(decoder_parameter,lr=args.decoder_lr)
    # decoder_optimizer = torch.optim.Adam(
    # params=filter(lambda p: p.requires_grad, decoder.parameters()),
    #    
    # 
    #                                       lr=args.decoder_lr)
    if args.CUDA:
        decoder = decoder.cuda()    
        encoder = encoder.cuda()

    # Loss function    
    if args.CUDA:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss() #gewoon naar cpu dan
    
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(args.data_path, split='TRAIN'),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=pad_collate_train,pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(args.data_path,split='VAL'),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers,collate_fn=pad_collate_train, pin_memory=True)

    # Epochs
    best_bleu4 = 0
    for epoch in range(args.start_epoch, args.epochs):
        
        loss = train(train_loader=train_loader, 
              encoder = encoder,              
              decoder=decoder,
              criterion=criterion,  
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,           
              epoch=epoch,args=args)

        # One epoch's validation
        if epoch%1==0:
            bleu4 = validate(val_loader=val_loader,   
                            encoder=encoder,
                            decoder=decoder,
                            criterion=criterion,
                            best_bleu=best_bleu4,
                            args=args)

        info = 'LOSS - {loss:.4f}, BLEU-4 - {bleu:.4f}\n'.format(
                loss=loss,
                bleu=bleu4)

        print(info)

        with open(save_path, "a") as file:
            file.write(info)    
       
        if bleu4>best_bleu4:
            best_bleu4=bleu4
            # torch.save(encoder.state_dict(),'model/encoder.pth')
            # torch.save(decoder.state_dict(),'model/decoder.pth')
            save_checkpoint(epoch, encoder, decoder, encoder_optimizer,
                            decoder_optimizer, bleu4)



            

def train(train_loader,encoder, decoder, criterion,encoder_optimizer,decoder_optimizer, epoch,args):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """
    torch.set_grad_enabled(True) #Why question 
    encoder.train()
    decoder.train()


    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    for i, (imgs, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        if args.CUDA:
            imgs = imgs.cuda()
            caps = caps.cuda().long()
            caplens = caplens.cuda() 
        else:
            caps= caps.long()     
       
        encoded_imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind, recover_ind = decoder(encoded_imgs, caps, caplens) #A lot of information coming from the decoder.

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

        # Calculate loss
        loss = criterion(scores, targets.squeeze(1))

        # Add doubly stochastic attention regularization
        loss += args.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back prop.
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
       
        loss.backward()

        # # Clip gradients
        # if grad_clip is not None:
        #     clip_gradient(decoder_optimizer, grad_clip)            
        # # Update weights
        encoder_optimizer.step()
        decoder_optimizer.step()        

        # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))
    return loss






if __name__ == '__main__':
    main(args)

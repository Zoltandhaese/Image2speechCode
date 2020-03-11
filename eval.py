import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm

# Parameters
batch_size = 1
workers = 0

data_path = 'dataset'  # folder with data files saved by create_input_files.py
# data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files
checkpoint = 'checkpoint_6.pth.tar'  # model checkpoint
word_map_file =  data_path + '/' + 'caps_dic.pickle' # word map, dus word => index, ensure it's the same the data was encoded with and the model was trained with
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

tophones_path = data_path + '/' + 'tophone_dic.pickle' 
with open(tophones_path,'rb') as f:
    tophone = pickle.load(f)# to phone: getal=> naar een phoneem

# Load model
checkpoint = torch.load(checkpoint)
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()

# Load word map (word2ix)
with open(word_map_file, 'rb') as f:
    word_map = pickle.load(f)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

def evaluate(beam_size):
    """
    Evaluation

    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    loader = torch.utils.data.DataLoader(
        CaptionDataset(data_path,split='VAL'),
        batch_size=batch_size, shuffle=False, num_workers=workers,collate_fn=pad_collate_train, pin_memory=True)

    references = list()
    hypotheses = list()

    # For each image
    for i, (image, caps, caplens) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):

        k = beam_size

        # Move to GPU device, if available
        image = image.to(device)  # (1, 3, 256, 256)

        # Encode
        """
        encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)

        # Flatten encoding
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # We'll treat the problem as having a batch size of k
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)
        """
        encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
        #enc_image_size = encoder_out.size(1)
       
    
        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Tensor to store top k sequences' alphas; now they're just 1s
        #seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device) 

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h, c = decoder.init_hidden_state(encoder_out)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

            awe, _ = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

           # alpha = alpha.view(-1, enc_image_size, enc_image_size) 

            gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe

            h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

            scores = decoder.fc(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # At
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
               # print ("stap 1")
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)
            if step > 100:
                next_word_inds=torch.tensor([word_map['<end>']]).to(device)
            
            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
            #seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],dim=1)  # (s, step+1, enc_image_size, enc_image_size)
            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
               # complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            #seqs_alpha = seqs_alpha[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 100:
                break
            step += 1

        #if len(complete_seqs_scores)>0:
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i] # zal onze referentie worden.
            #alphas = complete_seqs_alpha[i]
        #else :
         #   seq=[]

        # References
        """
        img_caps = allcaps[0].tolist()
        img_captions = list(
            map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))  # remove <start> and pads
        references.append(img_captions)       """
        # Hypotheses
        phone_cap = [tophone[str(seq[i])] for i in range(1,len(seq)-1)]    #de bedoeling is dus om voor de lijst caps, dit is een lijst van nummers om te zetten naar phonemes
        new_phone_cap = [tophone[str(seq[i])] + ' ' for i in range(1,len(seq)-1)] 
        hypotheses.append(phone_cap)

        with open("test_e7_l100.txt", "a") as f:
            if (len(new_phone_cap) > 0):
                for i in range(len(new_phone_cap)-1):
                    f.write(new_phone_cap[i])
                f.write(phone_cap[-1]+ "\n")  
            else:
                f.write("no results :( " + "\n")  

        #hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])
        
        num = caps[0]
        phone_cap = [tophone[str(int(num[i].tolist()[0]))] for i in range(len(num))]    #de bedoeling is dus om voor de lijst caps, dit is een lijst van nummers om te zetten naar phonemes
        new_phone_cap = [tophone[str(int(num[i].tolist()[0]))] + ' ' for i in range(len(num))]  #[:length] 
        # start_indx = phone_cap.index('<start>')+1
        # end_indx = phone_cap.index('<end>')-1        
        # phone_cap = phone_cap[start_indx:end_indx]
        # new_phone_cap = new_phone_cap[start_indx:end_indx]

        references.append(phone_cap)
        

        assert len(references) == len(hypotheses)

    #mat = np.matrix(hypotheses)
    #with open('test.txt','wb') as f:
      #  for line in mat:
       #     np.savetxt(f, line)

    bleu4 = corpus_bleu(references, hypotheses)

    return bleu4


if __name__ == '__main__':
    beam_size = 1
    print("\nBLEU-4 score @ beam size of %d is %.4f." % (beam_size, evaluate(beam_size)))

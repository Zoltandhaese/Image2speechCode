import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm


batch_size = 1
workers = 0

data_path = 'dataset'  
checkpoint = 'Checkpoint_concat39.pth.tar'  
word_map_file =  data_path + '/' + 'caps_dic.pickle' # dus word => index
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
cudnn.benchmark = True  
hypothese_name = "hypotheses.txt" # In deze file komen de voorspelde indices terug

tophones_path = data_path + '/' + 'tophone_dic.pickle' 
with open(tophones_path,'rb') as f:
    tophone = pickle.load(f) #getal=> naar een phoneem

checkpoint = torch.load(checkpoint)
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()


with open(word_map_file, 'rb') as f:
    word_map = pickle.load(f)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

def evaluate(beam_size):
    """
    Return: text file met hypothesen
    """
    loader = torch.utils.data.DataLoader(
        CaptionDataset(data_path,split='VAL'),
        batch_size=batch_size, shuffle=False, num_workers=workers,collate_fn=pad_collate_train, pin_memory=True)

    references = list()
    hypotheses = list()
    alphas_complete = list()
    betas_complete = list()


    for i, (image, caps, caplens) in enumerate(    # For each image
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):

        k = beam_size
        image = image.to(device)  
        encoder_out = encoder(image)  
        
        enc_image_size = encoder_out.size(1)
       
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device) 
        seqs = k_prev_words  
        top_k_scores = torch.zeros(k, 1).to(device) 
        seqs_alpha = torch.ones(k, 1, enc_image_size).to(device) 
        seqs_beta = torch.ones(k, 1, 1).to(device) 

        complete_seqs = list()
        complete_seqs_scores = list()
        complete_seqs_alpha = list() #alpha coefficienten
        complete_seqs_beta = list() #beta coefficienten
 
        step = 1
        h, c = decoder.init_hidden_state(encoder_out) #initialiseren
        while True:

            embeddings = decoder.embedding(k_prev_words).squeeze(1) 
            beta =0
            awe, alpha  = decoder.attention(encoder_out, h) 

            gate = decoder.sigmoid(decoder.f_beta(h))  
            awe = gate * awe

            h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  
            scores = decoder.fc(h) 
            scores = F.log_softmax(scores, dim=1)
           
            scores = top_k_scores.expand_as(scores) + scores  

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True) 

            prev_word_inds = top_k_words / vocab_size 
            next_word_inds = top_k_words % vocab_size  
            if step > 100:
                next_word_inds=torch.tensor([word_map['<end>']]).to(device)
            
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1) 
            seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],dim=1)  
            #seqs_beta = torch.cat([seqs_beta[prev_word_inds], beta[prev_word_inds].unsqueeze(1)],dim=1) # zal de beta opslaan of niet?
        
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
                #complete_seqs_beta.extend(seqs_beta[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            seqs_alpha = seqs_alpha[incomplete_inds]
            seqs_beta = seqs_beta[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long: arbitrary choice for length 100
            if step > 100:
                break
            step += 1

        #if len(complete_seqs_scores)>0:
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        alphas = complete_seqs_alpha[i]
       # betas = complete_seqs_beta[i]
        alphas_complete.append(alphas)
       # betas_complete.append(betas)

        # Hypotheses
        phone_cap = [tophone[str(seq[i])] for i in range(1,len(seq)-1)]    #de bedoeling is dus om voor de lijst caps om te zetten naar phonemes
        new_phone_cap = [tophone[str(seq[i])] + ' ' for i in range(1,len(seq)-1)] 
        hypotheses.append(phone_cap)

        with open(hypothese_name, "a") as f:
            if (len(new_phone_cap) > 0):
                for i in range(len(new_phone_cap)-1):
                    f.write(new_phone_cap[i])
                f.write(phone_cap[-1]+ "\n")  
            else:
                f.write("no results :( " + "\n")  

        
        num = caps[0]
        phone_cap = [tophone[str(int(num[i].tolist()[0]))] for i in range(len(num))]    #de bedoeling is dus om voor de lijst caps, dit is een lijst van nummers om te zetten naar phonemes
        new_phone_cap = [tophone[str(int(num[i].tolist()[0]))] + ' ' for i in range(len(num))]  #[:length] 
        # start_indx = phone_cap.index('<start>')+1
        # end_indx = phone_cap.index('<end>')-1        
        # phone_cap = phone_cap[start_indx:end_indx]
        # new_phone_cap = new_phone_cap[start_indx:end_indx]

        references.append(phone_cap) 
        

        assert len(references) == len(hypotheses)

    #code voor alpha's op te slaan;
    #mat = np.matrix(hypotheses)
    #with open('test.txt','wb') as f:
    #    for line in mat:
    #        np.savetxt(f, line)
    #with open("Alpha_turn_reshape.txt", "wb") as fp:   #Pickling
     #       pickle.dump(alphas_complete, fp)

    finish= "Evaluatie file opgesteld!"

    return finish


if __name__ == '__main__':
    beam_size = 1
    print("\nBLEU-4 score @ beam size of %d is %.4f." % (beam_size, evaluate(beam_size)))

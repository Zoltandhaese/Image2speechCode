import torch
import math
from torch import nn
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Zoltan cuda ("mac") is not available at the moment. 


class pBLSTMLayer(nn.Module):
    def __init__(self,input_feature_dim,hidden_dim,rnn_unit='LSTM',dropout_rate=0.2):
        super(pBLSTMLayer, self).__init__()
        self.rnn_unit = getattr(nn,rnn_unit.upper())

        # feature dimension will be doubled since time resolution reduction
        self.BLSTM = self.rnn_unit(input_feature_dim*2,hidden_dim,1, bidirectional=True, 
                                   dropout=dropout_rate,batch_first=True)
    
    def forward(self,input_x):
        batch_size = input_x.size(0)
        timestep = input_x.size(1)
        feature_dim = input_x.size(2)
 
        input_x = input_x.contiguous().view(batch_size,int(timestep/2),feature_dim*2)
    
        output,hidden = self.BLSTM(input_x)
        return output,hidden

class BLSTMLayer(nn.Module):
    def __init__(self,input_feature_dim,hidden_dim,rnn_unit='LSTM',dropout_rate=0.2):
        super(BLSTMLayer, self).__init__()
        self.rnn_unit = getattr(nn,rnn_unit.upper())

       
        self.BLSTM = self.rnn_unit(input_feature_dim,hidden_dim,1, bidirectional=True, 
                                   dropout=dropout_rate,batch_first=True)
    
    def forward(self,input_x):
        batch_size = input_x.size(0)
        timestep = input_x.size(1)
        feature_dim = input_x.size(2)

        input_x = input_x.contiguous().view(batch_size,int(timestep),feature_dim)

        output,hidden = self.BLSTM(input_x)
        return output,hidden

class Encoder(nn.Module):
    def __init__(self, input_feature_dim,encoder_hidden_dim,encoder_layer,rnn_unit,use_gpu,dropout_rate, **kwargs):
        super(Encoder, self).__init__()
        # Listener RNN layer
        self.encoder_layer = encoder_layer
        assert self.encoder_layer>=1,'Listener should have at least 1 layer'
        
        self.pLSTM_layer0 = pBLSTMLayer(input_feature_dim,encoder_hidden_dim, rnn_unit=rnn_unit, dropout_rate=dropout_rate)

        for i in range(1,self.encoder_layer):
            setattr(self, 'pLSTM_layer'+str(i), pBLSTMLayer(encoder_hidden_dim*2,encoder_hidden_dim, rnn_unit=rnn_unit, dropout_rate=dropout_rate))

        ## Dit stuk code aanzetten voor de BLSTM ipv het bovenstaande
        #self.LSTM_layer0 = BLSTMLayer(input_feature_dim,encoder_hidden_dim, rnn_unit=rnn_unit, dropout_rate=dropout_rate)

        #for i in range(1,self.encoder_layer):
         #   setattr(self, 'LSTM_layer'+str(i), BLSTMLayer(encoder_hidden_dim*2,encoder_hidden_dim, rnn_unit=rnn_unit, dropout_rate=dropout_rate))

        self.use_gpu = use_gpu
        if self.use_gpu:
            self = self.cuda()

    def forward(self,input_x):
        output,_  = self.pLSTM_layer0(input_x)
        for i in range(1,self.encoder_layer):
           output, _ = getattr(self,'pLSTM_layer'+str(i))(output)

        ## Dit stuk code aanzetten voor de BLSTM ipv het bovenstaande

        #output,_  = self.LSTM_layer0(input_x)
        #for i in range(1,self.encoder_layer):
        #   output, _ = getattr(self,'LSTM_layer'+str(i))(output) # hier worden de waarden dus doorgegeven via verschillende layers
        
        return output

class EncoderT(nn.Module):
    """
    Encoder for Transformer
    """
    def __init__(self, input_feature_dim,encoder_hidden_dim,encoder_layer,rnn_unit,use_gpu,dropout_rate, nhead, **kwargs):
        super(EncoderT, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        transformerlayer = TransformerEncoderLayer(input_feature_dim, nhead, encoder_hidden_dim, dropout_rate)
        self.transformer_encoder = TransformerEncoder(transformerlayer, encoder_layer)
        self.pos_encoder = PositionalEncoding(input_feature_dim, dropout_rate)
     
    def forward(self, src):

        output = self.transformer_encoder(src)
    
        return output

class PositionalEncoding(nn.Module):
    """
    PE for transformer
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MLPAttention(nn.Module):
    """
    Attention Network. Nothing changed with the previous version. At tthis moment only MLP attender, not yet Billinear or dotattender
    """
    def __init__(self, encoder_dim, decoder_dim, attention_dim):

        super(MLPAttention, self).__init__() 
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform bipyramidal to attention input
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.sigmoid_layer = nn.Linear(decoder_dim, 1)
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU() # Relu function: this can change.
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights
        self.sigmoidB = nn.Sigmoid()

    def forward(self, encoder_out, decoder_hidden):
    
        att1 = self.encoder_att(encoder_out) 
        att2 = self.decoder_att(decoder_hidden)  
        att = self.full_att(att1+att2.unsqueeze(1)).squeeze(2) 

        alpha = self.softmax(att)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  

        return attention_weighted_encoding, alpha

class NOAttention(nn.Module):
    """
    Attention network with no attention: no information from encoder. 
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
     
        super(NOAttention, self).__init__() 
        self.full_att = nn.Linear(attention_dim, 1)  


    def forward(self, encoder_out, decoder_hidden):
       
        alpha = torch.ones([encoder_out.shape[0],encoder_out.shape[1]]).to(device) 
        alpha = alpha/49 #Enkel getest op pyramidal encoder vandaar 49. 
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1) 
        return attention_weighted_encoding, alpha

class BilinearAttention(nn.Module):
    """
    Attention Network => Billinear :X_1*A*X_2
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
    
        super(BilinearAttention, self).__init__() 
        self.Bilinear_att = nn.Bilinear(attention_dim , attention_dim, 1)
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights
       

    def forward(self, encoder_out, decoder_hidden):
     
        omzetting = decoder_hidden.unsqueeze(1).expand_as(encoder_out).contiguous()
        att = self.Bilinear_att(encoder_out,omzetting).squeeze(2)
        alpha = self.softmax(att)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  
        return attention_weighted_encoding, alpha

class ConcatAttention(nn.Module):
    """
    Attention Network => Concat :A*[X_1,X_2]
    """
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
    
        super(ConcatAttention, self).__init__() 
        self.concat_att= nn.Linear(2*attention_dim,1)
        self.full_att = nn.Linear(attention_dim, 1)  
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)  
       
    def forward(self, encoder_out, decoder_hidden):
        
        omzetting = decoder_hidden.unsqueeze(1).expand_as(encoder_out).contiguous()
        concat= torch.cat((encoder_out, omzetting), 2)
        att = self.tanh(self.concat_att(concat).squeeze(2))
        alpha = self.softmax(att)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1) 
        return attention_weighted_encoding, alpha

class Attention(nn.Module): ## do
    """
    Attention Network: DOTattender: X1*X2
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
       
        super(Attention, self).__init__() 
        self.full_att = nn.Linear(attention_dim, 1) 
        self.relu = nn.ReLU() 
        self.softmax = nn.Softmax(dim=1)  

    def forward(self, encoder_out, decoder_hidden):
        att = self.full_att(encoder_out*decoder_hidden.unsqueeze(1)).squeeze(2) 
        alpha = self.softmax(att)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return attention_weighted_encoding, alpha

class HUGOSAttention(nn.Module):
    """
    Introducing extra Beta, based on MLP attender
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(HUGOSAttention, self).__init__() 
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  
        self.sigmoid_layer = nn.Linear(decoder_dim, 1)
        self.full_att = nn.Linear(attention_dim, 1)  
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1) 
        self.sigmoidB = nn.Sigmoid()

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)  
        att2 = self.decoder_att(decoder_hidden)  
        BetaL = self. sigmoid_layer(decoder_hidden)
        BetaS = self.sigmoidB(BetaL) 
        att = self.full_att(att1 + att2.unsqueeze(1)).squeeze(2)  
        alpha = self.softmax(att)*BetaS
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  
        return attention_weighted_encoding, alpha ,BetaS

class DecoderWithAttention(nn.Module):
    """
    Decoder return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=512, dropout=0.5):
       
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = ConcatAttention(encoder_dim, decoder_dim, attention_dim)  #attention network: make choice. 

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        
        self.init_c = nn.Linear(encoder_dim, decoder_dim) 
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  

        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  

        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size
        
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  
        len_seq = encoder_out.size(1)

        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        _,recover_ind = sort_ind.sort(dim=0)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions) 

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out) 

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1        
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), len_seq).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t])) 
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :].squeeze(1), attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  
            preds = self.fc(self.dropout(h)) 
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind, recover_ind





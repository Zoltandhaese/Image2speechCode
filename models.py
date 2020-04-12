import torch
from torch import nn
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Zoltan cuda ("mac") is not available at the moment. 


class pBLSTMLayer(nn.Module):
    def __init__(self,input_feature_dim,hidden_dim,rnn_unit='LSTM',dropout_rate=0.0):
        super(pBLSTMLayer, self).__init__()
        self.rnn_unit = getattr(nn,rnn_unit.upper())

        # feature dimension will be doubled since time resolution reduction
        self.BLSTM = self.rnn_unit(input_feature_dim*2,hidden_dim,1, bidirectional=True, 
                                   dropout=dropout_rate,batch_first=True)
    
    def forward(self,input_x):
        batch_size = input_x.size(0)
        timestep = input_x.size(1)
        feature_dim = input_x.size(2)
        # Reduce time resolution
        input_x = input_x.contiguous().view(batch_size,int(timestep/2),feature_dim*2)
        # Bidirectional RNN
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

        self.use_gpu = use_gpu
        if self.use_gpu:
            self = self.cuda()

    def forward(self,input_x):
        output,_  = self.pLSTM_layer0(input_x)
        for i in range(1,self.encoder_layer):
            output, _ = getattr(self,'pLSTM_layer'+str(i))(output)
        
        return output


class MLPAttention(nn.Module):
    """
    Attention Network. Nothing changed with the previous version. At tthis moment only MLP attender, not yet Billinear or dotattender
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images=> after bipyramidal encoder this is 49 
        :param decoder_dim: size of decoder's RNN => normal 512 
        :param attention_dim: size of the attention network => 512
        """
        super(MLPAttention, self).__init__() 
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform bipyramidal to attention input
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.sigmoid_layer = nn.Linear(decoder_dim, 1)
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU() # Relu function: this can change.
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights
        self.sigmoidB = nn.Sigmoid()

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, number_of_inputs (49), encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        # we doen dit dus ook met batch sizes, de lengte van batch
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim) #DIt is onze source state 
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim) #Dit is onze target state
        att = self.full_att(self.relu(att1+att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        #att = torch.mm(att1, att2.unsqueeze(1)) # Is dit de dot attender? 
        alpha = self.softmax(att)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)
        return attention_weighted_encoding, alpha

class BilinearAttention(nn.Module):
    """
    Attention Network. Nothing changed with the previous version. At tthis moment only MLP attender, not yet Billinear or dotattender
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images=> after bipyramidal encoder this is 49 
        :param decoder_dim: size of decoder's RNN => normal 512 
        :param attention_dim: size of the attention network => 512
        """
        super(BilinearAttention, self).__init__() 
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform bipyramidal to attention input
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.extra_dim=nn.Linear(1,49)
        self.Bilinear_att = nn.Bilinear(attention_dim , attention_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU() # Relu function: this can change.
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights
       

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, number_of_inputs (49), encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim) #DIt is onze source state 
        att2 = self.decoder_att(decoder_hidden)
        #att2.expand(60,49,512)
        omzetting = att2.unsqueeze(1).expand_as(att1).contiguous()
        att = self.full_att(self.relu(self.Bilinear_att(att1,omzetting))).squeeze(2)
        #att = torch.mm(att1, att2.unsqueeze(1)) # Is dit de dot attender? 
        alpha = self.softmax(att)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)
        return attention_weighted_encoding, alpha

class Attention(nn.Module): ##dotattention
    """
    Attention Network. Nothing changed with the previous version. At tthis moment only MLP attender, not yet Billinear or dotattender
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images=> after bipyramidal encoder this is 49 
        :param decoder_dim: size of decoder's RNN => normal 512 
        :param attention_dim: size of the attention network => 512
        """
        super(Attention, self).__init__() 
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform bipyramidal to attention input
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU() # Relu function: this can change.
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, number_of_inputs (49), encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        # we doen dit dus ook met batch sizes, de lengte van batch
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim) #DIt is onze source state 
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim) #Dit is onze target state
        att = self.full_att(self.relu(att1*att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        #att = torch.mm(att1, att2.unsqueeze(1)) # Is dit de dot attender? 
        alpha = self.softmax(att)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)
        return attention_weighted_encoding, alpha

class HUGOSAttention(nn.Module):
    """
    Attention Network. Nothing changed with the previous version. At tthis moment only MLP attender, not yet Billinear or dotattender
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images=> after bipyramidal encoder this is 49 
        :param decoder_dim: size of decoder's RNN => normal 512 
        :param attention_dim: size of the attention network => 512
        """
        super(HUGOSAttention, self).__init__() 
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform bipyramidal to attention input
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.sigmoid_layer = nn.Linear(decoder_dim, 1)
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU() # Relu function: this can change.
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights
        self.sigmoidB = nn.Sigmoid()

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, number_of_inputs (49), encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """

        ##HUGO##
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim) #DIt is onze source state 
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim) #Dit is onze target state
        BetaL = self. sigmoid_layer(decoder_hidden)
        BetaS = self.sigmoidB(BetaL) # manueel de attention uitzetten
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)*BetaS
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)
        return attention_weighted_encoding, alpha ,BetaS

class DecoderWithAttention(nn.Module):
    """
    Decoder. "Still the same as the old model. We have to introduce the transformer model if we want to mimic the xnmt output.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=512, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = MLPAttention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
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
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
     """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size
        
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        len_seq = encoder_out.size(1)

        # Sort input data by decreasing lengths; +> Voor juiste ordening
        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        _,recover_ind = sort_ind.sort(dim=0)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

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
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :].squeeze(1), attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind, recover_ind

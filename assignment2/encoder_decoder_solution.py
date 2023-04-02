import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import traceback


class GRU(nn.Module):
    def __init__(
            self,
            input_size, 
            hidden_size
            ):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.w_ir = nn.Parameter(torch.empty(hidden_size, input_size))
        self.w_iz = nn.Parameter(torch.empty(hidden_size, input_size))
        self.w_in = nn.Parameter(torch.empty(hidden_size, input_size))

        self.b_ir = nn.Parameter(torch.empty(hidden_size))
        self.b_iz = nn.Parameter(torch.empty(hidden_size))
        self.b_in = nn.Parameter(torch.empty(hidden_size))

        self.w_hr = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.w_hz = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.w_hn = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.b_hr = nn.Parameter(torch.empty(hidden_size))
        self.b_hz = nn.Parameter(torch.empty(hidden_size))
        self.b_hn = nn.Parameter(torch.empty(hidden_size))
        for param in self.parameters():
            nn.init.uniform_(param, a=-(1/hidden_size)**0.5, b=(1/hidden_size)**0.5)

    def forward(self, inputs, hidden_states):
        """GRU.
        
        This is a Gated Recurrent Unit
        Parameters
        ----------
        inputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, input_size)`)
          The input tensor containing the embedded sequences. input_size corresponds to embedding size.
          
        hidden_states (`torch.FloatTensor` of shape `(1, batch_size, hidden_size)`)
          The (initial) hidden state.
          
        Returns
        -------
        outputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`)
          A feature tensor encoding the input sentence. 
          
        hidden_states (`torch.FloatTensor` of shape `(1, batch_size, hidden_size)`)
          The final hidden state. 
        """
        batch_size, seq_len, _ = inputs.shape
        inputs = torch.swapaxes(inputs, 1, 0)
        h_prev = hidden_states
        all_hidden_states = torch.zeros(seq_len, batch_size, hidden_states.shape[2])
        for t in range(seq_len):
            r_t = F.sigmoid(inputs[t] @ self.w_ir.T + self.b_ir + h_prev @ self.w_hr.T + self.b_hr)
            z_t = F.sigmoid(inputs[t] @ self.w_iz.T + self.b_iz + h_prev @ self.w_hz.T + self.b_hz)
            n_t = F.tanh(   inputs[t] @ self.w_in.T + self.b_in + torch.mul(r_t, h_prev @ self.w_hn.T + self.b_hn))
            h_prev = torch.mul(1-z_t, n_t) + torch.mul(z_t, h_prev)
            all_hidden_states[t] = h_prev
        # ==========================
        return torch.swapaxes(all_hidden_states, 1, 0), h_prev



class Attn(nn.Module):
    def __init__(
        self,
        hidden_size=256,
        dropout=0.0 # note, this is an extrenous argument
        ):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size

        self.W = nn.Linear(hidden_size*2, hidden_size)
        self.V = nn.Linear(hidden_size, hidden_size) # in the forwards, after multiplying
                                                     # do a torch.sum(..., keepdim=True), its a linear operation

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, inputs, hidden_states, mask = None):
        """Soft Attention mechanism.

        This is a one layer MLP network that implements Soft (i.e. Bahdanau) Attention with masking
        Parameters
        ----------
        inputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`)
            The input tensor containing the embedded sequences.

        hidden_states (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)
            The (initial) hidden state.

        mask ( optional `torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The masked tensor containing the location of padding in the sequences.

        Returns
        -------
        outputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`)
            A feature tensor encoding the input sentence with attention applied.

        x_attn (`torch.FloatTensor` of shape `(batch_size, sequence_length, 1)`)
            The attention vector.
        """
        try:
            batch_size, seq_len, _ = inputs.shape
            n_layers, batch_size, _ = hidden_states.shape
            inputs = torch.swapaxes(inputs, 1, 0)                       # seq_len * batch_size * hidden_size
            outputs = torch.zeros((seq_len, batch_size, 1))
            for t in range(seq_len):
                layer_t_res = torch.zeros((n_layers, batch_size, self.hidden_size))
                for l in range(n_layers):
                    concat_res = torch.cat((inputs[t], hidden_states[l]), dim=1) # 1 x batch_size x 2*hidden_size -> 1 x batch_size x hidden_size
                    tanh_res   = self.tanh(self.W(concat_res))                  # 1 x batch_size x hidden_size   -> 1 x batch_size x hidden_size
                    layer_t_res[l] = self.V(tanh_res)                # 1 x batch_size x hidden_size   -> seq_len x batch_size x hidden_size 
                outputs[t]  = torch.sum(layer_t_res, 2, keepdim=True)   # seq_len x batch_size x hidden_size -> 1 x batch_size x hidden_size
            outputs = torch.swapaxes(outputs, 1, 0)                     # seq_len x batch_size x hidden_size -> batch_size x seq_len x hidden_size
            softmaxed_outputs = self.softmax(outputs)                     # batch_size x seq_len x hidden_size -> batch_size x seq_len x hidden_size
            res = torch.mul(softmaxed_outputs, torch.swapaxes(inputs, 1, 0))
            if mask is not None:
                mask = mask.unsqueeze(-1)==1
                res = res.masked_fill_(mask, -torch.inf)
                softmaxed_outputs = softmaxed_outputs.masked_fill_(mask, -torch.inf)
            
        except Exception as e: 
            traceback.print_exc()
        return res, softmaxed_outputs

class Encoder(nn.Module):
    def __init__(
        self,
        vocabulary_size=30522,
        embedding_size=256,
        hidden_size=256,
        num_layers=1,
        dropout=0.0
        ):
        super(Encoder, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(
            vocabulary_size, embedding_size, padding_idx=0,
        )

        self.dropout = nn.Dropout(p=dropout)
        self.rnn = nn.GRU(input_size=embedding_size, num_layers = num_layers, hidden_size=hidden_size, bidirectional=True, batch_first=True)

    def forward(self, inputs, hidden_states):
        """GRU Encoder.

        This is a Bidirectional Gated Recurrent Unit Encoder network
        Parameters
        ----------
        inputs (`torch.FloatTensor` of shape `(batch_size 5, sequence_length 256)`)
            The input tensor containing the token sequences.

        hidden_states(`torch.FloatTensor` of shape `(num_layers*2 2, batch_size 5, hidden_size 128)`)
            The (initial) hidden state for the bidrectional GRU.
            
        Returns
        -------
        outputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`)
            A feature tensor encoding the input sentence. 

        hidden_states (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)
            The final hidden state. 
        """
        inputs = torch.swapaxes(inputs,0,1)
        embeddings = torch.zeros(inputs.shape[0], inputs.shape[1], self.embedding_size)
        for t in range(inputs.shape[0]):
            embeddings[t] = self.embedding(inputs[t])
        embeddings = torch.swapaxes(embeddings,0,1)
        embeddings = self.dropout(embeddings)
        rnn_res = self.rnn(embeddings, hidden_states)
        outputs = rnn_res[0][:,:,:128] + rnn_res[0][:,:,128:]
        hidden = torch.unsqueeze(rnn_res[1][0] + rnn_res[1][1], 0)
        return outputs, hidden


    def initial_states(self, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device
        shape = (self.num_layers*2, batch_size, self.hidden_size)
        # The initial state is a constant here, and is not a learnable parameter
        h_0 = torch.zeros(shape, dtype=torch.float, device=device)
        return h_0

class DecoderAttn(nn.Module):
    def __init__(
        self,
        vocabulary_size=30522,
        embedding_size=256,
        hidden_size=256,
        num_layers=1,
        dropout=0.0, 
        ):

        super(DecoderAttn, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p=dropout)

        self.rnn = nn.GRU(embedding_size, hidden_size, num_layers=self.num_layers, batch_first=True)
        
        self.mlp_attn = Attn(hidden_size, dropout)

    def forward(self, inputs, hidden_states, mask=None):
        """GRU Decoder network with Soft attention

        This is a Unidirectional Gated Recurrent Unit Decoder network
        Parameters
        ----------
        inputs (`torch.LongTensor` of shape `(batch_size, sequence_length, hidden_size)`)
            The input tensor containing the encoded input sequence.

        hidden_states(`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)
            The (initial) hidden state for the unidrectional GRU.

        mask ( optional `torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The masked tensor containing the location of padding in the sequences.

        Returns
        -------
        outputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`)
            A feature tensor encoding the input sentence. 

        hidden_states (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)
            The final hidden state. 
        """
        attn_out, _ = self.mlp_attn.forward(inputs=inputs, hidden_states=hidden_states, mask=mask)                
        gru_out, gru_hid = self.rnn(attn_out, hidden_states)
        return gru_out, gru_hid
        
        
class EncoderDecoder(nn.Module):
    def __init__(
        self,
        vocabulary_size=30522,
        embedding_size=256,
        hidden_size=256,
        num_layers=1,
        dropout = 0.0,
        encoder_only=False
        ):
        super(EncoderDecoder, self).__init__()
        self.encoder_only = encoder_only
        self.encoder = Encoder(vocabulary_size, embedding_size, hidden_size,
                num_layers, dropout=dropout)
        if not encoder_only:
          self.decoder = DecoderAttn(vocabulary_size, embedding_size, hidden_size, num_layers, dropout=dropout)
        
    def forward(self, inputs, mask=None):
        """GRU Encoder-Decoder network with Soft attention.

        This is a Gated Recurrent Unit network for Sentiment Analysis. This
        module returns a decoded feature for classification. 
        Parameters
        ----------
        inputs (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The input tensor containing the token sequences.

        mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The masked tensor containing the location of padding in the sequences.

        Returns
        -------
        x (`torch.FloatTensor` of shape `(batch_size, hidden_size)`)
            A feature tensor encoding the input sentence. 

        hidden_states (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)
            The final hidden state. 
        """
        hidden_states = self.encoder.initial_states(inputs.shape[0])
        x, hidden_states = self.encoder(inputs, hidden_states)
        if self.encoder_only:
          x = x[:, 0]
          return x, hidden_states
        x, hidden_states = self.decoder(x, hidden_states, mask)
        x = x[:, 0]
        return x, hidden_states

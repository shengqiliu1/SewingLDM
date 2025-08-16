import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Lambda

from diffusion.model.builder import MODELS

from .modules import MixerLayer
    

@MODELS.register_module()
class Dress_FSQTOKENIZER(nn.Module):
    """ 
    Args:
        tokenizer (list): Config about the tokenizer.
        input_token_num (int): Number of token in the dataset.
        input_channels (int): Number of input channels.
    """

    def __init__(self,
                 tokenizer:str =None,
                 input_token_num=39,
                 input_channels=787
                 ):
        super().__init__()

        self.input_token_num = input_token_num

        self.drop_rate = tokenizer['encoder']['drop_rate']     
        self.enc_num_blocks = tokenizer['encoder']['num_blocks']
        self.enc_hidden_dim = tokenizer['encoder']['hidden_dim']
        self.enc_token_inter_dim = tokenizer['encoder']['token_inter_dim']
        self.enc_hidden_inter_dim = tokenizer['encoder']['hidden_inter_dim']
        self.enc_dropout = tokenizer['encoder']['dropout']

        self.dec_num_blocks = tokenizer['decoder']['num_blocks']
        self.dec_hidden_dim = tokenizer['decoder']['hidden_dim']
        self.dec_token_inter_dim = tokenizer['decoder']['token_inter_dim']
        self.dec_hidden_inter_dim = tokenizer['decoder']['hidden_inter_dim']
        self.dec_dropout = tokenizer['decoder']['dropout']

        self.token_num = tokenizer['codebook']['token_num']
        self.token_class_num = tokenizer['codebook']['token_class_num']
        self.token_dim = tokenizer['codebook']['token_dim']
        self.decay = tokenizer['codebook']['ema_decay']

        self.start_embed = nn.Linear(
            input_channels, int(self.enc_hidden_dim))
        
        # token and channel linear
        self.encoder = nn.ModuleList(
            [MixerLayer(self.enc_hidden_dim, self.enc_hidden_inter_dim, 
                self.input_token_num, self.enc_token_inter_dim,
                self.enc_dropout) for _ in range(self.enc_num_blocks)])
        self.encoder_layer_norm = nn.LayerNorm(self.enc_hidden_dim)
        
        self.token_mlp = nn.Linear(
            self.input_token_num, self.token_num)
        self.feature_embed = nn.Linear(
            self.enc_hidden_dim, self.token_dim)     
        
        self.decoder_token_mlp = nn.Linear(
            self.token_num, self.input_token_num)
        self.decoder_start = nn.Linear(
            self.token_dim, self.dec_hidden_dim)

        self.decoder = nn.ModuleList(
            [MixerLayer(self.dec_hidden_dim, self.dec_hidden_inter_dim,
                self.input_token_num, self.dec_token_inter_dim, 
                self.dec_dropout) for _ in range(self.dec_num_blocks)])
        self.decoder_layer_norm = nn.LayerNorm(self.dec_hidden_dim)

        self.recover_embed = nn.Linear(self.dec_hidden_dim, input_channels)
        if 'bound' in tokenizer.keys():
            self.bound = tokenizer['bound']
        else:
            self.bound = False

    def forward(self, input_vector):
        """Forward function. """

        encode_feat = self.encode_vector(input_vector)

        part_token_feat = self.quantize(encode_feat)

        recoverd_vector = self.decode_vector(part_token_feat)

        return recoverd_vector, None, F.mse_loss(input_vector, recoverd_vector)

    def encode_vector(self, input_vector):
        # Encoder of Tokenizer, Get the PCT groundtruth class labels.
        encode_feat = self.start_embed(input_vector)

        for num_layer in self.encoder:
            encode_feat = num_layer(encode_feat)
        encode_feat = self.encoder_layer_norm(encode_feat)
        
        encode_feat = encode_feat.transpose(2, 1)
        encode_feat = self.token_mlp(encode_feat).transpose(2, 1)
        encode_feat = self.feature_embed(encode_feat)

        if self.bound:
            # encode_feat = torch.clamp(encode_feat, -1, 1)
            # encode_feat = 2 * torch.sigmoid(encode_feat) - 1
            encode_feat = torch.tanh(encode_feat)

        return encode_feat

    def fsq(self, x):
        n = self.token_class_num - 1
        return Lambda(lambda x: torch.round(x * n) / n)(x)

    def quantize(self, encode_feat):
        bs = encode_feat.shape[0]

        xq = self.fsq(encode_feat)
        part_token_feat = encode_feat + (xq - encode_feat).detach()

        part_token_feat = part_token_feat.view(bs, -1, self.token_dim)

        return part_token_feat

    def decode_vector(self, part_token_feat):
        # Decoder of Tokenizer, Recover the joints.
        part_token_feat = part_token_feat.transpose(2,1)
        part_token_feat = self.decoder_token_mlp(part_token_feat).transpose(2,1)
        decode_feat = self.decoder_start(part_token_feat)

        for num_layer in self.decoder:
            decode_feat = num_layer(decode_feat)
        decode_feat = self.decoder_layer_norm(decode_feat)

        recoverd_vector = self.recover_embed(decode_feat)

        return recoverd_vector

    def init_weights(self, pretrained=""):
        """Initialize model weights."""

        parameters_names = set()
        for name, _ in self.named_parameters():
            parameters_names.add(name)

        buffers_names = set()
        for name, _ in self.named_buffers():
            buffers_names.add(name)

        if os.path.isfile(pretrained):
            assert (self.stage_pct == "classifier"), \
                "Training tokenizer does not need to load model"
            pretrained_state_dict = torch.load(pretrained, 
                            map_location=lambda storage, loc: storage)

            need_init_state_dict = {}

            for name, m in pretrained_state_dict['state_dict'].items():
                if 'keypoint_head.tokenizer.' in name:
                    name = name.replace('keypoint_head.tokenizer.', '')
                if name in parameters_names or name in buffers_names:
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=True)
        else:
            if self.stage_pct == "classifier":
                print('If you are training a classifier, '\
                    'must check that the well-trained tokenizer '\
                    'is located in the correct path.')


@MODELS.register_module()
class Dress_FSQTOKENIZER_Interface(Dress_FSQTOKENIZER):
    """ 
    Args:
        tokenizer (list): Config about the tokenizer.
        input_token_num (int): Number of annotated panels in the dataset.
        input_channels (int): Number of input channels.
    """

    def __init__(self,
                 tokenizer:str =None,
                 input_token_num=39,
                 input_channels=787,
                 criterion=None # calculate the loss of the cloth
                 ):
        super().__init__(tokenizer, input_token_num, input_channels, criterion)
    
    def encode(self, x):
        """Encode function. """

        encode_feat = self.encode_vector(x)

        return encode_feat
    
    def decode(self, x, accelerator, force_not_quantize=False):
        """Decode function. """

        if not force_not_quantize:
            part_token_feat, encoding_indices, e_latent_loss = self.quantize(x, accelerator)
        else:
            part_token_feat = x

        recoverd_vector = self.decode_vector(part_token_feat, x.shape[0])

        return recoverd_vector
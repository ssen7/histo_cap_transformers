from transformers.modeling_utils import PreTrainedModel

from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
    MaskedImageModelingOutput,
    Seq2SeqLMOutput,
)

from transformers.models.vit import ViTConfig

from transformers.models.vision_encoder_decoder import VisionEncoderDecoderConfig
from transformers.models.vision_encoder_decoder.modeling_vision_encoder_decoder import shift_tokens_right

from transformers.models.vit.modeling_vit import (
    ViTPreTrainedModel, 
    ViTPatchEmbeddings,
    ViTEmbeddings,
    ViTPooler,
    ViTLayer,
    ViTEncoder
)

from transformers.configuration_utils import PretrainedConfig
from transformers.models import auto

import collections.abc
import math
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_utils import *
from vision_transformer4k import vit4k_xs

import pandas as pd

from transformers import AutoModelForCausalLM, AutoTokenizer

VIT_CKPT_PATH = '/home/ss4yd/vision_transformer/HIPT/HIPT_4K/Checkpoints'
RESNET_CKPT_PATH = '/home/ss4yd/new_lstm_decoder/self_supervised_ckpts/tenpercent_resnet18.ckpt'

class ViTConfigCustom(PretrainedConfig):
    def __init__(
        self,
        hidden_size=576,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size


class ViTModelCustom(ViTPreTrainedModel):
    def __init__(self, config, path_input_dim=384,  size_arg = "small", dropout=0.25, pretrain_4k='None', freeze_4k=False, pretrain_WSI='None', freeze_WSI=False):
        super().__init__(config)
        self.config = config

        self.size_dict_path = {"small": [384, 192, 192], "big": [1024, 512, 384]}
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        size = self.size_dict_path[size_arg]

        ### Local Aggregation
        self.local_vit = vit4k_xs()
        if pretrain_4k != 'None':
            print("Loading Pretrained Local VIT model...",)
            state_dict = torch.load(f'{VIT_CKPT_PATH}/%s.pth' % pretrain_4k, map_location='cpu')['teacher']
            state_dict = {k.replace('module.', ""): v for k, v in state_dict.items()}
            state_dict = {k.replace('backbone.', ""): v for k, v in state_dict.items()}
            missing_keys, unexpected_keys = self.local_vit.load_state_dict(state_dict, strict=False)
            print("Done!")
        if freeze_4k:
            print("Freezing Pretrained Local VIT model")
            for param in self.local_vit.parameters():
                param.requires_grad = False
            print("Done")

        ### Global Aggregation
        self.pretrain_WSI = pretrain_WSI
        if pretrain_WSI != 'None':
            pass
        else:
            self.global_phi = nn.Sequential(nn.Linear(576, 576), nn.ReLU(), nn.Dropout(dropout))
            self.global_attn_pool = Attn_Net(L=576, D=256, dropout=dropout, n_classes=1)
            self.global_rho = nn.Sequential(*[nn.Linear(576, 576), nn.ReLU(), nn.Dropout(dropout)])
            
    def forward(self, pixel_values, **kwargs):
        
        pixel_values = pixel_values.squeeze(0)
        ### Local
        local_4096 = self.local_vit(pixel_values.unfold(1, 16, 16).transpose(1,2))
        h_4096 = local_4096.unsqueeze(1).repeat(1, pixel_values.shape[1], 1)
        h_4096 = torch.cat([pixel_values, h_4096], dim=2)
        init_shape = h_4096.shape
        
        #calc input for attn
        # features_mean256 = pixel_values.mean(dim=1)
        # attn_inp = torch.cat([features_mean256, local_4096], dim=1)
        attn_inp = h_4096.mean(dim=1)
        
        ### Global
        if self.pretrain_WSI != 'None':
            h_WSI = self.global_vit(h_4096.unsqueeze(dim=0))
        else:
            h_4096 = self.global_phi(h_4096)
            attn_inp = self.global_phi(attn_inp)
            A_4096, attn_inp = self.global_attn_pool(attn_inp)  
            A_4096 = torch.transpose(A_4096, 1, 0)
            A_4096 = F.softmax(A_4096, dim=1) 
            h_path = torch.mm(A_4096, h_4096.view(-1, h_4096.shape[1]*h_4096.shape[2]))
            h_path = h_path.view(-1, init_shape[1], init_shape[2])
            h_WSI = self.global_rho(h_path)

        return BaseModelOutput(
            last_hidden_state=h_WSI,
            hidden_states=None,
            attentions=A_4096,
        )

class CustomVEDConfig(PretrainedConfig):
    model_type = "vision-encoder-decoder"
    is_composition = True
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_encoder_decoder = True



class CustomVisionEncoderDecoder(PreTrainedModel):
    config_class = CustomVEDConfig
    base_model_prefix = "vision_encoder_decoder"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    
    def __init__( self,config: Optional[PretrainedConfig] = None, encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[PreTrainedModel] = None, n_layers=2):
        
        config.tie_word_embeddings = False
        super().__init__(config)
        
        self.encoder = encoder
        self.decoder = decoder
        self.n_layers = n_layers
        
        if (
            self.encoder.config.hidden_size != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            self.enc_to_dec_proj = nn.Linear(self.encoder.config.hidden_size, self.decoder.config.hidden_size)
        
        self.freeze_decoder_layers(n_layers)

    def freeze_decoder_layers(self, n_layers=2):
        for param in self.decoder.bert.embeddings.parameters():
            param.requires_grad=False

        if n_layers>0: # only freeze the last n_layers layers
            for module in list(*self.decoder.bert.encoder.children())[:-n_layers]:
                for param in module.parameters():
                    param.requires_grad=False
        else: # freeze all layers except for final trainable layer
            for module in list(*self.decoder.bert.encoder.children()):
                for param in module.parameters():
                    param.requires_grad=False

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        if encoder_outputs is None:
            if pixel_values is None:
                raise ValueError("You have to specify pixel_values")

            encoder_outputs = self.encoder(
                pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )
        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = BaseModelOutput(*encoder_outputs)

        encoder_hidden_states = encoder_outputs[0]

        # optionally project encoder_hidden_states
        if (
            self.encoder.config.hidden_size != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        # else:
        encoder_attention_mask = None

        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        # Compute loss independent from decoder (as some shift the logits inside them)
        loss = None
        if labels is not None:
            logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            loss_fct = nn.CrossEntropyLoss()
            labels[labels == self.config.pad_token_id] = -100
            loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.reshape(-1))

        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs

        return Seq2SeqLMOutput(
            loss=loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
    
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
        decoder_inputs = self.decoder.prepare_inputs_for_generation(input_ids, past_key_values=past_key_values)
        decoder_attention_mask = decoder_inputs["attention_mask"] if "attention_mask" in decoder_inputs else None
        input_dict = {
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_input_ids": decoder_inputs["input_ids"],
            "encoder_outputs": encoder_outputs,
            "past_key_values": decoder_inputs["past_key_values"],
            "use_cache": use_cache,
        }
        return input_dict
    
    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_output_embeddings(self):
        return self.decoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self.decoder.set_output_embeddings(new_embeddings)
    
    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            # logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past

import torch
import torch.nn as nn

import random
import math
import copy
import logging

from transformers import (
    BertTokenizer,
    BertPreTrainedModel,
    BertModel,
    BertConfig
)
from transformers.models.bert.modeling_bert import (
    BertEmbeddings,
    BertEncoder,
)
from transformers.file_utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    add_code_sample_docstrings,
)
from transformers.modeling_outputs import (
    SequenceClassifierOutput
)

from tools import *
from text import *


_CHECKPOINT_FOR_DOC = "bert-base-uncased"
_CONFIG_FOR_DOC = "BertConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"

class SimCSEModel(BertPreTrainedModel):
    def __init__(self, config,temp=0.05):
        super().__init__(config)

        self.embedding = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.projector = MLPLayer(config)

        self.loss_fct = InfoNCE(temp)
        
        self.init_weights()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False, # if true, return sentence embedding for evaluation
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None


        if sent_emb:
            # return sentence embedding for evaluation
            embedding = self.embedding(
                input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
            )   
            representation = self.encoder(
                embedding,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                return_dict=return_dict,
            )
            return representation
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # view_0 & view_1 are the same sentence go through bert twice
        embedding_0 = self.embedding(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        
        embedding_1 = self.embedding(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )


        representation_0 = self.encoder(
            embedding_0,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            return_dict=return_dict,
        )
        representation_1 = self.encoder(
            embedding_1,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            return_dict=return_dict,
        )

        # go through mlp twice
        pooled_output_0 = self.projector(representation_0.last_hidden_state[:, 0])
        pooled_output_1 = self.projector(representation_1.last_hidden_state[:, 0])

        loss = self.loss_fct(pooled_output_0,pooled_output_1)
        
        return SequenceClassifierOutput(
            loss=loss,
            hidden_states=[representation_0.last_hidden_state[:, 0],representation_1.last_hidden_state[:, 0]],
            attentions=None,
        )


class DirectCSEModel(BertPreTrainedModel):
    def __init__(self, config,temp=0.05):
        super().__init__(config)

        self.embedding = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        # self.projector = MLPLayer(config)
        self.cut_dim = config.cut_dim

        self.loss_fct = InfoNCE(temp)
        
        self.init_weights()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False, # if true, return sentence embedding for evaluation
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None


        ### start here!!
        if sent_emb:
            # return sentence embedding for evaluation
            embedding = self.embedding(
                input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
            )   
            representation = self.encoder(
                embedding,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                return_dict=return_dict,
            )
            return representation
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # view_0 & view_1 are the same sentence go through bert twice
        embedding_0 = self.embedding(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        
        embedding_1 = self.embedding(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )


        representation_0 = self.encoder(
            embedding_0,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            return_dict=return_dict,
        )
        representation_1 = self.encoder(
            embedding_1,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            return_dict=return_dict,
        )

        # go through mlp twice
        # pooled_output_0 = self.projector(representation_0.last_hidden_state[:, 0])
        # pooled_output_1 = self.projector(representation_1.last_hidden_state[:, 0])
        output_0 = representation_0.last_hidden_state[:, 0][:,:self.cut_dim]
        output_1 = representation_1.last_hidden_state[:, 0][:,:self.cut_dim]

        loss = self.loss_fct(output_0,output_1)
        
        return SequenceClassifierOutput(
            loss=loss,
            hidden_states=[representation_0.last_hidden_state[:, 0],representation_1.last_hidden_state[:, 0]],
            attentions=None,
        )


class BYOLSEModel(BertPreTrainedModel):
    def __init__(self, config,temp=0.05):
        super().__init__(config)
        self.decay = config.decay
        self.online_embedding = BertEmbeddings(config)
        self.online_encoder = BertEncoder(config)
        self.online_projector = MLPLayer(config)
        self.online_predictor = MLPLayer(config)

        self.loss_fct = BYOLMSE(temp)
        
        self.init_weights()

    def prepare(self):
        self.target_embedding = EMA(self.online_embedding, decay = self.decay)
        self.target_encoder = EMA(self.online_encoder,decay = self.decay)
        self.target_projector = EMA(self.online_projector,decay = self.decay)

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False, # if true, return sentence embedding for evaluation
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        if sent_emb:
            # return sentence embedding for evaluation
            embedding = self.online_embedding(
                input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
            )   
            representation = self.online_encoder(
                embedding,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                return_dict=return_dict,
            )
            return representation
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        self.target_embedding.update(self.online_embedding)
        self.target_encoder.update(self.online_encoder)
        self.target_projector.update(self.online_projector)

        # view_0 & view_1 are the same sentence go through bert twice
        embedding_0 = self.online_embedding(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        
        embedding_1 = self.target_embedding.model(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )


        representation_0 = self.online_encoder(
            embedding_0,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            return_dict=return_dict,
        )
        representation_1 = self.target_encoder.model(
            embedding_1,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            return_dict=return_dict,
        )

        # go through mlp twice
        pooled_output_0 = self.online_projector(representation_0.last_hidden_state[:, 0])
        pooled_output_1 = self.target_projector.model(representation_1.last_hidden_state[:, 0])

        pooled_output_0 = self.online_predictor(pooled_output_0)

        loss = self.loss_fct(pooled_output_0,pooled_output_1)
        
        return SequenceClassifierOutput(
            loss=loss,
            hidden_states=[representation_0.last_hidden_state[:, 0],representation_1.last_hidden_state[:, 0]],
            attentions=None,
        )


class DirectBYOLSEModel(BertPreTrainedModel):
    def __init__(self, config,temp=0.05):
        super().__init__(config)
        self.decay = config.decay
        self.cut_dim = config.cut_dim
        self.online_embedding = BertEmbeddings(config)
        self.online_encoder = BertEncoder(config)

        self.loss_fct = BYOLMSE(temp)
        
        self.init_weights()

    def prepare(self):
        self.target_embedding = EMA(self.online_embedding, decay = self.decay)
        self.target_encoder = EMA(self.online_encoder,decay = self.decay)

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False, # if true, return sentence embedding for evaluation
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        if sent_emb:
            # return sentence embedding for evaluation
            embedding = self.online_embedding(
                input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
            )   
            representation = self.online_encoder(
                embedding,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                return_dict=return_dict,
            )
            return representation
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        self.target_embedding.update(self.online_embedding)
        self.target_encoder.update(self.online_encoder)

        # view_0 & view_1 are the same sentence go through bert twice
        embedding_0 = self.online_embedding(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        
        embedding_1 = self.target_embedding.model(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )


        representation_0 = self.online_encoder(
            embedding_0,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            return_dict=return_dict,
        )
        representation_1 = self.target_encoder.model(
            embedding_1,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            return_dict=return_dict,
        )

        # go through mlp twice
        pooled_output_0 = representation_0.last_hidden_state[:, 0][:,:self.cut_dim]        
        pooled_output_1 = representation_1.last_hidden_state[:, 0][:,:self.cut_dim]

        loss = self.loss_fct(pooled_output_0,pooled_output_1)
        
        return SequenceClassifierOutput(
            loss=loss,
            hidden_states=[representation_0.last_hidden_state[:, 0],representation_1.last_hidden_state[:, 0]],
            attentions=None,
        )

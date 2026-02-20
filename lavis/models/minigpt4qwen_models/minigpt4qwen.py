"""
Requires Transformer 4.32 and above, implementation may change according the Llama implementation
"""
import os
import logging
import string
from packaging import version

from omegaconf import OmegaConf

import contextlib

import torch
# torch.autograd.set_detect_anomaly(True) # for debug
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

import transformers
from peft import LoraConfig, get_peft_model

from lavis.common.registry import registry
from lavis.models.minigpt4qwen_models.blip2 import Blip2Base, disabled_train

from functools import partial
import re
from copy import deepcopy

from .chat_utils import get_stop_words_ids, make_context, decode_tokens

_ERROR_BAD_CHAT_FORMAT = """\
We detect you are probably using the pretrained model (rather than chat model) for chatting, since the chat_format
in generation_config is not "chatml".
If you are directly using the model downloaded from Huggingface, please make sure you are using our "Qwen/Qwen-7B-Chat"
Huggingface model (rather than "Qwen/Qwen-7B") when you call model.chat().
我们检测到您可能在使用预训练模型（而非chat模型）进行多轮chat，因为您当前在generation_config指定的chat_format，
并未设置为我们在对话中所支持的"chatml"格式。
如果您在直接使用我们从Huggingface提供的模型，请确保您在调用model.chat()时，使用的是"Qwen/Qwen-7B-Chat"模型
（而非"Qwen/Qwen-7B"预训练模型）。
"""

_SENTINEL = object()
_ERROR_STREAM_IN_CHAT = """\
Pass argument `stream` to model.chat() is buggy, deprecated, and marked for removal. Please use model.chat_stream(...)
instead of model.chat(..., stream=True).
向model.chat()传入参数stream的用法可能存在Bug，该用法已被废弃，将在未来被移除。
请使用model.chat_stream(...)代替model.chat(..., stream=True)。
"""


@registry.register_model("minigpt4qwen")
class Minigpt4Qwen(Blip2Base):
    """
    BLIP2 + Projection + Qwen7B-chat = Minigpt4Qwen model.
    Supported model types:
        - qwen7b_chat
    Usage:
        >> from lavis.models import load_model
        >> model = load_model("minigpt4qwen", "qwen7b_chat")
    """

    # pretrained model config dict
    PRETRAINED_MODEL_CONFIG_DICT = {
        "qwen7b_chat": "configs/models/minigpt4qwen/minigpt4qwen.yaml",
        "qwen14b_chat": "configs/models/minigpt4qwen/minigpt4qwen-14b.yaml",
    }

    def __init__(
        self,
        # ===== 1. Vision Encoder =====
        vit_model="eva_clip_g",
        img_size=224,
        vit_precision="fp16",
        drop_path_rate=0,
        use_grad_checkpoint=False,
        freeze_vit=True,
        unfreeze_pos_embed=False,

        # ===== 2. Q-Former =====
        num_query_token=32,
        qformer_text_input=True,
        freeze_qformer=False,
        freeze_queries=False,

        # ===== 3. Projection (Q-Former → LLM) =====
        freeze_proj=False,

        # ===== 4. LLM (Qwen) =====
        llm_model="",
        max_txt_len=512,
        freeze_llm=True,
        llm_device_map="cpu",

        # ===== 5. LoRA / PEFT =====
        get_lora=False,
        lora_r=8,
        lora_alpha=32,
        lora_dropout=0.05,

        # ===== 6. Training / Inference Utilities =====
        enable_autocast=True,  # enable AMP
        apply_lemmatizer=False,
    ):
        super().__init__()

        # check transformer version
        transformers_version = version.parse(transformers.__version__)  # package.version()
        assert transformers_version >= version.parse("4.32"), "Minigpt4Qwen requires transformers>=4.32"
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

        # self.tokenizer = self.init_tokenizer(truncation_side="left")

        # ----------------- Visual Encoder -----------------
        # initialize a Visual Encoder and its Layer Norm
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(  # init_vision_encoder() from BLIP-2
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        # freeze ViT's parameters if need
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False

            self.visual_encoder = self.visual_encoder.eval()  # use ViT's eval mode
            self.visual_encoder.train = disabled_train  # close ViT's train mode
            logging.info("freeze vision encoder")

            # check if freezes pos_embed
            if unfreeze_pos_embed:
                self.visual_encoder.pos_embed.requires_grad_(True)

        # ----------------- Q-former -----------------
        self.num_query_token = num_query_token
        self.Qformer, self.query_tokens = self.init_Qformer(  # init_Qformer() from BLIP-2
            num_query_token, self.visual_encoder.num_features
        )

        # there's text input for Q-former or not
        if not qformer_text_input:
            logging.info("no text input for q-former")
            # Q-Former = BERT encoder + cross-attention + learnable query tokens
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                # close FNN
                layer.output = None  # FNN's down
                layer.intermediate = None  # FNN's up
        else:
            raise NotImplementedError  # not yet implemented
        self.Qformer.cls = None  # remove Bert's CLS token

        # freeze Q-former or not
        if freeze_qformer:
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train
            for _, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train

        # freeze queries or not
        if freeze_queries:
            self.query_tokens.requires_grad = False

        # ----------------- LLM -----------------
        print(f'Loading LLM:{llm_model}...')

        # initialize LLM tokenizer
        self.llm_tokenizer = AutoTokenizer.from_pretrained(  # AutoTokenizer.from_pretrained() from transformers
            llm_model,
            cache_dir=registry.get_path("cache_root"),
            model_max_length=max_txt_len,
            padding_side="right",
            use_fast=False,
            trust_remote_code=True
        )

        # initialize LLM config (needn't save it into class)
        llm_config = AutoConfig.from_pretrained(  # also from transformers
            llm_model,
            cache_dir=registry.get_path("cache_root"),
            trust_remote_code=True
        )

        # initialize LLM model
        self.llm_model = AutoModelForCausalLM.from_pretrained(  # also from transformers
            llm_model,
            config=llm_config,
            cache_dir=registry.get_path("cache_root"),
            trust_remote_code=True,
            device_map=llm_device_map
        )

        # Gradient Checkpointing: trades extra computation for much lower memory usage
        self.llm_model.gradient_checkpointing_enable()

        # Special Tokens: pad -> EOD, image token placeholder -> <|extra_0|>
        self.llm_tokenizer.pad_token_id = self.llm_tokenizer.eod_id
        self.replace_image_token_id = self.llm_tokenizer("<|extra_0|>").input_ids[0]
        self.replace_image_string = '<|extra_0|>'

        # freeze LLM or not
        self.freeze_llm = freeze_llm
        if self.freeze_llm:
            print("Freeze LLM...")
            for name, param in self.llm_model.named_parameters():
                param.requires_grad = False
        else:
            print("Unfreeze LLM!!!")
            for name, param in self.llm_model.named_parameters():
                param.requires_grad = True

        # put visual tokens into Q-former space (D_v -> D_q)
        self.llm_proj = nn.Linear(
            self.Qformer.config.hidden_size,
            self.llm_model.config.hidden_size
        )

        # ----------------- Lora -----------------
        self.get_lora = get_lora
        self.lora_alpha = lora_alpha
        self.lora_r = lora_r
        self.lora_dropout = lora_dropout

        if self.get_lora:
            peft_config = LoraConfig(
                target_modules=['q_proj', 'v_proj'],
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.llm_model = get_peft_model(self.llm_model, peft_config)
            self.llm_model.print_trainable_parameters()

        # enable autocast
        self.enable_autocast = enable_autocast

    def encode_image(self, image):
        with (self.maybe_autocast() if self.enable_autocast else contextlib.nullcontext()):
            image_embeds = self.visual_encoder(image)  # (B, C, H, W) -> (B, N_patch, D_v)
            image_embeds = self.ln_vision(image_embeds)  # layer norm

        # (B, N_img) ones, means all patch tokens are unmasked.
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)

        bs = image.size(0)  # batch size

        # (N_query, D_q) -> (B, N_query, D_q), expand() means view extension instead of copy.
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        # there is text input into Q-former or not
        if self.qformer_text_input:
            raise NotImplementedError  # not yet implemented
        else:
            # visual tokens + queries -> cross attention -> output queries (B, N_q, D_q)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,  # (B, N_q, D_q)
                encoder_hidden_states=image_embeds,  # (B, N_p, D_v)
                encoder_attention_mask=image_atts,
                return_dict=True
            )

        # (B, N_q, D_q) -> (B, N_q, D_t), this slice ensures there are N_q queries
        inputs_llm = self.llm_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :])

        return inputs_llm

    def preprocess(
        self,
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        max_len: int,
        image_len: int = 32,
        system_message: str = "You are a helpful assistant."
    ):
        IGNORE_TOKEN_ID = -100

        # ChatML role mapping dict, str -> token
        roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

        # get different tokens' ids
        im_start = tokenizer.im_start_id
        im_end = tokenizer.im_end_id
        nl_tokens = tokenizer('\n').input_ids  # \n's id
        _system = tokenizer('system').input_ids + nl_tokens  # 'system \n'
        _user = tokenizer('user').input_ids + nl_tokens  # 'user \n'
        _assistant = tokenizer('assistant').input_ids + nl_tokens  # 'assistant \n'

        # apply prompt templates
        input_ids, targets = [], []
        for i, source in enumerate(sources):
            img_visit_cnt = 0
            # {"from": "user", "value": "..."}, {"from": "assistant", "value": "..."}
            if roles[source[0]["from"]] != roles["user"]:  # this source is from assistant
                source = source[1:]  # skip this incomplete sample without user input

            input_id, target = [], []
            # system is like a rule statement at the beginning: "You are a helpful assistant."
            # only im_start and im_end are ints, others are lists.
            # system = <|im_start|> + system + \n + ‘You are a helpful assistant.’ + <|im_end|> + \n
            system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
            input_id += system

            # align with input length, -3 means [im_start], [im_end] and nl_tokens.
            # system message needn't be trained, so use IGNORE_TOKEN_ID.
            target += [im_start] + [IGNORE_TOKEN_ID] * (len(system)-3) + [im_end] + nl_tokens
            assert len(input_id) == len(target)

            for j, sentence in enumerate(source):
                role = roles[sentence['from']]
                content = sentence['value']

                # clear image placeholders
                if self.replace_image_string in content:
                    # attention: replace won't in-place update
                    content = content.replace(self.replace_image_string, "")

                # only process images in the user round
                if "<ImageHere>" in content and role == '<|im_start|>user':
                    # support multi-picture/video input
                    img_visit_cnt += content.count("<ImageHere>")
                    # replace "<ImageHere>" to image placeholders
                    content = content.replace("<ImageHere>", self.replace_image_string * image_len)
                    # input = input + n\ + contents + end + n\
                    _input_id = tokenizer(role).input_ids + nl_tokens + \
                                tokenizer(content).input_ids + [im_end] + nl_tokens
                else:
                    # LLM round doesn't have images.
                    _input_id = tokenizer(role).input_ids + nl_tokens + \
                                tokenizer(content).input_ids + [im_end] + nl_tokens

                """
                input_id  =
                [system tokens]
                [user tokens]
                [assistant tokens]
                [user tokens]
                [assistant tokens]
                """
                input_id += _input_id

                if role == '<|im_start|>user':
                    # if user, ignore all
                    _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id) - 3) + [im_end] + nl_tokens
                elif role == '<|im_start|>assistant':
                    # if assistant, only train the text
                    _target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                              _input_id[len(tokenizer(role).input_ids) + 1:-2] + [im_end] + nl_tokens
                else:
                    raise NotImplementedError

                """
                target =
                [-100  -100  -100  ...]   ← system（ignore all）
                [-100  -100  -100  ...]   ← user（ignore all）
                [-100  -100   A   B   C]  ← assistant（only train the text）
                """
                target += _target

            assert len(input_id) == len(target), "input_ids should have the same length as the target"

            # All the sequences in the batch must be of the same length for parallel computation.
            # So, pad input_id and target to the length max_len
            input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
            target += [IGNORE_TOKEN_ID] * (max_len - len(target))

            input_ids.append(input_id[:max_len])
            targets.append(target[:max_len])

        """
        input_ids and targets contain multiple sources, each source is a whole continuous multi-round dialogue.
        source = [
          {"from": "user", "value": "..."},
          {"from": "assistant", "value": "..."},
          {"from": "user", "value": "..."},
          {"from": "assistant", "value": "..."},
        ]
        """

        # lists to Tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        targets = torch.tensor(targets, dtype=torch.long)

        return dict(
            input_ids=input_ids,
            labels=targets,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),  # ne means not equal
        )

    def forward(self, samples):
        image = samples["image"]
        image_embeds = self.encode_image(image)  # (B, N_query, D)

        sources = samples["conversations"]
        data_dict = self.preprocess(sources, self.llm_tokenizer, self.max_txt_len, image_len = self.num_query_token)
        device = self.llm_model.device
        llm_tokens = data_dict['input_ids'].to(device)
        targets = data_dict['labels'].to(device)
        attention_mask = data_dict['attention_mask']

        # use torch.where() to find image placeholder indexes
        replace_image_idxs = torch.where(llm_tokens == self.replace_image_token_id)

        # input tokens -> input embeddings
        # get_input_embeddings() is from transformers.AutoModelForCausalLM
        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens)  # (B, L, D)

        # replace image placeholders with image query embeddings.
        _, _, D_t = inputs_embeds.shape
        # view: (B, N_query, D) -> (B * N_query, D)
        inputs_embeds[replace_image_idxs[0], replace_image_idxs[1]] \
            = image_embeds.view(-1, D_t).to(inputs_embeds.dtype)

        # put into LLM and get output
        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets
        )

        loss = outputs.loss
        return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        samples,  # samples = { "image": Tensor(B, C, H, W), "text": [...]}
        chat=False,  # single or multiple rounds QA
        generation_config=None,
        stop_words_ids=None,
        return_dict_in_generate=False,
        **kwargs
    ):
        # use default generation_config from transformers.from_pretrained() if None
        generation_config = generation_config if generation_config is not None\
            else self.llm_model.generation_config

        # tokenizer setup
        self.llm_tokenizer.padding_side = 'left'
        self.llm_tokenizer.pad_token_id = self.llm_tokenizer.eod_id  # some LLM doesn't have specific padding token

        # deep copy original data to prevent in-place ops from polluting the original data in reasoning
        image = deepcopy(samples['image'])
        text = deepcopy(samples['text'])

        # ------------- check text batch size -------------
        if isinstance(text, str):
            text = [text]
            bs = 1
        elif isinstance(text, list):
            bs = len(text)
        else:
            raise TypeError

        # qformer_text_input should be False
        if self.qformer_text_input:
            raise NotImplementedError

        # ------------- check image dimension -------------
        # for video data (to be continued ...)
        if image.dim() == 5:
            assert False, 'the dim of image is 5, but now we don\'t support 5D images/video input'
        elif image.dim() == 4:
            # get true image embeddings
            image_embeds = self.encode_image(image)
        else:
            assert False, f'the dim of image is {image.dim()}, we only support image input with a shape [B,C,H,W].'

        # ------------- add image placeholder tokens to all text[i] -------------
        for i in range(bs):
            image_num = text[i].count("<ImageHere>")
            if image_num:
                print(f"In Batch_{i} Query: {image_num} images!")
            # <|extra_0|> * num_query
            replace_string = ''.join([self.replace_image_string] * self.num_query_token)
            if self.replace_image_string in text[i]:
                # clean up the residual <|extra_0|>
                # attention: replace() doesn't support in-place edit
                text[i] = text[i].replace(self.replace_image_string, "")
            # <ImageHere> -> <|extra_0|> * num_query
            text[i] = text[i].replace('<ImageHere>', replace_string)

        # ------------- llm tokens, mask, input_ids and input_embeds -------------
        llm_tokens = self.llm_tokenizer(
            text,
            return_tensors='pt',  # return Pytorch Tensor
            padding='longest'  # all sequences pad to the longest length
        )
        attention_mask = llm_tokens.attention_mask.to(image.device)  # attention_mask from llm_tokenizer
        llm_tokens.input_ids = llm_tokens.input_ids.to(image.device)  # input_ids from llm_tokenizer

        replace_image_idxs = torch.where(llm_tokens.input_ids == self.replace_image_token_id)
        # get_input_embeddings() return a nn.Embedding module(function)
        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)

        # ------------- project image embeddings into inputs_embeds -------------
        _, _, D_t = inputs_embeds.shape
        # (B, N_q, D) -> (B * N_q, D)
        inputs_embeds[replace_image_idxs[0], replace_image_idxs[1]]\
            = image_embeds.view(-1, D_t).to(inputs_embeds.dtype)

        # ------------- output -------------
        # llm_model.generate(): logits → argmax / sampling → token_id
        outputs = self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            stop_words_ids=stop_words_ids,
            return_dict_in_generate=False,
            generation_config=generation_config,
            pad_token_id=self.llm_tokenizer.eod_id,
            bos_token_id=self.llm_tokenizer(' ').input_ids[0],  # use the first space token id as the BOS
            # roles = {<|im_start|>user, <|im_start|>assistant}, so im_start_id should also stop
            eos_token_id=[self.llm_tokenizer.im_end_id, self.llm_tokenizer.im_start_id]
        )

        if not chat:
            output_text = [
                # put tokenizer.decoder() into cpu
                self.llm_tokenizer.decode(_[:].cpu(), skip_special_tokens=True).strip() for _ in outputs
            ]
            return output_text
        else:
            return outputs

    def chat(
        self,
        query,  # current input prompt
        history,  # [("user question 1", "assistant answer 1"), ("user question 2", "assistant answer 2") ...]
        image_tensor,  # after preprocess: (B, C, H, W)
        system="You are a helpful assistant.",
        append_history=True,
        stream=_SENTINEL,  # streaming output, _SENTINEL means not specified
        stop_words_ids=None,
        generation_config=None,
        **kwargs,
    ):
        # ------------- initialization -------------
        # if generation_config is None, use default
        generation_config = generation_config if generation_config is not None \
            else self.llm_model.generation_config

        # forbid stream input
        assert stream is _SENTINEL, _ERROR_STREAM_IN_CHAT
        # must use 'chatml' chat_format
        assert generation_config.chat_format == 'chatml', _ERROR_BAD_CHAT_FORMAT

        if history is None:
            history = []
        if stop_words_ids is None:
            stop_words_ids = []

        # if max_window_size is not inputted, use default
        max_window_size = kwargs.get('max_window_size', None)
        if max_window_size is None:
            max_window_size = generation_config.max_window_size

        # get_stop_words_ids() from lavis
        stop_words_ids.extend(get_stop_words_ids(
            generation_config.chat_format, self.llm_tokenizer
        ))

        # ------------- Generate with history -------------

        # make_context() in lavis, return complete history conversations' raw text and tokens
        raw_text, context_tokens = make_context(
            self.llm_tokenizer,
            query,  # current input prompt
            history,  # History (user, assistant) pairs
            system=system,
            max_window_size=max_window_size,
            chat_format=generation_config.chat_format
        )

        sample = {
            'image': image_tensor,
            'text': raw_text
        }

        # (1, seq_len)
        outputs = self.generate(
            sample,
            chat=True,
            stop_words_ids=stop_words_ids,
            return_dict_in_generate=False,
            generation_config=generation_config,
            **kwargs
        )

        # decode_tokens() in lavis, decoding text tokens in ChatML rule.
        response = decode_tokens(
            outputs[0],  # (1, seq_len) -> (seq_len,)
            self.llm_tokenizer,
            chat_format=generation_config.chat_format,
            verbose=kwargs.pop('verbose', False),  # whether to print/return more debugging information
            errors='replace'  # how to deal with illegal characters
        )

        if append_history:
            history.append((query, response))

        return response, history

    # string → NLP token sequence (with POS, lemma, etc.)
    # example: "The cats were running" -> "the cat be run"
    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)  # self.lemmatizer() is super from BlipBase.__init__()

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:  # pos_: Part-of-Speech
                    words.append(token.lemma_)  # lemma_: original words
                else:
                    words.append(token.text)  # no operation
            answer = " ".join(words)

            return answer
        return [apply(answer) for answer in answers]

    # lazy load: only load it at the first visit
    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)
        return self._lemmatizer

    # initialize pretrained model
    @classmethod
    def from_pretrained(cls, model_type, llm_device_map="cpu"):
        """
        Build a pretrained model from default configuration file, specified by model_type.

        Args:
            - model_type (str): model type, specifying architecture and checkpoints.

        Returns:
            - model (nn.Module): pretrained or fine-tuned model, depending on the configuration.
        """
        # find default configuration file path according to model type, '.model' means model part in cfg.
        model_cfg = OmegaConf.load(cls.default_config_path(model_type)).model
        model_cfg['llm_device_map'] = llm_device_map  # running time override (without changing yaml)
        model = cls.from_config(model_cfg)  # initialize model class according to model_cfg.

        return model

    # config -> model
    @classmethod
    def from_config(cls, cfg):
        # ------------- text config -------------
        max_txt_len = cfg.get('max_txt_len', 512)
        apply_lemmatizer = cfg.get('apply_lemmatizer', False)

        # ------------- LLM model config path -------------
        llm_model = cfg.get('llm_model')  # model path
        # relative path -> absolute path
        if not os.path.isabs(llm_model):
            # LAVIS Model Cache Root Directory + llm model path
            llm_model = os.path.join(registry.get_path('cache_root'), llm_model)

        # ------------- lora config -------------
        get_lora = cfg.get("get_lora", False)
        lora_alpha = cfg.get("lora_alpha", 32)
        lora_r = cfg.get("lora_r", 8)
        lora_dropout = cfg.get("lora_dropout", 0.05)

        # ------------- vision encoder config -------------
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        unfreeze_pos_embed = cfg.get("unfreeze_pos_embed", False)
        if freeze_vit == False and unfreeze_pos_embed == False:
            print('unfreeze vit so it will unfreeze pos embed')

        # ------------- Q-former config -------------
        num_query_token = cfg.get("num_query_token")
        qformer_text_input = cfg.get("qformer_text_input", True)
        freeze_qformer = cfg.get("freeze_qformer", False)
        freeze_queries = cfg.get("freeze_queries", False)

        # ------------- proj config -------------
        freeze_proj = cfg.get("freeze_proj", False)

        # ------------- autocast config -------------
        enable_autocast = cfg.get("enable_autocast", True)

        # ------------- freeze llm -------------
        freeze_llm = cfg.get("freeze_llm",True)

        llm_device_map = cfg.get("llm_device_map", "cpu")
        assert llm_device_map in ['cpu', 'auto'],\
            ('please set `llm_device_map` in [`cpu`,`auto`] if training or single-gpu inference,'
             ' set `cpu`. if multi-gpu inference, set `auto`')

        # instantiate model class
        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            llm_model=llm_model,
            max_txt_len=max_txt_len,
            apply_lemmatizer=apply_lemmatizer,
            qformer_text_input=qformer_text_input,
            get_lora=get_lora,
            lora_alpha=lora_alpha,
            lora_r=lora_r,
            lora_dropout=lora_dropout,
            unfreeze_pos_embed=unfreeze_pos_embed,
            freeze_qformer=freeze_qformer,
            freeze_queries=freeze_queries,
            freeze_proj=freeze_proj,
            enable_autocast=enable_autocast,
            freeze_llm=freeze_llm,
            llm_device_map=llm_device_map,
        )

        model.load_checkpoint_from_config(cfg)  # from lavis.BaseModel

        return model

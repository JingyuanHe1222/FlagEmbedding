import wandb 
from typing import Optional, Tuple, Dict, Union, Any, List

from dataclasses import dataclass

import torch
from torch import nn 

# from transformers import WandbCallback
from transformers.modeling_outputs import ModelOutput
from transformers import Trainer
from transformers.trainer_pt_utils import nested_detach


def rewrite_logs(d):
    new_d = {}
    eval_prefix = "eval_"
    eval_prefix_len = len(eval_prefix)
    test_prefix = "test_"
    test_prefix_len = len(test_prefix)
    for k, v in d.items():
        if k.startswith(eval_prefix):
            new_d["eval/" + k[eval_prefix_len:]] = v
        elif k.startswith(test_prefix):
            new_d["test/" + k[test_prefix_len:]] = v
        else:
            new_d["train/" + k] = v
    return new_d


class CustomTrainer(Trainer): 

    def __init__(self, model, *args, **kwargs): 
        super().__init__(model, *args, **kwargs)
        self.ar_loss = []
        self.sum_loss = []
        self.pred_loss = []
        self.eval_ar_loss = []
        self.eval_sum_loss = []
        self.eval_pred_loss = []


    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss, output_dict = self.compute_loss(model, inputs, return_outputs=True)
            ar_loss = output_dict['ar_loss']
            sum_loss = output_dict['sum_loss']
            pred_loss = output_dict['pred_loss']

        del inputs
        torch.cuda.empty_cache()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
            self.ar_loss.append(ar_loss.mean().detach().item())
            self.sum_loss.append(sum_loss.mean().detach().item())
            self.pred_loss.append(pred_loss.mean().detach().item())
        else: 
            self.ar_loss.append(ar_loss.detach().item() if isinstance(ar_loss, torch.Tensor) else ar_loss)
            self.sum_loss.append(sum_loss.detach().item() if isinstance(sum_loss, torch.Tensor) else sum_loss)
            self.pred_loss.append(pred_loss.detach().item() if isinstance(pred_loss, torch.Tensor) else pred_loss) 

        self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps
    

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen


        if "eval_loss" in logs.keys(): 
            # eval loop
            mean_ar_loss = sum(self.eval_ar_loss) / len(self.eval_ar_loss) if len(self.eval_ar_loss) > 0 else len(self.eval_ar_loss)
            if mean_ar_loss: 
                logs["eval_ar_loss"] = mean_ar_loss
            mean_sum_loss = sum(self.eval_sum_loss) / len(self.eval_sum_loss) if len(self.eval_sum_loss) > 0 else len(self.eval_sum_loss)
            if mean_sum_loss: 
                logs["eval_sum_loss"] = mean_sum_loss
            mean_pred_loss = sum(self.eval_pred_loss) / len(self.eval_pred_loss) if len(self.eval_pred_loss) > 0 else len(self.eval_pred_loss)
            if mean_pred_loss: 
                logs["eval_pred_loss"] = mean_pred_loss
            print("logging eval stats... clearing eval loss cache... ")
            self.eval_ar_loss = []
            self.eval_sum_loss = []
            self.eval_pred_loss = []
        else: 
            # training steps
            mean_ar_loss = sum(self.ar_loss)/len(self.ar_loss) if len(self.ar_loss) > 0 else len(self.ar_loss)
            if mean_ar_loss: 
                logs["ar_loss"] = mean_ar_loss
                print("self.ar_loss: ", self.ar_loss, "mean: ", mean_ar_loss) ###
            mean_sum_loss = sum(self.sum_loss)/len(self.sum_loss) if len(self.sum_loss) > 0 else len(self.sum_loss)
            if mean_sum_loss: 
                logs["sum_loss"] = mean_sum_loss
            mean_pred_loss = sum(self.pred_loss)/len(self.pred_loss) if len(self.pred_loss) > 0 else len(self.pred_loss)
            if mean_pred_loss: 
                logs["pred_loss"] = mean_pred_loss
            self.ar_loss = []
            self.sum_loss = []
            self.pred_loss = []

        if "loss" in logs: 
            logs["ar_loss"] = logs["loss"]

        output = {**logs, **{"step": self.state.global_step}, }
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)


    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if has_labels or loss_without_labels:
                with self.compute_loss_context_manager(): # enter here
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

                    ar_loss = outputs['ar_loss']
                    sum_loss = outputs['sum_loss']
                    pred_loss = outputs['pred_loss']

                    self.eval_ar_loss.append(ar_loss.detach().item() if isinstance(ar_loss, torch.Tensor) else ar_loss)
                    self.eval_sum_loss.append(sum_loss.detach().item() if isinstance(sum_loss, torch.Tensor) else sum_loss)
                    self.eval_pred_loss.append(pred_loss.detach().item() if isinstance(pred_loss, torch.Tensor) else pred_loss)

                loss = loss.mean().detach()

                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                else:
                    logits = outputs[1:]
            else:
                loss = None
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                else:
                    logits = outputs
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]
    
        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)
    


@dataclass
class CustomCausalLMOutputWithPast(ModelOutput):
    """
    Custom class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    ar_loss: Optional[torch.FloatTensor] = None
    sum_loss: Optional[torch.FloatTensor] = None
    pred_loss: Optional[torch.FloatTensor] = None

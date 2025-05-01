# models_t5_swda.py
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import torch.nn as nn

class T5DialogueClassifierSWDA(nn.Module):
    def __init__(self, model_name='t5-small'):
        super(T5DialogueClassifierSWDA, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, labels=None):
        if labels is not None:
            output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            return output.loss, output.logits
        else:
            generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask)
            return generated_ids

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)












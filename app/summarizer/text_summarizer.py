import yaml

from torch import nn
from transformers import BartForConditionalGeneration


with open("config/model.yaml") as f:
    model_params = yaml.safe_load(f)


class TextSummarizer(nn.Module):
    def __init__(self):
        super(TextSummarizer, self).__init__()

        self.model = BartForConditionalGeneration.from_pretrained(
            model_params['bart']['bart_model'])

    def forward(self, input_ids):
        max_len = model_params['bart']['max_len']
        min_len = model_params['bart']['min_len']
        len_penalty = model_params['bart']['len_penalty']
        num_beams = model_params['bart']['num_beams']

        summary_ids = self.model.generate(input_ids,
                                          max_length=max_len,
                                          min_length=min_len,
                                          length_penalty=len_penalty,
                                          num_beams=num_beams,
                                          early_stopping=True)

        return summary_ids

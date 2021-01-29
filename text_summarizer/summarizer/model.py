import yaml
import torch
from transformers import BartTokenizer
from .text_summarizer import TextSummarizer


with open("config/model.yaml") as f:
    model_params = yaml.safe_load(f)


class Model:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")

        self.tokenizer = BartTokenizer.from_pretrained(
            model_params['bart']['bart_model'])

        summarizer = TextSummarizer()
        summarizer = summarizer.eval()
        self.model = summarizer.to(self.device)

    def predict(self, text):
        inputs = self.tokenizer([text],
                                max_length=model_params['bart']['max_seq_len'],
                                return_tensors='pt')

        summary_ids = self.model(inputs['input_ids'])
        summary = self.tokenizer.decode(summary_ids.squeeze(),
                                        skip_special_tokens=True)

        return summary


model = Model()


def get_model():
    return model

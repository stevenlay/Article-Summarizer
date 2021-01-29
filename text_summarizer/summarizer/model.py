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
        # summarizer.load_state_dict(torch.load(model_params['bart']['pretrained_model']], map_location=self.device))
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


# Dependency Injection via FastAPI to inject model in api.py
model = Model()


# Singleton function used for API handler
def get_model():
    return model

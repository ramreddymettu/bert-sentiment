import config
from model import BertBaseUncase
import torch
import joblib

from flask import (
    Flask,
    jsonify,
    request
)

app = Flask(__name__)

MODEL = None
PREDICTION_DICT = dict()
memory = joblib.Memory("../input/", verbose=0)

def predict_from_cache(sentence):
    if sentence in PREDICTION_DICT:
        return PREDICTION_DICT[sentence]
    else:
        result = sentence_prediction(sentence)
        PREDICTION_DICT[sentence] = result
        return result

@memory.cache
def parse_sentence(sentence, model):
    tokenizer = config.TOKENIZERS
    max_len = config.MAX_LENGTH

    review = str(sentence)
    review = " ".join(review.split())

    input_ = tokenizer.encode_plus(
            review,
            None,
            add_special_tokens=True,
            max_length=max_len
        )

    ids = input_['input_ids']
    mask = input_['attention_mask']
    token_type_ids = input_['token_type_ids']

    padding_len = max_len - len(ids)

    ids = ids + ( [0]*padding_len )
    mask = mask + ( [0]*padding_len )
    token_type_ids = token_type_ids + ( [0]*padding_len )


    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)

    ids = ids.to(config.DEVICE, dtype=torch.long)
    token_type_ids = token_type_ids.to(config.DEVICE, dtype=torch.long)
    mask = mask.to(config.DEVICE, dtype=torch.long)

    output = model(
        ids,
        mask,
        token_type_ids
    )
    output = torch.sigmoid(output).cpu().detach().numpy()
    return output[0][0]


@app.route("/predict")
def predict():
    sentence = request.args.get('sentence')
    positive_sentiment = parse_sentence(sentence, MODEL)
    negative_sentiment = 1 - positive_sentiment
    return jsonify({"positive_sentiment": str(positive_sentiment), 
                    "negative_sentiment": negative_sentiment,
                    "sentence" :sentence})

if __name__ == "__main__":
    MODEL = BertBaseUncase()
    MODEL.load_state_dict(torch.load(config.MODEL_PATH))
    MODEL.to(config.DEVICE)
    MODEL.eval()
    app.run(debug=True)
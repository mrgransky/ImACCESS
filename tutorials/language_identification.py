import os
import sys
import time
HOME, USER = os.getenv('HOME'), os.getenv('USER')
IMACCESS_PROJECT_WORKSPACE = os.path.join(HOME, "WS_Farid", "ImACCESS")

CLIP_DIR = os.path.join(IMACCESS_PROJECT_WORKSPACE, "clip")
sys.path.insert(0, CLIP_DIR)

MISC_DIR = os.path.join(IMACCESS_PROJECT_WORKSPACE, "misc")
sys.path.insert(0, MISC_DIR)

from utils import *

# import torch
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# from transformers import logging as transformers_logging
# transformers_logging.set_verbosity_error()  # Only show errors, not warnings/info

text = [
	"Fighters of the Regular People’s Army marching throught the first military training camp during its inauguration, in Sant Cugat del Vallès. Parade of the forces of the Popular School of War at the inauguration of the first area of instruction pre soldier of the Popular Army Regular, with motif of the Party of the Youth, Pins of the Vallès ."
	"Turzyniecki Doły. 9th Infantry Regiment of the Legions of the Home Army. The 9th Infantry Regiment of the Legions of the Home Army of Zamość Land was established in 1943. Its commander was Major Stanisław Prus ps. Adam’.",
	"On the left, Kälarne Sågs power station is visible with the timber and water gutter to the power station. in the 1940s. Tasks: Harald Thorsell, Ansjö, Kälarne, 1987.",
	"Amor, ch'a nullo amato amar perdona.",
	"President Rooseveltstraat achterzijde waar een windmolen staat.",
	"Tribune en dugout en spelerstunnel naar de kleedkamers.",
	"Interieur watertoren, Intze 1-reservoir.",
	"Young Infantry Chiefs Course in Biłgoraj District.",
	"Vom östlischen Kriegsschauplatz. Totalansicht von Wilna. Die Stadt mit 37 Kirchen.. The postcard is titled: From the Eastern Front. Panorama Wilna. City of 37 churches. Presents a the city with its numerous churches avers.",
	"Lublin, ul. Świętoduska. Market Square. Austrian prisoners. Reproduced by: Tygodnik Ilustrowany 1914 nr 43 p. 720.",
	"Chełm. Celebrations in 65. Moscow Infantry Regiment. Reproduced from: Perebyvanie. Nikolaja Aleksandrovica v g. Cholme on the 200th anniversary of 65 bad luck. Moskovskago. połka. Celebrations in the 65th Moscow Infantry Regiment related to the visit of Tsar Nicholas II in connection with the transformation of the regiment into them. Cara Nicholas II and the 25th anniversary of the Uniate Church’s return to the Orthodox Church.",
]

model_kwargs: Dict[str, Any] = {
	"low_cpu_mem_usage": True,
	"trust_remote_code": True,
	"cache_dir": cache_directory[USER],
	"dtype": torch.bfloat16,
}

model_ckpt = "papluca/xlm-roberta-base-language-detection"
tokenizer = tfs.AutoTokenizer.from_pretrained(
	model_ckpt,
	use_fast=True,
	trust_remote_code=True,
	cache_dir=cache_directory[USER],
)
if tokenizer.pad_token is None:
	tokenizer.pad_token = tokenizer.eos_token
	tokenizer.pad_token_id = tokenizer.eos_token_id

if hasattr(tokenizer, "padding_side") and tokenizer.padding_side is not None:
	tokenizer.padding_side = "left"

model = tfs.AutoModelForSequenceClassification.from_pretrained(model_ckpt, **model_kwargs)

inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
		logits = model(**inputs).logits

preds = torch.softmax(logits, dim=-1)

# Map raw predictions to languages
id2lang = model.config.id2label
vals, idxs = torch.max(preds, dim=1)

# Dict indexed by text position
results = {i: (id2lang[k.item()], v.item()) for i, (k, v) in enumerate(zip(idxs, vals))}
print(results)
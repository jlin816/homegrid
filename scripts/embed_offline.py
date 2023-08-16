import pickle
from transformers import T5Tokenizer, T5EncoderModel
import sys
import torch

def embed_t5(sentences, outf):
  tokenizer = T5Tokenizer.from_pretrained("t5-small", use_legacy=False)
  model = T5EncoderModel.from_pretrained("t5-small")
  embed_cache = {}
  token_cache = {}
  lens = []
  for s in sentences:
    s = s.strip()
    tokens = tokenizer(s, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
      embed = model(**tokens).last_hidden_state.squeeze(0).cpu().numpy()
    token_cache[s] = tokens["input_ids"].cpu().numpy()[0]
    embed_cache[s] = embed
    lens.append(len(token_cache[s]))

  token_cache["<pad>"] = tokenizer.pad_token_id
  with torch.no_grad():
    embed_cache["<pad>"] = model(
      **tokenizer("<pad>", add_special_tokens=False, return_tensors="pt")
    ).last_hidden_state[0][0].cpu().numpy()
  assert embed_cache["<pad>"].shape == (model.config.d_model,), \
    embed_cache["<pad>"].shape
  with open(outf, "wb") as f:
    pickle.dump((token_cache, embed_cache), f)
  print(lens)
  print("max len: ", max(lens))

def embed_st(sentences, outf):
  from sentence_transformers import SentenceTransformer
  model = SentenceTransformer("all-distilroberta-v1")
  embeds = {}
  for sent in sentences:
    embeds[sent] = model.encode([sent])[0]
  embeds["<pad>"]= model.encode(["<pad>"])[0]
  assert embeds["<pad>"].shape == (768,), \
    embeds["<pad>"].shape
  with open(outf, "wb") as f:
    pickle.dump((None, embeds), f)

if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--infile", help="file with strings to embed",
      default="homegrid/homegrid_sentences.txt"
  )
  parser.add_argument(
      "--outfile", help="file to output token and embedding cache",
      default="homegrid/homegrid_embeds.pkl"
  )

  parser.add_argument(
      "--model",
      help="model to use for embedding (t5 or sentence)",
      choices=["t5", "sentence"],
      default="t5",
  )

  args = parser.parse_args()
  with open(args.infile) as f:
    sentences = []
    for line in f:
      sentences.append(line.strip())

  if args.model == "t5":
    embed_t5(sentences, args.outfile)
  elif args.model == "st":
    embed_st(sentences, args.outfile)

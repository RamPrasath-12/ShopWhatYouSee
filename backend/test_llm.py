from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model = "google/flan-t5-base"
tok = AutoTokenizer.from_pretrained(model)
m = AutoModelForSeq2SeqLM.from_pretrained(model)

prompt = "Return JSON: {\"hello\": \"world\"}"
inputs = tok(prompt, return_tensors="pt")
out = m.generate(**inputs)
print(tok.decode(out[0], skip_special_tokens=True))

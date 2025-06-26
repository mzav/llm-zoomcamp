import os
import numpy as np
from fastembed import TextEmbedding

os.environ["FASTEMBED_CACHE_PATH"] = "cache/"
model_name = "jinaai/jina-embeddings-v2-small-en"

text= "I just discovered the course. Can I join now?"


emb_model = TextEmbedding(model_name = model_name)
print(np.min((next(emb_model.embed(text))))) # Answer 1

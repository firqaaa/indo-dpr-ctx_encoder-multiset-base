---
pipeline_tag: feature-extraction
tags:
- feature-extraction
- transformers
license: apache-2.0
language:
- id
metrics:
- accuracy
- f1
- precision
- recall
datasets:
- squad_v2
- natural_questions
---
### indo-dpr-ctx_encoder-multiset-base
<p style="font-size:16px">Indonesian Dense Passage Retrieval trained on translated SQuADv2.0 and Natural Question dataset in DPR format.</p>


### Evaluation 

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| hard_negative | 0.9961 | 0.9961 | 0.9961 | 384778 |
| positive | 0.8783 | 0.8783 | 0.8783 | 12414 |

| Metric | Value |
|--------|-------|
| Loss | 0.0220 |
| Accuracy | 0.9924 |
| Macro Average | 0.9372 |
| Weighted Average | 0.9924 |
| Accuracy and F1 | 0.9353 |
| Average Rank | 0.2194 |


<p style="font-size:16px">Note: This report is for evaluation on the dev set, after 27288 batches.</p>

### Usage

```python
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer

tokenizer = DPRContextEncoderTokenizer.from_pretrained('firqaaa/indo-dpr-ctx_encoder-multiset-base')
model = DPRContextEncoder.from_pretrained('firqaaa/indo-dpr-ctx_encoder-multiset-base')
input_ids = tokenizer("Siapa nama pengarang manga Yu-Gi-Oh?", return_tensors='pt')["input_ids"]
embeddings = model(input_ids).pooler_output
```

You can use it using `haystack` as follows:

```
from haystack.nodes import DensePassageRetriever
from haystack.document_stores import InMemoryDocumentStore

retriever = DensePassageRetriever(document_store=InMemoryDocumentStore(),
                                  query_embedding_model="firqaaa/indo-dpr-ctx_encoder-multiset-base",
                                  passage_embedding_model="firqaaa/indo-dpr-ctx_encoder-multiset-base",
                                  max_seq_len_query=64,
                                  max_seq_len_passage=256,
                                  batch_size=16,
                                  use_gpu=True,
                                  embed_title=True,
                                  use_fast_tokenizers=True)
```

# CONVINV: Interpreting Conversational Dense Retrieval by Rewriting-Enhanced Inversion of Session Embedding

### Introduction

CONVINV: a simple and effective approach aiming to shed light on the opacity problem of conversational dense retrieval. CONVINV demystifies the opaque conversational session embeddings by transforming them into ex plicitly interpretable text while faithfully maintain ing their retrieval performance as much as possible. This transformation allows us to intuitively deci pher the characteristics of behaviors of different conversational dense retrieval models.

### Training

To train the Vec2Text model, run:

```python
python run.py
```

### Inference

To use the trained model to interpret session embeddings, run:

```python
python invert_GTR_with_T5QR_batch.py
```


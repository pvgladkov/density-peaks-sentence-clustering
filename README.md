# Density Peaks Sentence Clustering

[Clustering Sentences with Density Peaks for Multi-document Summarization](https://www.aclweb.org/anthology/N15-1136/)


## CNN/Daily Mail

|Model|ROUGE-1|ROUGE-2|ROUGE-L|
|---|---|---|---|
|Lead 75|13.13|3.85|13.25|
|Lead 3 sent.|31.95|13.01|30.45|
|DPSC||||

## Gigaword

|Model|ROUGE-1|ROUGE-2|ROUGE-L|
|---|---|---|---|
|Lead 75|15.38|4.43|14.71|
|Lead 3 sent.|15.58|4.45|14.84|
|DPSC||||

## Run tests

```bash
pytest tests
```
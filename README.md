# Density Peaks Sentence Clustering

Yunqing Xia, Yi Liu, Yang Zhang, Wenmin Wang "**Clustering Sentences with Density Peaks for Multi-document Summarization**" (2015) [https://www.aclweb.org/anthology/N15-1136/](https://www.aclweb.org/anthology/N15-1136/).

The results differ from ones in the paper.

## CNN/Daily Mail

|Model|ROUGE-1|ROUGE-2|ROUGE-L|
|---|---|---|---|
|Lead 75|13.13|3.85|13.25|
|Lead 3 sent.|31.95|13.01|30.45|
|DPSC|21.67|7.95|23.31|

## Gigaword

|Model|ROUGE-1|ROUGE-2|ROUGE-L|
|---|---|---|---|
|Lead 75|15.38|4.43|14.71|
|Lead 3 sent.|15.58|4.45|14.84|
|DPSC|15.57|4.44|14.83|

## Run tests

```bash
pytest tests
```
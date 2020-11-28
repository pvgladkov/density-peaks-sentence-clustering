from nltk.tokenize import sent_tokenize
from rouge import Rouge

from data import datasets


def lead_75(t):
    return t[:75]


def lead_3s(t):
    sentences = [s for s in sent_tokenize(t)]
    return '. '.join(sentences[:3])


if __name__ == '__main__':

    for name, dataset in datasets.items():
        hyps_75 = []
        hyps_3s = []
        refs = []

        for text, ref in dataset:
            text = text.numpy()
            hyps_75.append(lead_75(str(text)))
            hyps_3s.append(lead_3s(str(text)))

            ref = ref.numpy()
            refs.append(str(ref))

        metric = Rouge()
        score_75 = metric.get_scores(hyps_75, refs, avg=True)
        score_3s = metric.get_scores(hyps_3s, refs, avg=True)
        print(name)
        print(f'lead 75: {score_75}')
        print(f'lead 3s: {score_3s}')

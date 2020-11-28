import gensim.downloader
from rouge import Rouge

from data import datasets
from dpsc.dpsc import overall_score, select_top


def summarize(x, vectors):
    scores = overall_score(x, vectors)
    return '. '.join(select_top(scores, 10))


if __name__ == '__main__':

    w2v_vectors = gensim.downloader.load('word2vec-google-news-300')

    results = {}

    for name, dataset in datasets.items():
        hyps = []
        refs = []

        for text, ref in dataset:
            text = text.numpy()
            hyps.append(summarize(str(text), w2v_vectors))

            ref = ref.numpy()
            refs.append(str(ref))

        metric = Rouge()
        score = metric.get_scores(hyps, refs, avg=True)
        results[name] = score

    for name in results:
        print(name)
        print(f'dpsc: {results[name]}')

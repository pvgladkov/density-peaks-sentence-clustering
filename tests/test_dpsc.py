import os

import numpy as np
import pytest
from am.lib.utils.obj_dump import load_object

import config
from antifraud.common.summarization.dpsc import (
    diversity_score_,
    effective_length,
    length_score_,
    overall_score,
    real_length,
    represent_score_,
    select_top,
    similarity_matrix,
    split_into_sentences,
)

path = os.path.join(config.data_path, 'models', 'text_detector')
VECTORIZER = load_object(os.path.join(path, 'word2vec', 'word2vec.pkl'))


@pytest.mark.parametrize(
    'sim_matrix, delta, expected_result', [
        ([[1, 1], [1, 1]], 0, [0.5, 0.5]),
        ([[1, 1], [1, 1]], 1, [1e-10, 1e-10]),
        ([[1, 0], [0, 1]], 0, [1e-10, 1e-10]),
        ([[1, 2], [0, 1]], 1, [0.5, 1e-10]),
    ],
)
def test_rep_scoring(sim_matrix, delta, expected_result):
    assert np.array_equal(represent_score_(sim_matrix, delta), np.array(expected_result))


@pytest.mark.parametrize(
    'sim_matrix, s_rep_vector, expected_result', [
        ([[1, 1], [1, 1]], [0.5, 0.5], [1e-10, 1e-10]),
        ([[1, 2], [0, 1]], [0.5, 0], [-1, 1]),
    ],
)
def test_div_scoring(sim_matrix, s_rep_vector, expected_result):
    assert np.array_equal(diversity_score_(sim_matrix, s_rep_vector), np.array(expected_result))


@pytest.mark.parametrize(
    'effective_lens, real_lens, expected_result', [
        ([4, 10], [2, 5], [0.36651629, 0.]),
    ],
)
def test_len_scoring(effective_lens, real_lens, expected_result):
    scores = length_score_(effective_lens, real_lens)
    assert len(scores) == len(expected_result)


@pytest.mark.parametrize(
    'text, expected_result', [
        ('iphone 8, new. 16 gb.', ['iphone 8, new', '16 gb.']),
        ('iphone 8, new\n16 gb.', ['iphone 8, new', '16 gb.']),
        ('iphone 8, new\n 16 gb.', ['iphone 8, new', '16 gb.']),
        ('iphone 8, new; 16 gb.', ['iphone 8, new', '16 gb.']),
        ('iphone 8, новый; 16 гб.', ['iphone 8, новый', '16 гб.']),
    ],
)
def test_split(text, expected_result):
    assert split_into_sentences(text) == expected_result


@pytest.mark.parametrize(
    'text, expected_result', [
        ('еще нет', 0),
        ('еще нет телефона', 1),
        ('продажа телефона', 2),
    ],
)
def test_eff_len(text, expected_result):
    assert effective_length(text) == expected_result


@pytest.mark.parametrize(
    'text, expected_result', [
        ('еще нет', 2),
        ('еще нет телефона', 3),
        ('продажа телефона', 2),
    ],
)
def test_real_len(text, expected_result):
    assert real_length(text) == expected_result


@pytest.mark.parametrize(
    'text, expected_result', [
        ('продаю телефон. продаю телефон', [[1.0, 1.0], [1.0, 1.0]]),
    ],
)
def test_sim_matrix(text, expected_result):
    expected_result = np.array(expected_result, dtype=float)
    result = similarity_matrix(text, VECTORIZER)
    assert expected_result.shape == result.shape
    for i in range(len(result)):
        for j in range(len(result)):
            assert round(result[i, j], 3) == round(expected_result[i, j], 3)


@pytest.mark.parametrize(
    'text, expected_result', [
        (
            'Продам iphone 8. В хорошем состоянии, все документы на месте. \
        В нашем магазине только лучшие телефоны.', 3,
        ),
        (
            'Гаражные ворота НОВЫЕ с рамой!\nРазмеры: \nширина с рамой 2.54м \nширина створок 2.42м\nВысота с \
            рамой 2.24м\nВысота створок 2.15м\nСамовывоз\nВорота демонтированны и готовы к продаже', 8,
        ),
    ],
)
def test_overall_score(text, expected_result):
    score = overall_score(text, VECTORIZER)
    assert len(score) == expected_result


@pytest.mark.parametrize(
    'sentences_with_score, top, expected_result', [
        ([('s0', 0), ('s1', -1), ('s2', 1)], 1, ['s2']),
        ([('s0', 0), ('s1', -1), ('s2', 1)], 2, ['s0', 's2']),
        ([('s0', 0), ('s1', -np.inf), ('s2', 1)], 2, ['s0', 's2']),
        ([('s0', 0), ('s1', np.nan), ('s2', 1)], 2, ['s0', 's2']),
    ],
)
def test_select(sentences_with_score, top, expected_result):
    assert select_top(sentences_with_score, top) == expected_result

import tensorflow_datasets as tfds

datasets = {
    'cnn_dailymail': tfds.load('cnn_dailymail', split='test', as_supervised=True, shuffle_files=False),
    'gigaword': tfds.load('gigaword', split='test', as_supervised=True, shuffle_files=False),
}
import tensorflow as tf
import numpy as np
import pandas as pd
import logging

logger = tf.get_logger()
logger.setLevel(logging.INFO)

pd.set_option("display.max_rows", 10, "display.max_columns", 100)
all_samples = pd.read_csv('file:///Users/zhewang/Workspace/SparrowRecSys/src/main/resources/webroot/sampledata'
                          '/modelSamples.csv',
                          dtype={'movieGenre1': str, 'movieGenre2': str, 'movieGenre3': str})

all_samples.fillna({'movieGenre1': '', 'movieGenre2': '', 'movieGenre3': '',
                    'userGenre1': '', 'userGenre2': '', 'userGenre3': '', 'userGenre4': '', 'userGenre5': '',
                    'userRatedMovie1': 0},
                   inplace=True)

all_samples['userRatedMovie1'] = all_samples['userRatedMovie1'].apply(np.int64)
all_samples['releaseYear'] = all_samples['releaseYear'].apply(lambda release_year: (release_year - 1990) / 10)
all_samples['userAvgReleaseYear'] = all_samples['userAvgReleaseYear'].apply(
    lambda release_year: (release_year - 1990) / 10)
all_samples['movieRatingCount'] = all_samples['movieRatingCount'].apply(lambda count: count / 10000)
all_samples['userRatingCount'] = all_samples['userRatingCount'].apply(lambda count: count / 50)
all_samples['userReleaseYearStddev'] = all_samples['userReleaseYearStddev'].apply(lambda count: count / 10)

train_samples = all_samples.sample(frac=0.8, random_state=200)  # random state is a seed value
test_samples = all_samples.drop(train_samples.index)

y_train = train_samples.pop('label')
y_test = test_samples.pop('label')

for col in train_samples.columns:
    print('column', col, ':', type(train_samples[col][0]))
print(train_samples.head())
print(y_train.head())


def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds

    return input_function


train_input_fn = make_input_fn(train_samples, y_train, num_epochs=10, shuffle=True, batch_size=12)
eval_input_fn = make_input_fn(test_samples, y_test, num_epochs=1, shuffle=False)

CATEGORICAL_COLUMNS = ['userGenre1', 'userGenre2', 'userGenre3', 'userGenre4', 'userGenre5',
                       'movieGenre1', 'movieGenre2', 'movieGenre3']

NUMERIC_COLUMNS = ['movieAvgRating', 'movieRatingStddev', 'movieRatingCount',
                   'userAvgRating', 'userRatingStddev', 'userRatingCount',
                   'releaseYear', 'userAvgReleaseYear', 'userReleaseYearStddev']

genre_vocab = ['Film-Noir', 'Action', 'Adventure', 'Horror', 'Romance', 'War', 'Comedy', 'Western', 'Documentary',
               'Sci-Fi', 'Drama', 'Thriller',
               'Crime', 'Fantasy', 'Animation', 'IMAX', 'Mystery', 'Children', 'Musical']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    cat_feature_col = tf.feature_column.categorical_column_with_vocabulary_list(feature_name, genre_vocab)
    feature_columns.append(tf.feature_column.embedding_column(cat_feature_col, 10))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

user_col = tf.feature_column.categorical_column_with_identity(key='userId', num_buckets=30001)
user_emb_col = tf.feature_column.embedding_column(user_col, 10)
feature_columns.append(user_emb_col)

movie_feature = tf.feature_column.categorical_column_with_identity(key='movieId', num_buckets=1001)
rated_movie_feature = tf.feature_column.categorical_column_with_identity(key='userRatedMovie1', num_buckets=1001)
crossed_feature = tf.feature_column.crossed_column([movie_feature, rated_movie_feature], 10000)

movie_emb_col = tf.feature_column.embedding_column(movie_feature, 10)
feature_columns.append(movie_emb_col)

wide_n_deep = tf.estimator.DNNLinearCombinedClassifier(
    # wide settings
    linear_feature_columns=[crossed_feature],
    linear_optimizer=tf.keras.optimizers.Ftrl(),
    # deep settings
    dnn_feature_columns=feature_columns,
    dnn_hidden_units=[128, 128],
    dnn_optimizer=tf.keras.optimizers.Adagrad())

# estimator = tf.estimator.DNNClassifier(
#    hidden_units=[30, 30], feature_columns=feature_columns, optimizer='Adagrad', activation_fn=tf.nn.relu)
# logging_hook = tf.estimator.LoggingTensorHook({"loss": loss,
#                                               "accuracy": accuracy}, every_n_iter=10)

wide_n_deep.train(train_input_fn)
metrics = wide_n_deep.evaluate(input_fn=eval_input_fn)
print(metrics)

predictions = wide_n_deep.predict(input_fn=eval_input_fn)
print(predictions)

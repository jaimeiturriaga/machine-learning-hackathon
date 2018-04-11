import tensorflow as tf
import argparse
import itertools
import numpy as np
import store_data

NDT = tf.float64

EMBEDDING_DIM_ITEMS = 9

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int, help='number of training steps')
parser.add_argument('--test_percent', default=20, type=float, help='percentage of examples to use as tests')
parser.add_argument('--test_selection', default=0, type=int, help='0 - random, 1 - end')
parser.add_argument('--reset_file', default=0, type=int, help='reset processed data file')
parser.add_argument('--model_dir', default=None, type=str, help='directory to put model in')
parser.add_argument('--train_eval', default=0, type=int, help='evaluate on training data')


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    data_set = tf.contrib.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    data_set = data_set.shuffle(1000).repeat().batch(batch_size)

    # Return the read end of the pipeline.
    return data_set.make_one_shot_iterator().get_next()


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    data_set = tf.contrib.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    data_set = data_set.batch(batch_size)

    # Return the read end of the pipeline.
    return data_set.make_one_shot_iterator().get_next()


def main(argv):

    args = parser.parse_args(argv[1:])

    (train_x, train_y), (test_x, test_y) = store_data.load_data(args.test_percent, args.test_selection)
    prices = test_x["unit_price"]

    # IDs
    stores = tf.feature_column.categorical_column_with_hash_bucket("store", 20, dtype=tf.int32)
    item = tf.feature_column.categorical_column_with_hash_bucket("item", 100, dtype=tf.int32)

    # Categories
    department = tf.feature_column.categorical_column_with_identity("department", 4)
    promotion = tf.feature_column.categorical_column_with_identity("promotion_type", 7)

    # Date processing
    month = tf.feature_column.categorical_column_with_identity("month", 12)
    year = tf.feature_column.numeric_column("year", dtype=NDT)
    day = tf.feature_column.categorical_column_with_identity("day", 31)

    total_time = tf.feature_column.numeric_column("total_time", dtype=NDT)
    sin_time = tf.feature_column.numeric_column("sin_time", dtype=NDT)
    cos_time = tf.feature_column.numeric_column("cos_time", dtype=NDT)
    sin_day = tf.feature_column.numeric_column("sin_day", dtype=NDT)
    cos_day = tf.feature_column.numeric_column("cos_day", dtype=NDT)

    # Embedding columns
    it_emb = tf.feature_column.embedding_column(item, EMBEDDING_DIM_ITEMS)

    # Indicator columns
    ind = {
        "stores":tf.feature_column.indicator_column(stores),
        "department":tf.feature_column.indicator_column(department),
        "promo":tf.feature_column.indicator_column(promotion),
        "month": tf.feature_column.indicator_column(month),
        "day": tf.feature_column.indicator_column(day)
    }

    #TF columns
    holidays =  [tf.feature_column.numeric_column("christmas"),
                 tf.feature_column.numeric_column("halloween"),
                 tf.feature_column.numeric_column("valentine"),
                 tf.feature_column.numeric_column("thanksgiving"),
                 tf.feature_column.numeric_column("st_patricks")]



    # Numerical values
    unit_price = tf.feature_column.numeric_column("unit_price", dtype=NDT)

    # Feature columns
    feature_cols = [
            it_emb, ind["stores"], ind["department"], ind["promo"], unit_price, sin_time, cos_time, sin_day, cos_day,
        ]
    feature_cols.extend(holidays)

    # Regressor model
    basic_estimator = tf.estimator.DNNRegressor(
        feature_columns=feature_cols,
        hidden_units=[1024, 512, 256, 128, 64],
        model_dir=args.model_dir,
        weight_column=unit_price
    )

    # Training
    basic_estimator.train(
        input_fn=lambda:train_input_fn(train_x, train_y, args.batch_size),
        steps=args.train_steps)

    # Evaluate the model on testing data.
    eval_result = basic_estimator.evaluate(input_fn=lambda: eval_input_fn(test_x, test_y, args.batch_size))
    print("Test eval:")
    print(eval_result)

    # Evaluate the model on training data
    if args.train_eval:
        eval_result = basic_estimator.evaluate(input_fn=lambda: eval_input_fn(train_x, train_y, args.batch_size))
        print("Train eval:")
        print(eval_result)

    # Get prediction

    prediction = basic_estimator.predict(input_fn=lambda: eval_input_fn(test_x, None, batch_size=args.batch_size))
    predictions = list(p['predictions'] for p in prediction)
    # Check error
    error, w_error, w_quantity = 0, 0, 0
    for value, expec, price in zip(predictions, test_y, prices):
        w_error += price * abs(value - expec)
        w_quantity += price * expec
        error += abs((value - expec)/expec * 100)

    print("Average error: {0}%".format(error/len(test_y)))
    print("Total weighted error: {0}".format(w_error))
    print("WAPE: {0}%".format(w_error/w_quantity))


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
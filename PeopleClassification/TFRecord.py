import tensorflow as tf
import peopleConfig as cfg

def get_data_size(tfrecords_filename):
    record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
    counter = 0
    for string_record in record_iterator:
       counter += 1
    return counter

def read_and_decode(filename_queue, readSize):
    # Construct TFRecordReader
    reader = tf.TFRecordReader()

    # Read TFRecords Data
    _, serialized_example = reader.read(filename_queue)

    # Read One Example
    features = tf.parse_single_example(serialized_example,
        features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'image_string': tf.FixedLenFeature([], tf.string),
        'label_uppercolor': tf.FixedLenFeature([], tf.int64),
        'label_uppersleeve': tf.FixedLenFeature([], tf.int64),
        'label_lowercolor': tf.FixedLenFeature([], tf.int64),
        'label_lowertype': tf.FixedLenFeature([], tf.int64),
        'label_lowerlength': tf.FixedLenFeature([], tf.int64),
        'label_backpack': tf.FixedLenFeature([], tf.int64),
        'label_handbag': tf.FixedLenFeature([], tf.int64),
        'label_umbrella': tf.FixedLenFeature([], tf.int64),
        'label_hat': tf.FixedLenFeature([], tf.int64),
        'label_age': tf.FixedLenFeature([], tf.int64),
        'label_gender': tf.FixedLenFeature([], tf.int64)
        })

    # Translate Image To uint8 Tensor
    image = tf.decode_raw(features['image_string'], tf.uint8)
    image = tf.reshape(image, [cfg.IMG_HEIGHT,cfg.IMG_WIDTH,cfg.NUM_CHANNELS])

    # Translate Label To int64 Tensor
    upperColor_Label = tf.cast(features['label_uppercolor'], tf.int64)
    upperSleeve_Label = tf.cast(features['label_uppersleeve'], tf.int64)
    lowerColor_Label = tf.cast(features['label_lowercolor'], tf.int64)
    lowerType_Label = tf.cast(features['label_lowertype'], tf.int64)
    lowerLength_Label = tf.cast(features['label_lowerlength'], tf.int64)
    backpack_Label = tf.cast(features['label_backpack'], tf.int64)
    handbag_Label = tf.cast(features['label_handbag'], tf.int64)
    umbrella_Label = tf.cast(features['label_umbrella'], tf.int64)
    hat_Label = tf.cast(features['label_hat'], tf.int64)
    age_Label = tf.cast(features['label_age'], tf.int64)
    gender_Label = tf.cast(features['label_gender'], tf.int64)

    # Shuffle Dataset
    images, upperColor, upperSleeve, lowerColor, lowerType, lowerLength, backpack, handbag, umbrella, hat, age, gender=\
        tf.train.shuffle_batch([image, upperColor_Label, upperSleeve_Label, lowerColor_Label, lowerType_Label,  \
        lowerLength_Label, backpack_Label, handbag_Label, umbrella_Label, hat_Label, age_Label, gender_Label],  \
        batch_size=readSize, capacity=6400, min_after_dequeue=800, num_threads=1, allow_smaller_final_batch=True)
    return images, upperColor, upperSleeve, lowerColor, lowerType, lowerLength, backpack, handbag, umbrella, hat, age, gender

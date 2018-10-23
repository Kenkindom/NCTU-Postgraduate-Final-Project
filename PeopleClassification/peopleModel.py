import tensorflow as tf
import numpy as np

import peopleConfig as cfg

# Input Size
INPUT_WIDTH = 112
INPUT_HEIGHT = 112
NUM_CHANNELS = 3

# Model Size
CONV1_SIZE = 3
CONV1_DEEP = 64

CONV2_SIZE = 3
CONV2_DEEP = 128

CONV3_SIZE = 3
CONV3_DEEP = 256

CONV4_SIZE = 3
CONV4_DEEP = 512

CONV5_SIZE = 3
CONV5_DEEP = 512

FC_SIZE = 2048

MODEL_NAME = 'People Attribute Model'

def interface(input, train, regularizer):
    with tf.variable_scope('Layer1_Convonlution1'):
        conv1_weights = tf.get_variable("weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.name_scope('Layer2_MaxPooling1'):
        pool1 = tf.nn.max_pool(relu1, ksize = [1,2,2,1],strides=[1,2,2,1],padding="SAME")

    with tf.variable_scope('Layer3_Convonlution2'):
        conv2_weights = tf.get_variable("weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope('Layer4_MaxPooling2'):
        pool2 = tf.nn.max_pool(relu2, ksize = [1,2,2,1],strides=[1,2,2,1],padding="SAME")

    with tf.variable_scope('Layer5_Convonlution3'):
        conv3_weights = tf.get_variable("weight", [CONV3_SIZE, CONV3_SIZE, CONV2_DEEP, CONV3_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable("bias", [CONV3_DEEP], initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))

    with tf.name_scope('Layer6_MaxPooling3'):
        pool3 = tf.nn.max_pool(relu3, ksize = [1,2,2,1],strides=[1,2,2,1],padding="SAME")

    with tf.variable_scope('Layer7_Convonlution4'):
        conv4_weights = tf.get_variable("weight", [CONV4_SIZE, CONV4_SIZE, CONV3_DEEP, CONV4_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv4_biases = tf.get_variable("bias", [CONV4_DEEP], initializer=tf.constant_initializer(0.0))
        conv4 = tf.nn.conv2d(pool3, conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))

    with tf.name_scope('Layer8_MaxPooling4'):
        pool4 = tf.nn.max_pool(relu4, ksize = [1,2,2,1],strides=[1,2,2,1],padding="SAME")
        pool_shape = pool4.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        reshaped = tf.reshape(pool4, [-1, nodes])

    """with tf.variable_scope('Layer9_Convonlution5'):
        conv5_weights = tf.get_variable("weight", [CONV5_SIZE, CONV5_SIZE, CONV4_DEEP, CONV5_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv5_biases = tf.get_variable("bias", [CONV5_DEEP], initializer=tf.constant_initializer(0.0))
        conv5 = tf.nn.conv2d(pool4, conv5_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu5 = tf.nn.relu(tf.nn.bias_add(conv5, conv5_biases))

    with tf.name_scope('Layer10_MaxPooling5'):
        pool5 = tf.nn.max_pool(relu5, ksize = [1,2,2,1],strides=[1,2,2,1],padding="SAME")
        pool_shape = pool5.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        reshaped = tf.reshape(pool5, [-1, nodes])"""

    # UpperColor
    with tf.variable_scope('UpperColor_FullConnect1'):
        upperColor_fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('upperColor_losses', regularizer(upperColor_fc1_weights))
        upperColor_fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))

        upperColor_fc1 = tf.nn.relu(tf.matmul(reshaped, upperColor_fc1_weights) + upperColor_fc1_biases)
        if train: upperColor_fc1 = tf.nn.dropout(upperColor_fc1, 0.7)

    with tf.variable_scope('UpperColor_FullConnect2'):
        upperColor_fc2_weights = tf.get_variable("weight", [FC_SIZE, cfg.UPPER_COLOR_NUM],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('upperColor_losses', regularizer(upperColor_fc2_weights))
        upperColor_fc2_biases = tf.get_variable("bias", [cfg.UPPER_COLOR_NUM], initializer=tf.constant_initializer(0.1))
        upperColor_Out = tf.matmul(upperColor_fc1, upperColor_fc2_weights) + upperColor_fc2_biases

    # UpperSleeve
    with tf.variable_scope('UpperSleeve_FullConnect1'):
        upperSleeve_fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('upperSleeve_losses', regularizer(upperSleeve_fc1_weights))
        upperSleeve_fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))

        upperSleeve_fc1 = tf.nn.relu(tf.matmul(reshaped, upperSleeve_fc1_weights) + upperSleeve_fc1_biases)
        if train: upperSleeve_fc1 = tf.nn.dropout(upperSleeve_fc1, 0.7)

    with tf.variable_scope('UpperSleeve_FullConnect2'):
        upperSleeve_fc2_weights = tf.get_variable("weight", [FC_SIZE, cfg.UPPER_SLEEVE_NUM],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('upperSleeve_losses', regularizer(upperSleeve_fc2_weights))
        upperSleeve_fc2_biases = tf.get_variable("bias", [cfg.UPPER_SLEEVE_NUM], initializer=tf.constant_initializer(0.1))
        upperSleeve_Out = tf.matmul(upperSleeve_fc1, upperSleeve_fc2_weights) + upperSleeve_fc2_biases

    # LowerColor
    with tf.variable_scope('LowerColor_FullConnect1'):
        lowerColor_fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('lowerColor_losses', regularizer(lowerColor_fc1_weights))
        lowerColor_fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))

        lowerColor_fc1 = tf.nn.relu(tf.matmul(reshaped, lowerColor_fc1_weights) + lowerColor_fc1_biases)
        if train: lowerColor_fc1 = tf.nn.dropout(lowerColor_fc1, 0.7)

    with tf.variable_scope('LowerColor_FullConnect2'):
        lowerColor_fc2_weights = tf.get_variable("weight", [FC_SIZE, cfg.LOWER_COLOR_NUM],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('lowerColor_losses', regularizer(lowerColor_fc2_weights))
        lowerColor_fc2_biases = tf.get_variable("bias", [cfg.LOWER_COLOR_NUM], initializer=tf.constant_initializer(0.1))
        lowerColor_Out = tf.matmul(lowerColor_fc1, lowerColor_fc2_weights) + lowerColor_fc2_biases

    # LowerType
    with tf.variable_scope('LowerType_FullConnect1'):
        lowerType_fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('lowerType_losses', regularizer(lowerType_fc1_weights))
        lowerType_fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))

        lowerType_fc1 = tf.nn.relu(tf.matmul(reshaped, lowerType_fc1_weights) + lowerType_fc1_biases)
        if train: lowerType_fc1 = tf.nn.dropout(lowerType_fc1, 0.7)

    with tf.variable_scope('LowerType_FullConnect2'):
        lowerType_fc2_weights = tf.get_variable("weight", [FC_SIZE, cfg.LOWER_TYPE_NUM],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('lowerType_losses', regularizer(lowerType_fc2_weights))
        lowerType_fc2_biases = tf.get_variable("bias", [cfg.LOWER_TYPE_NUM], initializer=tf.constant_initializer(0.1))
        lowerType_Out = tf.matmul(lowerType_fc1, lowerType_fc2_weights) + lowerType_fc2_biases

    # LowerLength
    with tf.variable_scope('LowerLength_FullConnect1'):
        lowerLength_fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('lowerLength_losses', regularizer(lowerLength_fc1_weights))
        lowerLength_fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))

        lowerLength_fc1 = tf.nn.relu(tf.matmul(reshaped, lowerLength_fc1_weights) + lowerLength_fc1_biases)
        if train: lowerLength_fc1 = tf.nn.dropout(lowerLength_fc1, 0.7)

    with tf.variable_scope('LowerLength_FullConnect2'):
        lowerLength_fc2_weights = tf.get_variable("weight", [FC_SIZE, cfg.LOWER_LENGTH_NUM],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('lowerLength_losses', regularizer(lowerLength_fc2_weights))
        lowerLength_fc2_biases = tf.get_variable("bias", [cfg.LOWER_LENGTH_NUM], initializer=tf.constant_initializer(0.1))
        lowerLength_Out = tf.matmul(lowerLength_fc1, lowerLength_fc2_weights) + lowerLength_fc2_biases

    # Backpack
    with tf.variable_scope('Backpack_FullConnect1'):
        backpack_fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('backpack_losses', regularizer(backpack_fc1_weights))
        backpack_fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))

        backpack_fc1 = tf.nn.relu(tf.matmul(reshaped, backpack_fc1_weights) + backpack_fc1_biases)
        if train: backpack_fc1 = tf.nn.dropout(backpack_fc1, 0.7)

    with tf.variable_scope('Backpack_FullConnect2'):
        backpack_fc2_weights = tf.get_variable("weight", [FC_SIZE, cfg.BACKPACK_NUM],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('backpack_losses', regularizer(backpack_fc2_weights))
        backpack_fc2_biases = tf.get_variable("bias", [cfg.BACKPACK_NUM], initializer=tf.constant_initializer(0.1))
        backpack_Out = tf.matmul(backpack_fc1, backpack_fc2_weights) + backpack_fc2_biases

    # Handbag
    with tf.variable_scope('Handbag_FullConnect1'):
        handbag_fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('handbag_losses', regularizer(handbag_fc1_weights))
        handbag_fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))

        handbag_fc1 = tf.nn.relu(tf.matmul(reshaped, handbag_fc1_weights) + handbag_fc1_biases)
        if train: handbag_fc1 = tf.nn.dropout(handbag_fc1, 0.7)

    with tf.variable_scope('Handbag_FullConnect2'):
        handbag_fc2_weights = tf.get_variable("weight", [FC_SIZE, cfg.HANDBAG_NUM],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('handbag_losses', regularizer(handbag_fc2_weights))
        handbag_fc2_biases = tf.get_variable("bias", [cfg.HANDBAG_NUM], initializer=tf.constant_initializer(0.1))
        handbag_Out = tf.matmul(handbag_fc1, handbag_fc2_weights) + handbag_fc2_biases

    # Umbrella
    with tf.variable_scope('Umbrella_FullConnect1'):
        umbrella_fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('umbrella_losses', regularizer(umbrella_fc1_weights))
        umbrella_fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))

        umbrella_fc1 = tf.nn.relu(tf.matmul(reshaped, umbrella_fc1_weights) + umbrella_fc1_biases)
        if train: umbrella_fc1 = tf.nn.dropout(umbrella_fc1, 0.7)

    with tf.variable_scope('Umbrella_FullConnect2'):
        umbrella_fc2_weights = tf.get_variable("weight", [FC_SIZE, cfg.UMBRELLA_NUM],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('umbrella_losses', regularizer(umbrella_fc2_weights))
        umbrella_fc2_biases = tf.get_variable("bias", [cfg.UMBRELLA_NUM], initializer=tf.constant_initializer(0.1))
        umbrella_Out = tf.matmul(umbrella_fc1, umbrella_fc2_weights) + umbrella_fc2_biases

    # Hat
    with tf.variable_scope('Hat_FullConnect1'):
        hat_fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('hat_losses', regularizer(hat_fc1_weights))
        hat_fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))

        hat_fc1 = tf.nn.relu(tf.matmul(reshaped, hat_fc1_weights) + hat_fc1_biases)
        if train: hat_fc1 = tf.nn.dropout(hat_fc1, 0.7)

    with tf.variable_scope('Hat_FullConnect2'):
        hat_fc2_weights = tf.get_variable("weight", [FC_SIZE, cfg.HAT_NUM],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('hat_losses', regularizer(hat_fc2_weights))
        hat_fc2_biases = tf.get_variable("bias", [cfg.HAT_NUM], initializer=tf.constant_initializer(0.1))
        hat_Out = tf.matmul(hat_fc1, hat_fc2_weights) + hat_fc2_biases

    # Age
    with tf.variable_scope('Age_FullConnect1'):
        age_fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('age_losses', regularizer(age_fc1_weights))
        age_fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))

        age_fc1 = tf.nn.relu(tf.matmul(reshaped, age_fc1_weights) + age_fc1_biases)
        if train: age_fc1 = tf.nn.dropout(age_fc1, 0.7)

    with tf.variable_scope('Age_FullConnect2'):
        age_fc2_weights = tf.get_variable("weight", [FC_SIZE, cfg.AGE_NUM],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('age_losses', regularizer(age_fc2_weights))
        age_fc2_biases = tf.get_variable("bias", [cfg.AGE_NUM], initializer=tf.constant_initializer(0.1))
        age_Out = tf.matmul(age_fc1, age_fc2_weights) + age_fc2_biases

    # Gender
    with tf.variable_scope('Gender_FullConnect1'):
        gender_fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('gender_losses', regularizer(gender_fc1_weights))
        gender_fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))

        gender_fc1 = tf.nn.relu(tf.matmul(reshaped, gender_fc1_weights) + gender_fc1_biases)
        if train: gender_fc1 = tf.nn.dropout(gender_fc1, 0.7)

    with tf.variable_scope('Gender_FullConnect2'):
        gender_fc2_weights = tf.get_variable("weight", [FC_SIZE, cfg.GENDER_NUM],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('gender_losses', regularizer(gender_fc2_weights))
        gender_fc2_biases = tf.get_variable("bias", [cfg.GENDER_NUM], initializer=tf.constant_initializer(0.1))
        gender_Out = tf.matmul(gender_fc1, gender_fc2_weights) + gender_fc2_biases

    return upperColor_Out, upperSleeve_Out, lowerColor_Out, lowerType_Out, lowerLength_Out, \
        backpack_Out, handbag_Out, umbrella_Out, hat_Out, age_Out, gender_Out

from __future__ import print_function
import sys
import tensorflow as tf
import numpy as np
import os
from matplotlib import pyplot as plt
from tensorflow.python.framework import graph_util
from tensorflow.python import pywrap_tensorflow
import cv2,csv
import scipy.io
import scipy.misc
import time
import random
import math
from os import walk

import peopleModel as model
import peopleConfig as cfg
import TFRecord as TFR
import MinMax_Scaler as mms

"""# Input Model Size
model.INPUT_WIDTH = 112
model.INPUT_HEIGHT = 112"""

"""# Image Size 
cfg.IMG_WIDTH = 200
cfg.IMG_HEIGHT = 400
cfg.NUM_CHANNELS = 3"""

"""# People Attribute Class Number
cfg.UPPER_COLOR_NUM = 11
cfg.UPPER_SLEEVE_NUM = 2
cfg.LOWER_COLOR_NUM = 11
cfg.LOWER_TYPE_NUM = 2
cfg.LOWER_LENGTH_NUM = 2
cfg.BACKPACK_NUM = 2
cfg.HANDBAG_NUM = 2
cfg.UMBRELLA_NUM = 2
cfg.HAT_NUM = 2
cfg.AGE_NUM = 3
cfg.GENDER_NUM = 2"""

"""# Training And Fine-tuning Parameters
cfg.BATCH_SIZE = 30
cfg.LEARNING_RATE_BASE = 0.0001
cfg.FINETUNE_LEARNING_RATE_BASE = 0.0001
cfg.LEARNING_RATE_DECAY = 0.99
cfg.REGULARIZATION_RATE = 0.0001
cfg.TRAINING_STEPS = 60
cfg.FINETUNE_STEPS = 50
cfg.FINETUNE_Epoch = 30"""

TRANSFER = False

MODEL_SAVE_PATH = "./Model/"

TFDIR = "../TFData/"
TFDATA = "PETA_People/" #PETA

MMS_PATH = TFDIR + TFDATA + "Scaler_PETA_Morning" #PETA
TFRecord_Train_Path = TFDIR + TFDATA + "train_part"
TFRecord_Test_Path = TFDIR + TFDATA + "test_part"

CROSS_PART = 5

USE_KM = True
MIN_KM = 2

UPPERCOLOR_TRAIN_LOSS = []
UPPERSLEEVE_TRAIN_LOSS = []
LOWERCOLOR_TRAIN_LOSS = []
LOWERTYPE_TRAIN_LOSS = []
LOWERLENGTH_TRAIN_LOSS = []
BACKPACK_TRAIN_LOSS = []
HANDBAG_TRAIN_LOSS = []
UMBRELLA_TRAIN_LOSS = []
HAT_TRAIN_LOSS = []
AGE_TRAIN_LOSS = []
GENDER_TRAIN_LOSS = []

UPPERCOLOR_VAL_LOSS = []
UPPERSLEEVE_VAL_LOSS = []
LOWERCOLOR_VAL_LOSS = []
LOWERTYPE_VAL_LOSS = []
LOWERLENGTH_VAL_LOSS = []
BACKPACK_VAL_LOSS = []
HANDBAG_VAL_LOSS = []
UMBRELLA_VAL_LOSS = []
HAT_VAL_LOSS = []
AGE_VAL_LOSS = []
GENDER_VAL_LOSS = []

UPPERCOLOR_TRAIN_ACC = []
UPPERSLEEVE_TRAIN_ACC = []
LOWERCOLOR_TRAIN_ACC = []
LOWERTYPE_TRAIN_ACC = []
LOWERLENGTH_TRAIN_ACC = []
BACKPACK_TRAIN_ACC = []
HANDBAG_TRAIN_ACC = []
UMBRELLA_TRAIN_ACC = []
HAT_TRAIN_ACC = []
AGE_TRAIN_ACC = []
GENDER_TRAIN_ACC = []

UPPERCOLOR_VAL_ACC = []
UPPERSLEEVE_VAL_ACC = []
LOWERCOLOR_VAL_ACC = []
LOWERTYPE_VAL_ACC = []
LOWERLENGTH_VAL_ACC = []
BACKPACK_VAL_ACC = []
HANDBAG_VAL_ACC = []
UMBRELLA_VAL_ACC = []
HAT_VAL_ACC = []
AGE_VAL_ACC = []
GENDER_VAL_ACC = []

np.set_printoptions(threshold=np.nan)

"""def get_data_size(tfrecords_filename):
    record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
    counter = 0
    for string_record in record_iterator:
       counter += 1
    return counter"""

"""# Print Confusion Matrix
def cfg.print_confusion_matrix(confusion_matrix):
    # get confusion matrix
    # temp_sc -> sum of correct
    # temp_s -> sum
    # temp_dp -> denominator of precision
    # temp_dr -> denominator of recall
    # temp_p -> precision
    # temp_r -> recall

    class_num = confusion_matrix.shape[0]
    precision = np.zeros([class_num], dtype=float)
    recall = np.zeros([class_num], dtype=float)
    real_num = np.zeros([class_num], dtype=int)
    pred_num = np.zeros([class_num], dtype=int)
    accuracy = 0
    averagePrecision = 0
    averageRecall = 0
    temp_sc = 0
    temp_s = 0
    temp_ap = 0
    temp_ar = 0

    for cfm_g_r in range(0,class_num):
        temp_sc = temp_sc + confusion_matrix[cfm_g_r, cfm_g_r]
        temp_dp = 0
        temp_dr = 0
        for cfm_g_c in range(0,class_num):
            temp_dp = temp_dp + confusion_matrix[cfm_g_c, cfm_g_r]
            temp_dr = temp_dr + confusion_matrix[cfm_g_r, cfm_g_c]
        temp_s = temp_s + temp_dp
        temp_p = 0 if temp_dp == 0 else (confusion_matrix[cfm_g_r, cfm_g_r]/temp_dp)
        temp_r = 0 if temp_dr == 0 else (confusion_matrix[cfm_g_r, cfm_g_r]/temp_dr)
        precision[cfm_g_r] = temp_p
        recall[cfm_g_r] = temp_r
        real_num[cfm_g_r] = temp_dr
        pred_num[cfm_g_r] = temp_dp
        temp_ar += temp_r
        temp_ap += temp_p
    accuracy = temp_sc / temp_s
    averagePrecision = temp_ap / class_num
    averageRecall = temp_ar / class_num

    print("Confusion Matrix : ")
    print("Matrix\t:\t", end=" ")
    for ci in range(0,class_num):
        print("e%d\t" % (ci+1), end=" ")
    print("Recall")
    for ri in range(0,class_num):
        print("p%d\t:\t" % (ri+1), end=" ")
        for ci in range(0,class_num):
            print("%-6d\t" % (confusion_matrix[ri,ci]), end=" ")
        print("%.2f%%\t%-6d" % (recall[ri]*100, real_num[ri]))
    print("Preci\t:\t", end=" ")
    for ci in range(0,class_num):
        print("%.2f%%\t" % (precision[ci]*100), end=" ")
    print("\n\t\t", end=" ")
    for ci in range(0,class_num):
        print("%-6d\t" % (pred_num[ci]), end=" ")
    print("\nACC: %.2f%%\tAP: %.2f%%\tAR: %.2f%%" % (accuracy*100, averagePrecision*100, averageRecall*100))
    print("\n")"""

# Label Encode
def encode_labels( y, k):
    """Encode labels into one-hot representation"""
    onehot = np.zeros((y.shape[0],k ))
    for idx, val in enumerate(y):
        onehot[idx,val] = 1.0
    return onehot

"""def read_and_decode(filename_queue, readSize):
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
        cfg.BATCH_SIZE=readSize, capacity=6400, min_after_dequeue=800, num_threads=1, allow_smaller_final_batch=True)
    return images, upperColor, upperSleeve, lowerColor, lowerType, lowerLength, backpack, handbag, umbrella, hat, age, gender"""

# Define Loss Function
def loss_function(cross_entropy, expected_labels, class_num):
    # size -> input batch size
    # MK -> each class's num
    # NK -> 1/MK
    # NKS -> sum of NK
    # BK -> weight of each class's loss
    # BK_L -> BK Limited by BK_MAX
    # BK_W -> BK weight list for loss
    size = tf.shape(expected_labels)[0]
    MK = tf.bincount(tf.cast(expected_labels, tf.int32), minlength = class_num)
    MK_masked = tf.less(MK,MIN_KM)
    MK_L = tf.where(MK_masked, tf.fill([class_num],MIN_KM), MK)
    NK = tf.divide(1,tf.cast(MK_L, dtype=tf.float32))
    NKS = tf.reduce_sum(NK)
    BK = tf.divide(NK,NKS)

    BK_W = tf.zeros([size], dtype=tf.float32, name=None)
    count = 0

    def BK_set(count, expected_labels, class_num, BK, BK_W):
        size = tf.shape(expected_labels)[0]
        zeros = tf.zeros([size], dtype=tf.float32, name="zeros")
        masked = tf.equal(expected_labels, tf.cast(count,dtype=tf.int64))
        temp_BK = tf.where(masked,  tf.fill([size], BK[count]), zeros)
        BK_W = tf.add(BK_W, temp_BK)
        count = count + 1
        return count, expected_labels, class_num, BK, BK_W

    count, expected_labels, class_num, BK, BK_W = tf.while_loop((                         \
        lambda count, expected_labels, class_num, BK_L, BK_W: tf.less(count, class_num)), \
        BK_set, [count, expected_labels, class_num, BK, BK_W])

    weighted_cross_entropy = tf.multiply(cross_entropy, BK_W)
    if USE_KM:
        cost = tf.reduce_mean(weighted_cross_entropy)
    else:
        cost = tf.reduce_mean(cross_entropy)
    return cost
    
def evaluate(TFRecord_Test_Name, test_Number, partIndex, mode):
    tf.reset_default_graph()

    test_batch_len = int(math.ceil(float(test_Number)/cfg.BATCH_SIZE))

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        input_Image_ = tf.placeholder(tf.float32, [None, model.INPUT_WIDTH*model.INPUT_HEIGHT*cfg.NUM_CHANNELS], name='input_Image')
        input_Image = tf.reshape(input_Image_, shape=[-1, model.INPUT_WIDTH, model.INPUT_HEIGHT, cfg.NUM_CHANNELS])

        upperColor_Label  = tf.placeholder(tf.float32, [None, cfg.UPPER_COLOR_NUM],  name='upperColor_Label')
        upperSleeve_Label = tf.placeholder(tf.float32, [None, cfg.UPPER_SLEEVE_NUM], name='upperSleeve_Label')
        lowerColor_Label  = tf.placeholder(tf.float32, [None, cfg.LOWER_COLOR_NUM],  name='lowerColor_Label')
        lowerType_Label   = tf.placeholder(tf.float32, [None, cfg.LOWER_TYPE_NUM],   name='lowerType_Label')
        lowerLength_Label = tf.placeholder(tf.float32, [None, cfg.LOWER_LENGTH_NUM], name='lowerLength_Label')
        backpack_Label    = tf.placeholder(tf.float32, [None, cfg.BACKPACK_NUM],     name='backpack_Label')
        handbag_Label     = tf.placeholder(tf.float32, [None, cfg.HANDBAG_NUM],      name='handbag_Label')
        umbrella_Label    = tf.placeholder(tf.float32, [None, cfg.UMBRELLA_NUM],     name='umbrella_Label')
        hat_Label         = tf.placeholder(tf.float32, [None, cfg.HAT_NUM],          name='hat_Label')
        age_Label         = tf.placeholder(tf.float32, [None, cfg.AGE_NUM],          name='age_Label')
        gender_Label      = tf.placeholder(tf.float32, [None, cfg.GENDER_NUM],       name='gender_Label')

        regularizer = tf.contrib.layers.l2_regularizer(cfg.REGULARIZATION_RATE)

        # From Model Get Predict Answer
        upperColor_Output, upperSleeve_Output, lowerColor_Output, lowerType_Output, lowerLength_Output, backpack_Output, \
            handbag_Output, umbrella_Output, hat_Output, age_Output, gender_Output = model.interface(input_Image, False, regularizer)

        global_step = tf.Variable(0, trainable=False)

        # Predict
        upperColor_Pred = tf.argmax(upperColor_Output, 1)
        upperSleeve_Pred = tf.argmax(upperSleeve_Output, 1)
        lowerColor_Pred = tf.argmax(lowerColor_Output, 1)
        lowerType_Pred = tf.argmax(lowerType_Output, 1)
        lowerLength_Pred = tf.argmax(lowerLength_Output, 1)
        backpack_Pred = tf.argmax(backpack_Output, 1)
        handbag_Pred = tf.argmax(handbag_Output, 1)
        umbrella_Pred = tf.argmax(umbrella_Output, 1)
        hat_Pred = tf.argmax(hat_Output, 1)
        age_Pred = tf.argmax(age_Output, 1)
        gender_Pred = tf.argmax(gender_Output, 1)

        # Confidence
        upperColor_Softmax = tf.nn.softmax(upperColor_Output)
        upperSleeve_Softmax = tf.nn.softmax(upperSleeve_Output)
        lowerColor_Softmax = tf.nn.softmax(lowerColor_Output)
        lowerType_Softmax = tf.nn.softmax(lowerType_Output)
        lowerLength_Softmax = tf.nn.softmax(lowerLength_Output)
        backpack_Softmax = tf.nn.softmax(backpack_Output)
        handbag_Softmax = tf.nn.softmax(handbag_Output)
        umbrella_Softmax = tf.nn.softmax(umbrella_Output)
        hat_Softmax = tf.nn.softmax(hat_Output)
        age_Softmax = tf.nn.softmax(age_Output)
        gender_Softmax = tf.nn.softmax(gender_Output)

        saver = tf.train.Saver(var_list=tf.global_variables())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        filename_queue = tf.train.string_input_producer([TFRecord_Test_Name], shuffle = False, num_epochs=1)
        # Read TFRecords Data
        images, upperColorTest, upperSleeveTest, lowerColorTest, lowerTypeTest, lowerLengthTest, backpackTest, \
            handbagTest, umbrellaTest, hatTest, ageTest, genderTest = TFR.read_and_decode(filename_queue, cfg.BATCH_SIZE)

        init_local = tf.local_variables_initializer()

        # Confusion Matrix
        upperColor_ConfusionMatrix = np.zeros([cfg.UPPER_COLOR_NUM,cfg.UPPER_COLOR_NUM], dtype=float)
        upperSleeve_ConfusionMatrix = np.zeros([cfg.UPPER_SLEEVE_NUM,cfg.UPPER_SLEEVE_NUM], dtype=float)
        lowerColor_ConfusionMatrix = np.zeros([cfg.LOWER_COLOR_NUM,cfg.LOWER_COLOR_NUM], dtype=float)
        lowerType_ConfusionMatrix = np.zeros([cfg.LOWER_TYPE_NUM,cfg.LOWER_TYPE_NUM], dtype=float)
        lowerLength_ConfusionMatrix = np.zeros([cfg.LOWER_LENGTH_NUM,cfg.LOWER_LENGTH_NUM], dtype=float)
        backpack_ConfusionMatrix = np.zeros([cfg.BACKPACK_NUM,cfg.BACKPACK_NUM], dtype=float)
        handbag_ConfusionMatrix = np.zeros([cfg.HANDBAG_NUM,cfg.HANDBAG_NUM], dtype=float)
        umbrella_ConfusionMatrix = np.zeros([cfg.UMBRELLA_NUM,cfg.UMBRELLA_NUM], dtype=float)
        hat_ConfusionMatrix = np.zeros([cfg.HAT_NUM,cfg.HAT_NUM], dtype=float)
        age_ConfusionMatrix = np.zeros([cfg.AGE_NUM,cfg.AGE_NUM], dtype=float)
        gender_ConfusionMatrix = np.zeros([cfg.GENDER_NUM,cfg.GENDER_NUM], dtype=float)

        if mode == 1:
            finetune_name = ""
        elif mode == 2:
            finetune_name = "_Finetune"

        upperColor_fp = open(MODEL_SAVE_PATH+MODEL_NAME+"/Txt/UpperColor_Out"+finetune_name+".txt", "w")
        upperSleeve_fp = open(MODEL_SAVE_PATH+MODEL_NAME+"/Txt/UpperSleeve_Out"+finetune_name+".txt", "w")
        lowerColor_fp = open(MODEL_SAVE_PATH+MODEL_NAME+"/Txt/LowerColor_Out"+finetune_name+".txt", "w")
        lowerType_fp = open(MODEL_SAVE_PATH+MODEL_NAME+"/Txt/LowerType_Out"+finetune_name+".txt", "w")
        lowerLength_fp = open(MODEL_SAVE_PATH+MODEL_NAME+"/Txt/LowerLength_Out"+finetune_name+".txt", "w")
        backpack_fp = open(MODEL_SAVE_PATH+MODEL_NAME+"/Txt/Backpack_Out"+finetune_name+".txt", "w")
        handbag_fp = open(MODEL_SAVE_PATH+MODEL_NAME+"/Txt/Handbag_Out"+finetune_name+".txt", "w")
        umbrella_fp = open(MODEL_SAVE_PATH+MODEL_NAME+"/Txt/Umbrella_Out"+finetune_name+".txt", "w")
        hat_fp = open(MODEL_SAVE_PATH+MODEL_NAME+"/Txt/Hat_Out"+finetune_name+".txt", "w")
        age_fp = open(MODEL_SAVE_PATH+MODEL_NAME+"/Txt/Age_Out"+finetune_name+".txt", "w")
        gender_fp = open(MODEL_SAVE_PATH+MODEL_NAME+"/Txt/Gender_Out"+finetune_name+".txt", "w")


        print ("Restore  Start!")
        if mode == 1:
            saver.restore(sess, os.path.join(MODEL_SAVE_PATH+MODEL_NAME+"/", (MODEL_NAME + "_part" + str(partIndex) + "_final")))
        elif mode == 2:
            saver.restore(sess, os.path.join(MODEL_SAVE_PATH+MODEL_NAME+"/Finetune/", (MODEL_NAME + "_part" + str(partIndex) + "_final_finetune")))
        sess.run(init_local)
        print ("Restore  Finished!")

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        print ("Start  Testing!")
        for step in range(test_batch_len):
            Image_Batch, upperColor_Batch, upperSleeve_Batch, lowerColor_Batch, lowerType_Batch, lowerLength_Batch, \
                backpack_Batch, handbag_Batch, umbrella_Batch, hat_Batch, age_Batch, gender_Batch = sess.run([images,\
                upperColorTest, upperSleeveTest, lowerColorTest, lowerTypeTest, lowerLengthTest,                    \
                backpackTest, handbagTest, umbrellaTest, hatTest, ageTest, genderTest])

            Image_Scaler = mms.mms_trans(Image_Batch, MMS_PATH)
            # Data Augumentation
            Image_Augumentation = cfg.DataAugmentation(Image_Scaler, False)
            Image_Reshaped = np.reshape(Image_Augumentation, (Image_Augumentation.shape[0], model.INPUT_WIDTH, model.INPUT_HEIGHT, cfg.NUM_CHANNELS))

            # Encode Label
            upperColor_Encode = encode_labels(upperColor_Batch, cfg.UPPER_COLOR_NUM)
            upperSleeve_Encode = encode_labels(upperSleeve_Batch, cfg.UPPER_SLEEVE_NUM)
            lowerColor_Encode = encode_labels(lowerColor_Batch, cfg.LOWER_COLOR_NUM)
            lowerType_Encode = encode_labels(lowerType_Batch, cfg.LOWER_TYPE_NUM)
            lowerLength_Encode = encode_labels(lowerLength_Batch, cfg.LOWER_LENGTH_NUM)
            backpack_Encode = encode_labels(backpack_Batch, cfg.BACKPACK_NUM)
            handbag_Encode = encode_labels(handbag_Batch, cfg.HANDBAG_NUM)
            umbrella_Encode = encode_labels(umbrella_Batch, cfg.UMBRELLA_NUM)
            hat_Encode = encode_labels(hat_Batch, cfg.HAT_NUM)
            age_Encode = encode_labels(age_Batch, cfg.AGE_NUM)
            gender_Encode = encode_labels(gender_Batch, cfg.GENDER_NUM)


            # Fit training using batch data
            upperColor_Predict, upperSleeve_Predict, lowerColor_Predict, lowerType_Predict, lowerLength_Predict, \
                backpack_Predict, handbag_Predict, umbrella_Predict, hat_Predict, age_Predict, gender_Predict,   \
                upperColor_Confidence, upperSleeve_Confidence, lowerColor_Confidence, lowerType_Confidence,      \
                lowerLength_Confidence, backpack_Confidence, handbag_Confidence, umbrella_Confidence, \
                hat_Confidence, age_Confidence, gender_Confidence = sess.run([upperColor_Pred, upperSleeve_Pred, \
                lowerColor_Pred, lowerType_Pred, lowerLength_Pred, backpack_Pred, handbag_Pred, umbrella_Pred,   \
                hat_Pred, age_Pred, gender_Pred,                                                                 \
                upperColor_Softmax, upperSleeve_Softmax, lowerColor_Softmax, lowerType_Softmax, lowerLength_Softmax,\
                backpack_Softmax, handbag_Softmax, umbrella_Softmax, hat_Softmax, age_Softmax, gender_Softmax],  \
                feed_dict={input_Image: Image_Reshaped,                                                          \
                upperColor_Label: upperColor_Encode, upperSleeve_Label: upperSleeve_Encode,                      \
                lowerColor_Label: lowerColor_Encode, lowerType_Label: lowerType_Encode,                          \
                lowerLength_Label: lowerLength_Encode, backpack_Label: backpack_Encode,                          \
                handbag_Label: handbag_Encode, umbrella_Label: umbrella_Encode, hat_Label: hat_Encode,           \
                age_Label: age_Encode, gender_Label: gender_Encode})

            # Set Confusion Matrix And Set Output Txt
            line = []
            for ci in range(len(upperColor_Predict)):
                upperColor_ConfusionMatrix[upperColor_Batch[ci], upperColor_Predict[ci]] +=1
                line = [str(upperColor_Batch[ci])+"\t"+str(upperColor_Predict[ci])+"\t"+str(upperColor_Confidence[ci])+"\n"]
                upperColor_fp.writelines(line)
            for ci in range(len(upperSleeve_Predict)):
                upperSleeve_ConfusionMatrix[upperSleeve_Batch[ci], upperSleeve_Predict[ci]] +=1
                line = [str(upperSleeve_Batch[ci])+"\t"+str(upperSleeve_Predict[ci])+"\t"+str(upperSleeve_Confidence[ci])+"\n"]
                upperSleeve_fp.writelines(line)
            for ci in range(len(lowerColor_Predict)):
                lowerColor_ConfusionMatrix[lowerColor_Batch[ci], lowerColor_Predict[ci]] +=1
                line = [str(lowerColor_Batch[ci])+"\t"+str(lowerColor_Predict[ci])+"\t"+str(lowerColor_Confidence[ci])+"\n"]
                lowerColor_fp.writelines(line)
            for ci in range(len(lowerType_Predict)):
                lowerType_ConfusionMatrix[lowerType_Batch[ci], lowerType_Predict[ci]] +=1
                line = [str(lowerType_Batch[ci])+"\t"+str(lowerType_Predict[ci])+"\t"+str(lowerType_Confidence[ci])+"\n"]
                lowerType_fp.writelines(line)
            for ci in range(len(lowerLength_Predict)):
                lowerLength_ConfusionMatrix[lowerLength_Batch[ci], lowerLength_Predict[ci]] +=1
                line = [str(lowerLength_Batch[ci])+"\t"+str(lowerLength_Predict[ci])+"\t"+str(lowerLength_Confidence[ci])+"\n"]
                lowerLength_fp.writelines(line)
            for ci in range(len(backpack_Predict)):
                backpack_ConfusionMatrix[backpack_Batch[ci], backpack_Predict[ci]] += 1
                line = [str(backpack_Batch[ci])+"\t"+str(backpack_Predict[ci])+"\t"+str(backpack_Confidence[ci])+"\n"]
                backpack_fp.writelines(line)
            for ci in range(len(handbag_Predict)):
                handbag_ConfusionMatrix[handbag_Batch[ci], handbag_Predict[ci]] += 1
                line = [str(handbag_Batch[ci])+"\t"+str(handbag_Predict[ci])+"\t"+str(handbag_Confidence[ci])+"\n"]
                handbag_fp.writelines(line)
            for ci in range(len(umbrella_Predict)):
                umbrella_ConfusionMatrix[umbrella_Batch[ci], umbrella_Predict[ci]] += 1
                line = [str(umbrella_Batch[ci])+"\t"+str(umbrella_Predict[ci])+"\t"+str(umbrella_Confidence[ci])+"\n"]
                umbrella_fp.writelines(line)
            for ci in range(len(hat_Predict)):
                hat_ConfusionMatrix[hat_Batch[ci], hat_Predict[ci]] += 1
                line = [str(hat_Batch[ci])+"\t"+str(hat_Predict[ci])+"\t"+str(hat_Confidence[ci])+"\n"]
                hat_fp.writelines(line)
            for ci in range(len(age_Predict)):
                age_ConfusionMatrix[age_Batch[ci], age_Predict[ci]] += 1
                line = [str(age_Batch[ci])+"\t"+str(age_Predict[ci])+"\t"+str(age_Confidence[ci])+"\n"]
                age_fp.writelines(line)
            for ci in range(len(gender_Predict)):
                gender_ConfusionMatrix[gender_Batch[ci], gender_Predict[ci]] += 1
                line = [str(gender_Batch[ci])+"\t"+str(gender_Predict[ci])+"\t"+str(gender_Confidence[ci])+"\n"]
                gender_fp.writelines(line)

            step += 1
        upperColor_fp.close()
        upperSleeve_fp.close()
        lowerColor_fp.close()
        lowerType_fp.close()
        lowerLength_fp.close()
        backpack_fp.close()
        handbag_fp.close()
        umbrella_fp.close()
        hat_fp.close()
        age_fp.close()
        gender_fp.close()

        coord.request_stop()
        coord.join(threads)
    return upperColor_ConfusionMatrix, upperSleeve_ConfusionMatrix, lowerColor_ConfusionMatrix, lowerType_ConfusionMatrix, \
        lowerLength_ConfusionMatrix, backpack_ConfusionMatrix, handbag_ConfusionMatrix, umbrella_ConfusionMatrix,          \
        hat_ConfusionMatrix, age_ConfusionMatrix, gender_ConfusionMatrix

def train(TFRecord_Train_Name, train_Number, partIndex, TFRecord_Validation_Name, validation_Number):
    tf.reset_default_graph()
    shuffle = True

    BATCH_NUM = int(math.ceil(float(train_Number/cfg.BATCH_SIZE)))
    VAL_BATCH_NUM = int(validation_Number/cfg.BATCH_SIZE)
    isTrain = True
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        input_Image_ = tf.placeholder(tf.float32, [None, model.INPUT_WIDTH*model.INPUT_HEIGHT*cfg.NUM_CHANNELS], name='input_Image')
        input_Image = tf.reshape(input_Image_, shape=[-1, model.INPUT_WIDTH, model.INPUT_HEIGHT, cfg.NUM_CHANNELS])

        upperColor_Label  = tf.placeholder(tf.float32, [None, cfg.UPPER_COLOR_NUM],  name='upperColor_Label')
        upperSleeve_Label = tf.placeholder(tf.float32, [None, cfg.UPPER_SLEEVE_NUM], name='upperSleeve_Label')
        lowerColor_Label  = tf.placeholder(tf.float32, [None, cfg.LOWER_COLOR_NUM],  name='lowerColor_Label')
        lowerType_Label   = tf.placeholder(tf.float32, [None, cfg.LOWER_TYPE_NUM],   name='lowerType_Label')
        lowerLength_Label = tf.placeholder(tf.float32, [None, cfg.LOWER_LENGTH_NUM], name='lowerLength_Label')
        backpack_Label    = tf.placeholder(tf.float32, [None, cfg.BACKPACK_NUM],     name='backpack_Label')
        handbag_Label     = tf.placeholder(tf.float32, [None, cfg.HANDBAG_NUM],      name='handbag_Label')
        umbrella_Label    = tf.placeholder(tf.float32, [None, cfg.UMBRELLA_NUM],     name='umbrella_Label')
        hat_Label         = tf.placeholder(tf.float32, [None, cfg.HAT_NUM],          name='hat_Label')
        age_Label         = tf.placeholder(tf.float32, [None, cfg.AGE_NUM],          name='age_Label')
        gender_Label      = tf.placeholder(tf.float32, [None, cfg.GENDER_NUM],       name='gender_Label')

        regularizer = tf.contrib.layers.l2_regularizer(cfg.REGULARIZATION_RATE)

        # From Model Get Predict Answer
        upperColor_Output, upperSleeve_Output, lowerColor_Output, lowerType_Output, lowerLength_Output, backpack_Output, \
            handbag_Output, umbrella_Output, hat_Output, age_Output, gender_Output = model.interface(input_Image, isTrain, regularizer)

        global_step = tf.Variable(0, trainable=False)

        # Predict
        upperColor_Pred = tf.argmax(upperColor_Output, 1)
        upperSleeve_Pred = tf.argmax(upperSleeve_Output, 1)
        lowerColor_Pred = tf.argmax(lowerColor_Output, 1)
        lowerType_Pred = tf.argmax(lowerType_Output, 1)
        lowerLength_Pred = tf.argmax(lowerLength_Output, 1)
        backpack_Pred = tf.argmax(backpack_Output, 1)
        handbag_Pred = tf.argmax(handbag_Output, 1)
        umbrella_Pred = tf.argmax(umbrella_Output, 1)
        hat_Pred = tf.argmax(hat_Output, 1)
        age_Pred = tf.argmax(age_Output, 1)
        gender_Pred = tf.argmax(gender_Output, 1)

        # Expect
        upperColor_Real = tf.argmax(upperColor_Label, 1)
        upperSleeve_Real = tf.argmax(upperSleeve_Label, 1)
        lowerColor_Real = tf.argmax(lowerColor_Label, 1)
        lowerType_Real = tf.argmax(lowerType_Label, 1)
        lowerLength_Real = tf.argmax(lowerLength_Label, 1)
        backpack_Real = tf.argmax(backpack_Label, 1)
        handbag_Real = tf.argmax(handbag_Label, 1)
        umbrella_Real = tf.argmax(umbrella_Label, 1)
        hat_Real = tf.argmax(hat_Label, 1)
        age_Real = tf.argmax(age_Label, 1)
        gender_Real = tf.argmax(gender_Label, 1)

        # Correct
        upperColor_Correct = tf.equal(upperColor_Pred, upperColor_Real)
        upperSleeve_Correct = tf.equal(upperSleeve_Pred, upperSleeve_Real)
        lowerColor_Correct = tf.equal(lowerColor_Pred, lowerColor_Real)
        lowerType_Correct = tf.equal(lowerType_Pred, lowerType_Real)
        lowerLength_Correct = tf.equal(lowerLength_Pred, lowerLength_Real)
        backpack_Correct = tf.equal(backpack_Pred, backpack_Real)
        handbag_Correct = tf.equal(handbag_Pred, handbag_Real)
        umbrella_Correct = tf.equal(umbrella_Pred, umbrella_Real)
        hat_Correct = tf.equal(hat_Pred, hat_Real)
        age_Correct = tf.equal(age_Pred, age_Real)
        gender_Correct = tf.equal(gender_Pred, gender_Real)

        # Accuracy
        upperColor_Accuracy = tf.reduce_mean(tf.cast(upperColor_Correct, tf.float32))
        upperSleeve_Accuracy = tf.reduce_mean(tf.cast(upperSleeve_Correct, tf.float32))
        lowerColor_Accuracy = tf.reduce_mean(tf.cast(lowerColor_Correct, tf.float32))
        lowerType_Accuracy = tf.reduce_mean(tf.cast(lowerType_Correct, tf.float32))
        lowerLength_Accuracy = tf.reduce_mean(tf.cast(lowerLength_Correct, tf.float32))
        backpack_Accuracy = tf.reduce_mean(tf.cast(backpack_Correct, tf.float32))
        handbag_Accuracy = tf.reduce_mean(tf.cast(handbag_Correct, tf.float32))
        umbrella_Accuracy = tf.reduce_mean(tf.cast(umbrella_Correct, tf.float32))
        hat_Accuracy = tf.reduce_mean(tf.cast(hat_Correct, tf.float32))
        age_Accuracy = tf.reduce_mean(tf.cast(age_Correct, tf.float32))
        gender_Accuracy = tf.reduce_mean(tf.cast(gender_Correct, tf.float32))

        # Cross Entropy
        upperColor_CrossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits( \
            logits=upperColor_Output, labels=tf.argmax(upperColor_Label, 1))
        upperSleeve_CrossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits(\
            logits=upperSleeve_Output, labels=tf.argmax(upperSleeve_Label, 1))
        lowerColor_CrossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits( \
            logits=lowerColor_Output, labels=tf.argmax(lowerColor_Label, 1))
        lowerType_CrossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits(  \
            logits=lowerType_Output, labels=tf.argmax(lowerType_Label, 1))
        lowerLength_CrossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits(\
            logits=lowerLength_Output, labels=tf.argmax(lowerLength_Label, 1))
        backpack_CrossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits(   \
            logits=backpack_Output, labels=tf.argmax(backpack_Label, 1))
        handbag_CrossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits(    \
            logits=handbag_Output, labels=tf.argmax(handbag_Label, 1))
        umbrella_CrossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits(   \
            logits=umbrella_Output, labels=tf.argmax(umbrella_Label, 1))
        hat_CrossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits(        \
            logits=hat_Output, labels=tf.argmax(hat_Label, 1))
        age_CrossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits(        \
            logits=age_Output, labels=tf.argmax(age_Label, 1))
        gender_CrossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits(     \
            logits=gender_Output, labels=tf.argmax(gender_Label, 1))

        # Loss
        upperColor_CrossEntropy_Loss = loss_function(upperColor_CrossEntropy, upperColor_Real, cfg.UPPER_COLOR_NUM)
        upperSleeve_CrossEntropy_Loss = loss_function(upperSleeve_CrossEntropy, upperSleeve_Real, cfg.UPPER_SLEEVE_NUM)
        lowerColor_CrossEntropy_Loss = loss_function(lowerColor_CrossEntropy, lowerColor_Real, cfg.LOWER_COLOR_NUM)
        lowerType_CrossEntropy_Loss = loss_function(lowerType_CrossEntropy, lowerType_Real, cfg.LOWER_TYPE_NUM)
        lowerLength_CrossEntropy_Loss = loss_function(lowerLength_CrossEntropy, lowerLength_Real, cfg.LOWER_LENGTH_NUM)
        backpack_CrossEntropy_Loss = loss_function(backpack_CrossEntropy, backpack_Real, cfg.BACKPACK_NUM)
        handbag_CrossEntropy_Loss = loss_function(handbag_CrossEntropy, handbag_Real, cfg.HANDBAG_NUM)
        umbrella_CrossEntropy_Loss = loss_function(umbrella_CrossEntropy, umbrella_Real, cfg.UMBRELLA_NUM)
        hat_CrossEntropy_Loss = loss_function(hat_CrossEntropy, hat_Real, cfg.HAT_NUM)
        age_CrossEntropy_Loss = loss_function(age_CrossEntropy, age_Real, cfg.AGE_NUM)
        gender_CrossEntropy_Loss = loss_function(gender_CrossEntropy, gender_Real, cfg.GENDER_NUM)

        loss = (upperColor_CrossEntropy_Loss + upperSleeve_CrossEntropy_Loss + lowerColor_CrossEntropy_Loss +\
            lowerType_CrossEntropy_Loss + lowerLength_CrossEntropy_Loss + backpack_CrossEntropy_Loss +       \
            handbag_CrossEntropy_Loss + umbrella_CrossEntropy_Loss + hat_CrossEntropy_Loss +                 \
            age_CrossEntropy_Loss + gender_CrossEntropy_Loss) / 11

        learning_rate = tf.train.exponential_decay(cfg.LEARNING_RATE_BASE, global_step, BATCH_NUM*cfg.BATCH_SIZE, cfg.LEARNING_RATE_DECAY, staircase=True)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step = global_step)

        # Construct Coordinate
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        filename_queue_train = tf.train.string_input_producer([TFRecord_Train_Name], shuffle = True, num_epochs=cfg.TRAINING_STEPS+cfg.TRAINING_STEPS/5)
        # Read TFRecords Data
        imagesTrain, upperColorTrain, upperSleeveTrain, lowerColorTrain, lowerTypeTrain, lowerLengthTrain,  \
            backpackTrain, handbagTrain, umbrellaTrain, hatTrain, ageTrain, genderTrain =                   \
            TFR.read_and_decode(filename_queue_train, cfg.BATCH_SIZE)

        filename_queue_val = tf.train.string_input_producer([TFRecord_Validation_Name], shuffle = True, num_epochs=cfg.TRAINING_STEPS/5)
        imagesVal, upperColorVal, upperSleeveVal, lowerColorVal, lowerTypeVal, lowerLengthVal,              \
            backpackVal, handbagVal, umbrellaVal, hatVal, ageVal, genderVal = TFR.read_and_decode(filename_queue_val, cfg.BATCH_SIZE)

        saver = tf.train.Saver(var_list=tf.global_variables())
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        init_local = tf.local_variables_initializer()

        if TRANSFER:
            # Load Old Model
            print ("Restore  Start!")
            saver.restore(sess, os.path.join(MODEL_SAVE_PATH+MODEL_NAME+"/", (MODEL_NAME + "_part" + str(partIndex) + "_final")))
            sess.run(init_local)
            print ("Restore  Finished!")
        else :
            # Initial New Model
            print ("Initialize  Start!")
            sess.run(init)
            print ("Initialize  Finished!")

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        print(BATCH_NUM)
        print("Start training!")
        for epoch in range(cfg.TRAINING_STEPS):
            isTrain = True
            for step in range(BATCH_NUM):
                Image_Batch, upperColor_Batch, upperSleeve_Batch, lowerColor_Batch, lowerType_Batch, lowerLength_Batch, \
                    backpack_Batch, handbag_Batch, umbrella_Batch, hat_Batch, age_Batch, gender_Batch = sess.run([      \
                    imagesTrain, upperColorTrain, upperSleeveTrain, lowerColorTrain, lowerTypeTrain, lowerLengthTrain,  \
                    backpackTrain, handbagTrain, umbrellaTrain, hatTrain, ageTrain, genderTrain])
                
                # MinMax_Scaler
                Image_Scaler = mms.mms_trans(Image_Batch, MMS_PATH)

                # Data Augumentation
                Image_Augumentation = cfg.DataAugmentation(Image_Scaler, True)
                Image_Reshaped = np.reshape(Image_Augumentation, (Image_Augumentation.shape[0], model.INPUT_WIDTH, model.INPUT_HEIGHT, cfg.NUM_CHANNELS))

                # Encode Label
                upperColor_Encode = encode_labels(upperColor_Batch, cfg.UPPER_COLOR_NUM)
                upperSleeve_Encode = encode_labels(upperSleeve_Batch, cfg.UPPER_SLEEVE_NUM)
                lowerColor_Encode = encode_labels(lowerColor_Batch, cfg.LOWER_COLOR_NUM)
                lowerType_Encode = encode_labels(lowerType_Batch, cfg.LOWER_TYPE_NUM)
                lowerLength_Encode = encode_labels(lowerLength_Batch, cfg.LOWER_LENGTH_NUM)
                backpack_Encode = encode_labels(backpack_Batch, cfg.BACKPACK_NUM)
                handbag_Encode = encode_labels(handbag_Batch, cfg.HANDBAG_NUM)
                umbrella_Encode = encode_labels(umbrella_Batch, cfg.UMBRELLA_NUM)
                hat_Encode = encode_labels(hat_Batch, cfg.HAT_NUM)
                age_Encode = encode_labels(age_Batch, cfg.AGE_NUM)
                gender_Encode = encode_labels(gender_Batch, cfg.GENDER_NUM)

                # Fit training using batch data
                upperColor_Predict, upperSleeve_Predict, lowerColor_Predict, lowerType_Predict, lowerLength_Predict,     \
                    backpack_Predict, handbag_Predict, umbrella_Predict, hat_Predict, age_Predict, gender_Predict,       \
                    loss_value, upperColor_Cost, upperSleeve_Cost, lowerColor_Cost, lowerType_Cost, lowerLength_Cost,    \
                    backpack_Cost, handbag_Cost, umbrella_Cost, hat_Cost, age_Cost, gender_Cost,                         \
                    upperColor_Acc, upperSleeve_Acc, lowerColor_Acc, lowerType_Acc, lowerLength_Acc,                     \
                    backpack_Acc, handbag_Acc, umbrella_Acc, hat_Acc, age_Acc, gender_Acc, opt = sess.run([              \
                    upperColor_Pred, upperSleeve_Pred, lowerColor_Pred, lowerType_Pred, lowerLength_Pred,                \
                    backpack_Pred, handbag_Pred, umbrella_Pred, hat_Pred, age_Pred, gender_Pred, loss,                   \
                    upperColor_CrossEntropy_Loss, upperSleeve_CrossEntropy_Loss, lowerColor_CrossEntropy_Loss,           \
                    lowerType_CrossEntropy_Loss, lowerLength_CrossEntropy_Loss, backpack_CrossEntropy_Loss,              \
                    handbag_CrossEntropy_Loss, umbrella_CrossEntropy_Loss, hat_CrossEntropy_Loss, age_CrossEntropy_Loss, \
                    gender_CrossEntropy_Loss,                                                                            \
                    upperColor_Accuracy, upperSleeve_Accuracy, lowerColor_Accuracy, lowerType_Accuracy,                  \
                    lowerLength_Accuracy, backpack_Accuracy, handbag_Accuracy, umbrella_Accuracy, hat_Accuracy,          \
                    age_Accuracy, gender_Accuracy, optimizer], feed_dict={input_Image: Image_Reshaped,                   \
                    upperColor_Label : upperColor_Encode , upperSleeve_Label: upperSleeve_Encode, \
                    lowerColor_Label : lowerColor_Encode , lowerType_Label  : lowerType_Encode  , \
                    lowerLength_Label: lowerLength_Encode, backpack_Label   : backpack_Encode   , \
                    handbag_Label    : handbag_Encode    , umbrella_Label   : umbrella_Encode   , \
                    hat_Label        : hat_Encode, age_Label: age_Encode, gender_Label: gender_Encode})

                if (step+1) % BATCH_NUM == 0:
                    # Show Loss, Accuracy, Predict, Real
                    # Save Model
                    print("Training Set, Epoch %d, loss %g" % (epoch+1, loss_value))
                    print("UpperColor Loss %g, Accuracy %g" % (upperColor_Cost, upperColor_Acc))
                    print("UpperColor Real\t", upperColor_Batch)
                    print("UpperColor Pred\t", upperColor_Predict)
                    print("UpperSleeve Loss %g, Accuracy %g" % (upperSleeve_Cost, upperSleeve_Acc))
                    print("UpperSleeve Real\t", upperSleeve_Batch)
                    print("UpperSleeve Pred\t", upperSleeve_Predict)
                    print("LowerColor Loss %g, Accuracy %g" % (lowerColor_Cost, lowerColor_Acc))
                    print("LowerColor Real\t", lowerColor_Batch)
                    print("LowerColor Pred\t", lowerColor_Predict)
                    print("LowerType Loss %g, Accuracy %g" % (lowerType_Cost, lowerType_Acc))
                    print("LowerType Real\t", lowerType_Batch)
                    print("LowerType Pred\t", lowerType_Predict)
                    print("LowerLength Loss %g, Accuracy %g" % (lowerLength_Cost, lowerLength_Acc))
                    print("LowerLength Real\t", lowerLength_Batch)
                    print("LowerLength Pred\t", lowerLength_Predict)
                    print("Backpack Loss %g, Accuracy %g" % (backpack_Cost, backpack_Acc))
                    print("Backpack Real ", backpack_Batch)
                    print("Backpack Pred ", backpack_Predict)
                    print("Handbag Loss %g, Accuracy %g" % (handbag_Cost, handbag_Acc))
                    print("Handbag Real ", handbag_Batch)
                    print("Handbag Pred ", handbag_Predict)
                    print("Umbrella Loss %g, Accuracy %g" % (umbrella_Cost, umbrella_Acc))
                    print("Umbrella Real ", umbrella_Batch)
                    print("Umbrella Pred ", umbrella_Predict)
                    print("Hat Loss %g, Accuracy %g" % (hat_Cost, hat_Acc))
                    print("Hat Real ", hat_Batch)
                    print("Hat Pred ", hat_Predict)
                    print("Age Loss %g, Accuracy %g" % (age_Cost, age_Acc))
                    print("Age Real ", age_Batch)
                    print("Age Pred ", age_Predict)
                    print("Gender Loss %g, Accuracy %g" % (gender_Cost, gender_Acc))
                    print("Gender Real ", gender_Batch)
                    print("Gender Pred ", gender_Predict)
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH+MODEL_NAME+"/", (MODEL_NAME + "_part" + str(partIndex) + "_final")))

            if (epoch+1) % 5 == 0:
                isTrain = False
                # Training Set
                uppercolorCost = 0
                uppersleeveCost = 0
                lowercolorCost = 0
                lowertypeCost = 0
                lowerlengthCost = 0
                backpackCost = 0
                handbagCost = 0
                umbrellaCost = 0
                hatCost = 0
                ageCost = 0
                genderCost = 0

                uppercolorAcc = 0
                uppersleeveAcc = 0
                lowercolorAcc = 0
                lowertypeAcc = 0
                lowerlengthAcc = 0
                backpackAcc = 0
                handbagAcc = 0
                umbrellaAcc = 0
                hatAcc = 0
                ageAcc = 0
                genderAcc = 0

                for step in range(BATCH_NUM):
                    Image_Batch, upperColor_Batch, upperSleeve_Batch, lowerColor_Batch, lowerType_Batch, lowerLength_Batch, \
                        backpack_Batch, handbag_Batch, umbrella_Batch, hat_Batch, age_Batch, gender_Batch = sess.run([      \
                        imagesTrain, upperColorTrain, upperSleeveTrain, lowerColorTrain, lowerTypeTrain, lowerLengthTrain,  \
                        backpackTrain, handbagTrain, umbrellaTrain, hatTrain, ageTrain, genderTrain])
                    Image_Scaler = mms.mms_trans(Image_Batch, MMS_PATH)

                    # Data Augumentation
                    Image_Augumentation = cfg.DataAugmentation(Image_Scaler, False)
                    Image_Reshaped = np.reshape(Image_Augumentation, (cfg.BATCH_SIZE, model.INPUT_WIDTH, model.INPUT_HEIGHT, cfg.NUM_CHANNELS))

                    # Encode Label
                    upperColor_Encode = encode_labels(upperColor_Batch, cfg.UPPER_COLOR_NUM)
                    upperSleeve_Encode = encode_labels(upperSleeve_Batch, cfg.UPPER_SLEEVE_NUM)
                    lowerColor_Encode = encode_labels(lowerColor_Batch, cfg.LOWER_COLOR_NUM)
                    lowerType_Encode = encode_labels(lowerType_Batch, cfg.LOWER_TYPE_NUM)
                    lowerLength_Encode = encode_labels(lowerLength_Batch, cfg.LOWER_LENGTH_NUM)
                    backpack_Encode = encode_labels(backpack_Batch, cfg.BACKPACK_NUM)
                    handbag_Encode = encode_labels(handbag_Batch, cfg.HANDBAG_NUM)
                    umbrella_Encode = encode_labels(umbrella_Batch, cfg.UMBRELLA_NUM)
                    hat_Encode = encode_labels(hat_Batch, cfg.HAT_NUM)
                    age_Encode = encode_labels(age_Batch, cfg.AGE_NUM)
                    gender_Encode = encode_labels(gender_Batch, cfg.GENDER_NUM)

                    # Fit training using batch data
                    loss_value, upperColor_Cost, upperSleeve_Cost, lowerColor_Cost, lowerType_Cost, lowerLength_Cost,        \
                        backpack_Cost, handbag_Cost, umbrella_Cost, hat_Cost, age_Cost, gender_Cost,                         \
                        upperColor_Acc  , upperSleeve_Acc    , lowerColor_Acc    , lowerType_Acc    , lowerLength_Acc    ,   \
                        backpack_Acc, handbag_Acc, umbrella_Acc, hat_Acc, age_Acc, gender_Acc = sess.run([                   \
                        loss, upperColor_CrossEntropy_Loss, upperSleeve_CrossEntropy_Loss, lowerColor_CrossEntropy_Loss,     \
                        lowerType_CrossEntropy_Loss, lowerLength_CrossEntropy_Loss, backpack_CrossEntropy_Loss,              \
                        handbag_CrossEntropy_Loss, umbrella_CrossEntropy_Loss, hat_CrossEntropy_Loss, age_CrossEntropy_Loss, \
                        gender_CrossEntropy_Loss,   \
                        upperColor_Accuracy , upperSleeve_Accuracy, lowerColor_Accuracy, lowerType_Accuracy,                 \
                        lowerLength_Accuracy, backpack_Accuracy   , handbag_Accuracy   , umbrella_Accuracy , hat_Accuracy,   \
                        age_Accuracy, gender_Accuracy], feed_dict={           \
                        input_Image: Image_Reshaped,                                                                         \
                        upperColor_Label : upperColor_Encode , upperSleeve_Label: upperSleeve_Encode,                        \
                        lowerColor_Label : lowerColor_Encode , lowerType_Label  : lowerType_Encode  ,                        \
                        lowerLength_Label: lowerLength_Encode, backpack_Label   : backpack_Encode   ,                        \
                        handbag_Label    : handbag_Encode    , umbrella_Label   : umbrella_Encode   ,                        \
                        hat_Label        : hat_Encode, age_Label: age_Encode, gender_Label: gender_Encode})

                    uppercolorCost += upperColor_Cost
                    uppersleeveCost += upperSleeve_Cost
                    lowercolorCost += lowerColor_Cost
                    lowertypeCost += lowerType_Cost
                    lowerlengthCost += lowerLength_Cost
                    backpackCost += backpack_Cost
                    handbagCost += handbag_Cost
                    umbrellaCost += umbrella_Cost
                    hatCost += hat_Cost
                    ageCost += age_Cost
                    genderCost += gender_Cost

                    uppercolorAcc += upperColor_Acc
                    uppersleeveAcc += upperSleeve_Acc
                    lowercolorAcc += lowerColor_Acc
                    lowertypeAcc += lowerType_Acc
                    lowerlengthAcc += lowerLength_Acc
                    backpackAcc += backpack_Acc
                    handbagAcc += handbag_Acc
                    umbrellaAcc += umbrella_Acc
                    hatAcc += hat_Acc
                    ageAcc += age_Acc
                    genderAcc += gender_Acc

                UPPERCOLOR_TRAIN_LOSS.append(uppercolorCost/BATCH_NUM)
                UPPERSLEEVE_TRAIN_LOSS.append(uppersleeveCost/BATCH_NUM)
                LOWERCOLOR_TRAIN_LOSS.append(lowercolorCost/BATCH_NUM)
                LOWERTYPE_TRAIN_LOSS.append(lowertypeCost/BATCH_NUM)
                LOWERLENGTH_TRAIN_LOSS.append(lowerlengthCost/BATCH_NUM)
                BACKPACK_TRAIN_LOSS.append(backpackCost/BATCH_NUM)
                HANDBAG_TRAIN_LOSS.append(handbagCost/BATCH_NUM)
                UMBRELLA_TRAIN_LOSS.append(umbrellaCost/BATCH_NUM)
                HAT_TRAIN_LOSS.append(hatCost/BATCH_NUM)
                AGE_TRAIN_LOSS.append(ageCost/BATCH_NUM)
                GENDER_TRAIN_LOSS.append(genderCost/BATCH_NUM)

                UPPERCOLOR_TRAIN_ACC.append(uppercolorAcc/BATCH_NUM)
                UPPERSLEEVE_TRAIN_ACC.append(uppersleeveAcc/BATCH_NUM)
                LOWERCOLOR_TRAIN_ACC.append(lowercolorAcc/BATCH_NUM)
                LOWERTYPE_TRAIN_ACC.append(lowertypeAcc/BATCH_NUM)
                LOWERLENGTH_TRAIN_ACC.append(lowerlengthAcc/BATCH_NUM)
                BACKPACK_TRAIN_ACC.append(backpackAcc/BATCH_NUM)
                HANDBAG_TRAIN_ACC.append(handbagAcc/BATCH_NUM)
                UMBRELLA_TRAIN_ACC.append(umbrellaAcc/BATCH_NUM)
                HAT_TRAIN_ACC.append(hatAcc/BATCH_NUM)
                AGE_TRAIN_ACC.append(ageAcc/BATCH_NUM)
                GENDER_TRAIN_ACC.append(genderAcc/BATCH_NUM)

                # Validation Set
                uppercolorCost = 0
                uppersleeveCost = 0
                lowercolorCost = 0
                lowertypeCost = 0
                lowerlengthCost = 0
                backpackCost = 0
                handbagCost = 0
                umbrellaCost = 0
                hatCost = 0
                ageCost = 0
                genderCost = 0

                uppercolorAcc = 0
                uppersleeveAcc = 0
                lowercolorAcc = 0
                lowertypeAcc = 0
                lowerlengthAcc = 0
                backpackAcc = 0
                handbagAcc = 0
                umbrellaAcc = 0
                hatAcc = 0
                ageAcc = 0
                genderAcc = 0

                for step in range(VAL_BATCH_NUM):
                    Image_Batch, upperColor_Batch, upperSleeve_Batch, lowerColor_Batch, lowerType_Batch, lowerLength_Batch, \
                        backpack_Batch, handbag_Batch, umbrella_Batch, hat_Batch, age_Batch, gender_Batch =                 \
                        sess.run([imagesVal, upperColorVal, upperSleeveVal, lowerColorVal, lowerTypeVal, lowerLengthVal,    \
                        backpackVal, handbagVal, umbrellaVal, hatVal, ageVal, genderVal])
                    Image_Scaler = mms.mms_trans(Image_Batch, MMS_PATH)

                    # Data Augumentation
                    Image_Augumentation = cfg.DataAugmentation(Image_Scaler, False)
                    Image_Reshaped = np.reshape(Image_Augumentation, (cfg.BATCH_SIZE, model.INPUT_WIDTH, model.INPUT_HEIGHT, cfg.NUM_CHANNELS))

                    # Encode Label
                    upperColor_Encode = encode_labels(upperColor_Batch, cfg.UPPER_COLOR_NUM)
                    upperSleeve_Encode = encode_labels(upperSleeve_Batch, cfg.UPPER_SLEEVE_NUM)
                    lowerColor_Encode = encode_labels(lowerColor_Batch, cfg.LOWER_COLOR_NUM)
                    lowerType_Encode = encode_labels(lowerType_Batch, cfg.LOWER_TYPE_NUM)
                    lowerLength_Encode = encode_labels(lowerLength_Batch, cfg.LOWER_LENGTH_NUM)
                    backpack_Encode = encode_labels(backpack_Batch, cfg.BACKPACK_NUM)
                    handbag_Encode = encode_labels(handbag_Batch, cfg.HANDBAG_NUM)
                    umbrella_Encode = encode_labels(umbrella_Batch, cfg.UMBRELLA_NUM)
                    hat_Encode = encode_labels(hat_Batch, cfg.HAT_NUM)
                    age_Encode = encode_labels(age_Batch, cfg.AGE_NUM)
                    gender_Encode = encode_labels(gender_Batch, cfg.GENDER_NUM)

                    # Fit training using batch data
                    loss_value, upperColor_Cost, upperSleeve_Cost, lowerColor_Cost, lowerType_Cost, lowerLength_Cost,        \
                        backpack_Cost, handbag_Cost, umbrella_Cost, hat_Cost, age_Cost, gender_Cost,                         \
                        upperColor_Acc  , upperSleeve_Acc    , lowerColor_Acc    , lowerType_Acc    , lowerLength_Acc    ,   \
                        backpack_Acc, handbag_Acc, umbrella_Acc, hat_Acc, age_Acc, gender_Acc = sess.run([                   \
                        loss, upperColor_CrossEntropy_Loss, upperSleeve_CrossEntropy_Loss, lowerColor_CrossEntropy_Loss,     \
                        lowerType_CrossEntropy_Loss, lowerLength_CrossEntropy_Loss, backpack_CrossEntropy_Loss,              \
                        handbag_CrossEntropy_Loss, umbrella_CrossEntropy_Loss, hat_CrossEntropy_Loss, age_CrossEntropy_Loss, \
                        gender_CrossEntropy_Loss,   \
                        upperColor_Accuracy , upperSleeve_Accuracy, lowerColor_Accuracy, lowerType_Accuracy,                 \
                        lowerLength_Accuracy, backpack_Accuracy   , handbag_Accuracy   , umbrella_Accuracy , hat_Accuracy,   \
                        age_Accuracy, gender_Accuracy], feed_dict={           \
                        input_Image: Image_Reshaped,                                                                         \
                        upperColor_Label : upperColor_Encode , upperSleeve_Label: upperSleeve_Encode,                        \
                        lowerColor_Label : lowerColor_Encode , lowerType_Label  : lowerType_Encode  ,                        \
                        lowerLength_Label: lowerLength_Encode, backpack_Label   : backpack_Encode   ,                        \
                        handbag_Label    : handbag_Encode    , umbrella_Label   : umbrella_Encode   ,                        \
                        hat_Label        : hat_Encode, age_Label: age_Encode, gender_Label: gender_Encode})

                    uppercolorCost += upperColor_Cost
                    uppersleeveCost += upperSleeve_Cost
                    lowercolorCost += lowerColor_Cost
                    lowertypeCost += lowerType_Cost
                    lowerlengthCost += lowerLength_Cost
                    backpackCost += backpack_Cost
                    handbagCost += handbag_Cost
                    umbrellaCost += umbrella_Cost
                    hatCost += hat_Cost
                    ageCost += age_Cost
                    genderCost += gender_Cost

                    uppercolorAcc += upperColor_Acc
                    uppersleeveAcc += upperSleeve_Acc
                    lowercolorAcc += lowerColor_Acc
                    lowertypeAcc += lowerType_Acc
                    lowerlengthAcc += lowerLength_Acc
                    backpackAcc += backpack_Acc
                    handbagAcc += handbag_Acc
                    umbrellaAcc += umbrella_Acc
                    hatAcc += hat_Acc
                    ageAcc += age_Acc
                    genderAcc += gender_Acc

                UPPERCOLOR_VAL_LOSS.append(uppercolorCost/VAL_BATCH_NUM)
                UPPERSLEEVE_VAL_LOSS.append(uppersleeveCost/VAL_BATCH_NUM)
                LOWERCOLOR_VAL_LOSS.append(lowercolorCost/VAL_BATCH_NUM)
                LOWERTYPE_VAL_LOSS.append(lowertypeCost/VAL_BATCH_NUM)
                LOWERLENGTH_VAL_LOSS.append(lowerlengthCost/VAL_BATCH_NUM)
                BACKPACK_VAL_LOSS.append(backpackCost/VAL_BATCH_NUM)
                HANDBAG_VAL_LOSS.append(handbagCost/VAL_BATCH_NUM)
                UMBRELLA_VAL_LOSS.append(umbrellaCost/VAL_BATCH_NUM)
                HAT_VAL_LOSS.append(hatCost/VAL_BATCH_NUM)
                AGE_VAL_LOSS.append(ageCost/VAL_BATCH_NUM)
                GENDER_VAL_LOSS.append(genderCost/VAL_BATCH_NUM)
                UPPERCOLOR_VAL_ACC.append(uppercolorAcc/VAL_BATCH_NUM)
                UPPERSLEEVE_VAL_ACC.append(uppersleeveAcc/VAL_BATCH_NUM)
                LOWERCOLOR_VAL_ACC.append(lowercolorAcc/VAL_BATCH_NUM)
                LOWERTYPE_VAL_ACC.append(lowertypeAcc/VAL_BATCH_NUM)
                LOWERLENGTH_VAL_ACC.append(lowerlengthAcc/VAL_BATCH_NUM)
                BACKPACK_VAL_ACC.append(backpackAcc/VAL_BATCH_NUM)
                HANDBAG_VAL_ACC.append(handbagAcc/VAL_BATCH_NUM)
                UMBRELLA_VAL_ACC.append(umbrellaAcc/VAL_BATCH_NUM)
                HAT_VAL_ACC.append(hatAcc/VAL_BATCH_NUM)
                AGE_VAL_ACC.append(ageAcc/VAL_BATCH_NUM)
                GENDER_VAL_ACC.append(genderAcc/VAL_BATCH_NUM)

        coord.request_stop()
        coord.join(threads)
        print("Optimization Finished!")
        saver.save(sess, os.path.join(MODEL_SAVE_PATH+MODEL_NAME+"/", (MODEL_NAME + "_part" + str(partIndex) + "_final")))
        saver.save(sess, os.path.join(MODEL_SAVE_PATH+MODEL_NAME+"/Finetune/", (MODEL_NAME + "_part" + str(partIndex) + "_final_finetune")))
        print("Save model...")

"""def cfg.DataAugmentation(batch_x, train):
    # Data Augmentation
    ranID = np.random.permutation(batch_x.shape[0])
    augmetation_xs = np.zeros((batch_x.shape[0], model.INPUT_WIDTH*model.INPUT_HEIGHT*cfg.NUM_CHANNELS), dtype=float)
    index = 0
    
    # Make Interesting Bounding Box
    # All Body (x, y) = (50, 100), (width, height) = (125, 250)
    BBX1 = 50
    BBY1 = 20
    BBX2 = 175
    BBY2 = 350
    while index < batch_x.shape[0]:
        # Train Data Will Do Data Augumentation
        # Validation Data and Test Data Will Not Do Augumentation
        image = batch_x[index,:]
        if train:
            # Decide Whether Do Augumentation
            # Scale Bounding Box
            agScale = random.randint(0, 2)
            sizeScale = 0
            if agScale == 1:
                # Scale Bounding Box +10 Pixels Each Edge
                sizeScale = 10
            elif agScale == 2:
                # Scale Bounding Box -10 Pixels Each Edge
                sizeScale = -10

            # Move Bounding Box
            moveX = random.randint(-5, 5)
            moveY = random.randint(-5, 5)

            # Mirror Or Not
            agMirror = random.randint(0, 1)
            if agMirror == 1:
                # Mirror
                imgBBOG = cv2.resize(cv2.flip(image[BBY1-sizeScale+moveY:BBY2+sizeScale+moveY,                                      \
                    BBX1-sizeScale+moveX:BBX2+sizeScale+moveX, :], 1), (model.INPUT_WIDTH, model.INPUT_HEIGHT), interpolation=cv2.INTER_CUBIC)
            else:
                imgBBOG = cv2.resize(image[BBY1-sizeScale+moveY:BBY2+sizeScale+moveY, BBX1-sizeScale+moveX:BBX2+sizeScale+moveX, :],\
                    (model.INPUT_WIDTH, model.INPUT_HEIGHT), interpolation=cv2.INTER_CUBIC)
        else:
            # Origin Data
            imgBBOG = cv2.resize(image[BBY1:BBY2, BBX1:BBX2], (model.INPUT_WIDTH, model.INPUT_HEIGHT), interpolation=cv2.INTER_CUBIC)
        augmetation_xs[index] = np.reshape(imgBBOG, (-1, model.INPUT_WIDTH*model.INPUT_HEIGHT*cfg.NUM_CHANNELS))
        index += 1
    return augmetation_xs"""

def look_model_weight(mode, partIndex):
    tf.reset_default_graph()
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        print("Restore  Start!")
        if mode == 1:
            saver = tf.train.import_meta_graph(os.path.join(MODEL_SAVE_PATH+MODEL_NAME+"/", (MODEL_NAME + "_part" + str(partIndex) + "_final.meta")))
            saver.restore(sess, os.path.join(MODEL_SAVE_PATH+MODEL_NAME+"/", (MODEL_NAME + "_part" + str(partIndex) + "_final")))
        elif mode == 2:
            saver = tf.train.import_meta_graph(os.path.join(MODEL_SAVE_PATH+MODEL_NAME+"/Finetune/", (MODEL_NAME + "_part" + str(partIndex) + "_final_finetune.meta")))
            saver.restore(sess, os.path.join(MODEL_SAVE_PATH+MODEL_NAME+"/Finetune/", (MODEL_NAME + "_part" + str(partIndex) + "_final_finetune")))
        print("Restore  Finished!")
        #get_all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        #print(len(get_all_vars))
        all_vars = tf.trainable_variables()
        for v in all_vars:
            if v.name == "Layer1_Convonlution1/weight:0":
                v_weight = sess.run(v)
                print("%s with value %s" % (v.name, v_weight[0][0]))

def finetune(TFRecord_Train_Name, train_Number, partIndex, FINETUNE_NUM, FINETUNE_CLASS, finetune_layer):
    tf.reset_default_graph()
    BATCH_NUM = int(train_Number/cfg.BATCH_SIZE)

    config = tf.ConfigProto(allow_soft_placement=True)
    
    # Get All Data From TFRData
    with tf.Session(config=config) as sess1:
        # Construct Coordinate
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        filename_queue_finetune = tf.train.string_input_producer([TFRecord_Train_Name], shuffle = True, num_epochs=1)
        # Read TFRecords Data
        imagesTrain, upperColorTrain, upperSleeveTrain, lowerColorTrain, lowerTypeTrain, lowerLengthTrain,  \
            backpackTrain, handbagTrain, umbrellaTrain, hatTrain, ageTrain, genderTrain =                   \
            TFR.read_and_decode(filename_queue_finetune, cfg.BATCH_SIZE)
        init_local = tf.local_variables_initializer()

        sess1.run(init_local)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for getIndex in range(BATCH_NUM):
            Image_Get, upperColor_Get, upperSleeve_Get, lowerColor_Get, lowerType_Get, lowerLength_Get, \
                backpack_Get, handbag_Get, umbrella_Get, hat_Get, age_Get, gender_Get = sess1.run([imagesTrain,\
                upperColorTrain, upperSleeveTrain, lowerColorTrain, lowerTypeTrain, lowerLengthTrain,   \
                backpackTrain, handbagTrain, umbrellaTrain, hatTrain, ageTrain, genderTrain])
            if getIndex == 0:
                Image_Set = Image_Get
                upperColor_Set = upperColor_Get
                upperSleeve_Set = upperSleeve_Get
                lowerColor_Set = lowerColor_Get
                lowerType_Set = lowerType_Get
                lowerLength_Set = lowerLength_Get
                backpack_Set = backpack_Get
                handbag_Set = handbag_Get
                umbrella_Set = umbrella_Get
                hat_Set = hat_Get
                age_Set = age_Get
                gender_Set = gender_Get
                continue
            Image_Set = np.concatenate((Image_Set, Image_Get), axis = 0)
            upperColor_Set = np.concatenate((upperColor_Set, upperColor_Get), axis = 0)
            upperSleeve_Set = np.concatenate((upperSleeve_Set, upperSleeve_Get), axis = 0)
            lowerColor_Set = np.concatenate((lowerColor_Set, lowerColor_Get), axis = 0)
            lowerType_Set = np.concatenate((lowerType_Set, lowerType_Get), axis = 0)
            lowerLength_Set = np.concatenate((lowerLength_Set, lowerLength_Get), axis = 0)
            backpack_Set = np.concatenate((backpack_Set, backpack_Get), axis = 0)
            handbag_Set = np.concatenate((handbag_Set, handbag_Get), axis = 0)
            umbrella_Set = np.concatenate((umbrella_Set, umbrella_Get), axis = 0)
            hat_Set = np.concatenate((hat_Set, hat_Get), axis = 0)
            age_Set = np.concatenate((age_Set, age_Get), axis = 0)
            gender_Set = np.concatenate((gender_Set, gender_Get), axis = 0)
        coord.request_stop()
        coord.join(threads)

    # Fine-tune
    with tf.Session(config=config) as sess:
        input_Image_ = tf.placeholder(tf.float32, [None, model.INPUT_WIDTH*model.INPUT_HEIGHT*cfg.NUM_CHANNELS], name='input_Image')
        input_Image = tf.reshape(input_Image_, shape=[-1, model.INPUT_WIDTH, model.INPUT_HEIGHT, cfg.NUM_CHANNELS])

        upperColor_Label  = tf.placeholder(tf.float32, [None, cfg.UPPER_COLOR_NUM],  name='upperColor_Label')
        upperSleeve_Label = tf.placeholder(tf.float32, [None, cfg.UPPER_SLEEVE_NUM], name='upperSleeve_Label')
        lowerColor_Label  = tf.placeholder(tf.float32, [None, cfg.LOWER_COLOR_NUM],  name='lowerColor_Label')
        lowerType_Label   = tf.placeholder(tf.float32, [None, cfg.LOWER_TYPE_NUM],   name='lowerType_Label')
        lowerLength_Label = tf.placeholder(tf.float32, [None, cfg.LOWER_LENGTH_NUM], name='lowerLength_Label')
        backpack_Label    = tf.placeholder(tf.float32, [None, cfg.BACKPACK_NUM],     name='backpack_Label')
        handbag_Label     = tf.placeholder(tf.float32, [None, cfg.HANDBAG_NUM],      name='handbag_Label')
        umbrella_Label    = tf.placeholder(tf.float32, [None, cfg.UMBRELLA_NUM],     name='umbrella_Label')
        hat_Label         = tf.placeholder(tf.float32, [None, cfg.HAT_NUM],          name='hat_Label')
        age_Label         = tf.placeholder(tf.float32, [None, cfg.AGE_NUM],          name='age_Label')
        gender_Label      = tf.placeholder(tf.float32, [None, cfg.GENDER_NUM],       name='gender_Label')

        regularizer = tf.contrib.layers.l2_regularizer(cfg.REGULARIZATION_RATE)

        # From Model Get Predict Answer
        upperColor_Output, upperSleeve_Output, lowerColor_Output, lowerType_Output, lowerLength_Output, backpack_Output, \
            handbag_Output, umbrella_Output, hat_Output, age_Output, gender_Output = model.interface(input_Image, True, regularizer)

        global_step = tf.Variable(0, trainable=False)

        """# Predict
        upperColor_Pred = tf.argmax(upperColor_Output, 1)
        upperSleeve_Pred = tf.argmax(upperSleeve_Output, 1)
        lowerColor_Pred = tf.argmax(lowerColor_Output, 1)
        lowerType_Pred = tf.argmax(lowerType_Output, 1)
        lowerLength_Pred = tf.argmax(lowerLength_Output, 1)
        backpack_Pred = tf.argmax(backpack_Output, 1)
        handbag_Pred = tf.argmax(handbag_Output, 1)
        umbrella_Pred = tf.argmax(umbrella_Output, 1)
        hat_Pred = tf.argmax(hat_Output, 1)
        age_Pred = tf.argmax(age_Output, 1)
        gender_Pred = tf.argmax(gender_Output, 1)"""

        """# Expect
        upperColor_Real = tf.argmax(upperColor_Label, 1)
        upperSleeve_Real = tf.argmax(upperSleeve_Label, 1)
        lowerColor_Real = tf.argmax(lowerColor_Label, 1)
        lowerType_Real = tf.argmax(lowerType_Label, 1)
        lowerLength_Real = tf.argmax(lowerLength_Label, 1)
        backpack_Real = tf.argmax(backpack_Label, 1)
        handbag_Real = tf.argmax(handbag_Label, 1)
        umbrella_Real = tf.argmax(umbrella_Label, 1)
        hat_Real = tf.argmax(hat_Label, 1)
        age_Real = tf.argmax(age_Label, 1)
        gender_Real = tf.argmax(gender_Label, 1)"""

        """# Correct
        upperColor_Correct = tf.equal(upperColor_Pred, upperColor_Real)
        upperSleeve_Correct = tf.equal(upperSleeve_Pred, upperSleeve_Real)
        lowerColor_Correct = tf.equal(lowerColor_Pred, lowerColor_Real)
        lowerType_Correct = tf.equal(lowerType_Pred, lowerType_Real)
        lowerLength_Correct = tf.equal(lowerLength_Pred, lowerLength_Real)
        backpack_Correct = tf.equal(backpack_Pred, backpack_Real)
        handbag_Correct = tf.equal(handbag_Pred, handbag_Real)
        umbrella_Correct = tf.equal(umbrella_Pred, umbrella_Real)
        hat_Correct = tf.equal(hat_Pred, hat_Real)
        age_Correct = tf.equal(age_Pred, age_Real)
        gender_Correct = tf.equal(gender_Pred, gender_Real)"""

        """# Accuracy
        upperColor_Accuracy = tf.reduce_mean(tf.cast(upperColor_Correct, tf.float32))
        upperSleeve_Accuracy = tf.reduce_mean(tf.cast(upperSleeve_Correct, tf.float32))
        lowerColor_Accuracy = tf.reduce_mean(tf.cast(lowerColor_Correct, tf.float32))
        lowerType_Accuracy = tf.reduce_mean(tf.cast(lowerType_Correct, tf.float32))
        lowerLength_Accuracy = tf.reduce_mean(tf.cast(lowerLength_Correct, tf.float32))
        backpack_Accuracy = tf.reduce_mean(tf.cast(backpack_Correct, tf.float32))
        handbag_Accuracy = tf.reduce_mean(tf.cast(handbag_Correct, tf.float32))
        umbrella_Accuracy = tf.reduce_mean(tf.cast(umbrella_Correct, tf.float32))
        hat_Accuracy = tf.reduce_mean(tf.cast(hat_Correct, tf.float32))
        age_Accuracy = tf.reduce_mean(tf.cast(age_Correct, tf.float32))
        gender_Accuracy = tf.reduce_mean(tf.cast(gender_Correct, tf.float32))"""

        """# Cross Entropy
        upperColor_CrossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits( \
            logits=upperColor_Output, labels=tf.argmax(upperColor_Label, 1))
        upperSleeve_CrossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits(\
            logits=upperSleeve_Output, labels=tf.argmax(upperSleeve_Label, 1))
        lowerColor_CrossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits( \
            logits=lowerColor_Output, labels=tf.argmax(lowerColor_Label, 1))
        lowerType_CrossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits(  \
            logits=lowerType_Output, labels=tf.argmax(lowerType_Label, 1))
        lowerLength_CrossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits(\
            logits=lowerLength_Output, labels=tf.argmax(lowerLength_Label, 1))
        backpack_CrossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits(   \
            logits=backpack_Output, labels=tf.argmax(backpack_Label, 1))
        handbag_CrossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits(    \
            logits=handbag_Output, labels=tf.argmax(handbag_Label, 1))
        umbrella_CrossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits(   \
            logits=umbrella_Output, labels=tf.argmax(umbrella_Label, 1))
        hat_CrossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits(        \
            logits=hat_Output, labels=tf.argmax(hat_Label, 1))
        age_CrossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits(        \
            logits=age_Output, labels=tf.argmax(age_Label, 1))
        gender_CrossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits(     \
            logits=gender_Output, labels=tf.argmax(gender_Label, 1))"""

        """# Loss
        upperColor_CrossEntropy_Loss = loss_function(upperColor_CrossEntropy, upperColor_Real, cfg.UPPER_COLOR_NUM)
        upperSleeve_CrossEntropy_Loss = loss_function(upperSleeve_CrossEntropy, upperSleeve_Real, cfg.UPPER_SLEEVE_NUM)
        lowerColor_CrossEntropy_Loss = loss_function(lowerColor_CrossEntropy, lowerColor_Real, cfg.LOWER_COLOR_NUM)
        lowerType_CrossEntropy_Loss = loss_function(lowerType_CrossEntropy, lowerType_Real, cfg.LOWER_TYPE_NUM)
        lowerLength_CrossEntropy_Loss = loss_function(lowerLength_CrossEntropy, lowerLength_Real, cfg.LOWER_LENGTH_NUM)
        backpack_CrossEntropy_Loss = loss_function(backpack_CrossEntropy, backpack_Real, cfg.BACKPACK_NUM)
        handbag_CrossEntropy_Loss = loss_function(handbag_CrossEntropy, handbag_Real, cfg.HANDBAG_NUM)
        umbrella_CrossEntropy_Loss = loss_function(umbrella_CrossEntropy, umbrella_Real, cfg.UMBRELLA_NUM)
        hat_CrossEntropy_Loss = loss_function(hat_CrossEntropy, hat_Real, cfg.HAT_NUM)
        age_CrossEntropy_Loss = loss_function(age_CrossEntropy, age_Real, cfg.AGE_NUM)
        gender_CrossEntropy_Loss = loss_function(gender_CrossEntropy, gender_Real, cfg.GENDER_NUM)"""

        if finetune_layer == "UpperSleeve_FullConnect":
            Output = upperSleeve_Output
            Label = upperSleeve_Label
        elif finetune_layer == "LowerType_FullConnect":
            Output = lowerType_Output
            Label = lowerType_Label
        elif finetune_layer == "LowerLength_FullConnect":
            Output = lowerLength_Output
            Label = lowerLength_Label
        elif finetune_layer == "Backpack_FullConnect":
            Output = backpack_Output
            Label = backpack_Label
        elif finetune_layer == "Handbag_FullConnect":
            Output = handbag_Output
            Label = handbag_Label
        elif finetune_layer == "Umbrella_FullConnect":
            Output = umbrella_Output
            Label = umbrella_Label
        elif finetune_layer == "Hat_FullConnect":
            Output = hat_Output
            Label = hat_Label
        elif finetune_layer == "Age_FullConnect":
            Output = age_Output
            Label = age_Label
        elif finetune_layer == "Gender_FullConnect":
            Output = gender_Output
            Label = gender_Label
            
        Pred = tf.argmax(Output, 1)
        Real = tf.argmax(Label, 1)
        Correct = tf.equal(Pred, Real)
        Accuracy = tf.reduce_mean(tf.cast(Correct, tf.float32))
        CrossEntropy_Loss = tf.nn.sparse_softmax_cross_entropy_with_logits( \
            logits=Output, labels=tf.argmax(Label, 1))
                
        if finetune_layer == "UpperSleeve_FullConnect":
            loss = loss_function(CrossEntropy_Loss, Real, cfg.UPPER_SLEEVE_NUM)
        elif finetune_layer == "LowerType_FullConnect":
            loss = loss_function(CrossEntropy_Loss, Real, cfg.LOWER_TYPE_NUM)
        elif finetune_layer == "LowerLength_FullConnect":
            loss = loss_function(CrossEntropy_Loss, Real, cfg.LOWER_LENGTH_NUM)
        elif finetune_layer == "Backpack_FullConnect":
            loss = loss_function(CrossEntropy_Loss, Real, cfg.BACKPACK_NUM)
        elif finetune_layer == "Handbag_FullConnect":
            loss = loss_function(CrossEntropy_Loss, Real, cfg.HANDBAG_NUM)
        elif finetune_layer == "Umbrella_FullConnect":
            loss = loss_function(CrossEntropy_Loss, Real, cfg.UMBRELLA_NUM)
        elif finetune_layer == "Hat_FullConnect":
            loss = loss_function(CrossEntropy_Loss, Real, cfg.HAT_NUM)
        elif finetune_layer == "Age_FullConnect":
            loss = loss_function(CrossEntropy_Loss, Real, cfg.AGE_NUM)
        elif finetune_layer == "Gender_FullConnect":
            loss = loss_function(CrossEntropy_Loss, Real, cfg.GENDER_NUM)

        optimizer = tf.train.AdamOptimizer(cfg.FINETUNE_LEARNING_RATE_BASE)

        finetune_var = []
        finetune_var.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, finetune_layer + "1"))
        finetune_var.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, finetune_layer + "2"))

        update_ops = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        with tf.control_dependencies(update_ops):
            finetune_op = optimizer.minimize(loss, global_step = global_step, var_list=finetune_var)

        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        saver = tf.train.Saver(update_ops)

        # Load Old Model
        print ("Restore  Start!")
        sess.run(init)
        saver = tf.train.import_meta_graph(os.path.join(MODEL_SAVE_PATH+MODEL_NAME+"/Finetune/", (MODEL_NAME + "_part" + str(partIndex) + "_final_finetune.meta")))
        saver.restore(sess, os.path.join(MODEL_SAVE_PATH+MODEL_NAME+"/Finetune/", (MODEL_NAME + "_part" + str(partIndex) + "_final_finetune")))
        print ("Restore  Finished!")


        # Every Class First Index In Training Array, except UpperColor and LowerColor
        finetune_Index = np.zeros(FINETUNE_NUM, dtype=int)

        print(BATCH_NUM)
        print("Start Finetune!")

        Finetune_Class = FINETUNE_CLASS
        print("Finetune Class %d" % (Finetune_Class))
        for epoch in range(cfg.FINETUNE_Epoch):
            isTrain = True
            randomSetID = np.random.permutation(np.shape(Image_Set)[0])
            Image_Set = Image_Set[randomSetID]
            upperColor_Set = upperColor_Set[randomSetID]
            upperSleeve_Set = upperSleeve_Set[randomSetID]
            lowerColor_Set = lowerColor_Set[randomSetID]
            lowerType_Set = lowerType_Set[randomSetID]
            lowerLength_Set = lowerLength_Set[randomSetID]
            backpack_Set = backpack_Set[randomSetID]
            handbag_Set = handbag_Set[randomSetID]
            umbrella_Set = umbrella_Set[randomSetID]
            hat_Set = hat_Set[randomSetID]
            age_Set = age_Set[randomSetID]
            gender_Set = gender_Set[randomSetID]

            for step in range(cfg.FINETUNE_STEPS):
                if Finetune_Class == 0:
                    finetune_Index, batchID = cfg.getBalanceID(upperColor_Set, FINETUNE_NUM, finetune_Index)
                elif Finetune_Class == 1:
                    finetune_Index, batchID = cfg.getBalanceID(upperSleeve_Set, FINETUNE_NUM, finetune_Index)
                elif Finetune_Class == 2:
                    finetune_Index, batchID = cfg.getBalanceID(lowerColor_Set, FINETUNE_NUM, finetune_Index)
                elif Finetune_Class == 3:
                    finetune_Index, batchID = cfg.getBalanceID(lowerType_Set, FINETUNE_NUM, finetune_Index)
                elif Finetune_Class == 4:
                    finetune_Index, batchID = cfg.getBalanceID(lowerLength_Set, FINETUNE_NUM, finetune_Index)
                elif Finetune_Class == 5:
                    finetune_Index, batchID = cfg.getBalanceID(backpack_Set, FINETUNE_NUM, finetune_Index)
                elif Finetune_Class == 6:
                    finetune_Index, batchID = cfg.getBalanceID(handbag_Set, FINETUNE_NUM, finetune_Index)
                elif Finetune_Class == 7:
                    finetune_Index, batchID = cfg.getBalanceID(umbrella_Set, FINETUNE_NUM, finetune_Index)
                elif Finetune_Class == 8:
                    finetune_Index, batchID = cfg.getBalanceID(hat_Set, FINETUNE_NUM, finetune_Index)
                elif Finetune_Class == 9:
                    finetune_Index, batchID = cfg.getBalanceID(age_Set, FINETUNE_NUM, finetune_Index)
                elif Finetune_Class == 10:
                    finetune_Index, batchID = cfg.getBalanceID(gender_Set, FINETUNE_NUM, finetune_Index)

                ranID = np.random.permutation(cfg.BATCH_SIZE)
                batchID = batchID[ranID]
                Image_Batch = Image_Set[batchID]
                upperColor_Batch = upperColor_Set[batchID]
                upperSleeve_Batch = upperSleeve_Set[batchID]
                lowerColor_Batch = lowerColor_Set[batchID]
                lowerType_Batch = lowerType_Set[batchID]
                lowerLength_Batch = lowerLength_Set[batchID]
                backpack_Batch = backpack_Set[batchID]
                handbag_Batch = handbag_Set[batchID]
                umbrella_Batch = umbrella_Set[batchID]
                hat_Batch = hat_Set[batchID]
                age_Batch = age_Set[batchID]
                gender_Batch = gender_Set[batchID]

                Image_Scaler = mms.mms_trans(Image_Batch, MMS_PATH)

                # Data Augumentation
                Image_Augumentation = cfg.DataAugmentation(Image_Scaler, True)
                Image_Reshaped = np.reshape(Image_Augumentation, (cfg.BATCH_SIZE, model.INPUT_WIDTH, model.INPUT_HEIGHT, cfg.NUM_CHANNELS))

                # Encode Label
                upperColor_Encode = encode_labels(upperColor_Batch, cfg.UPPER_COLOR_NUM)
                upperSleeve_Encode = encode_labels(upperSleeve_Batch, cfg.UPPER_SLEEVE_NUM)
                lowerColor_Encode = encode_labels(lowerColor_Batch, cfg.LOWER_COLOR_NUM)
                lowerType_Encode = encode_labels(lowerType_Batch, cfg.LOWER_TYPE_NUM)
                lowerLength_Encode = encode_labels(lowerLength_Batch, cfg.LOWER_LENGTH_NUM)
                backpack_Encode = encode_labels(backpack_Batch, cfg.BACKPACK_NUM)
                handbag_Encode = encode_labels(handbag_Batch, cfg.HANDBAG_NUM)
                umbrella_Encode = encode_labels(umbrella_Batch, cfg.UMBRELLA_NUM)
                hat_Encode = encode_labels(hat_Batch, cfg.HAT_NUM)
                age_Encode = encode_labels(age_Batch, cfg.AGE_NUM)
                gender_Encode = encode_labels(gender_Batch, cfg.GENDER_NUM)

                Predict, Cost, Acc, Confidence, opt = sess.run([Pred, CrossEntropy_Loss, Accuracy,\
                    Output, finetune_op], feed_dict={input_Image: Image_Reshaped,                 \
                    upperColor_Label : upperColor_Encode , upperSleeve_Label: upperSleeve_Encode, \
                    lowerColor_Label : lowerColor_Encode , lowerType_Label  : lowerType_Encode  , \
                    lowerLength_Label: lowerLength_Encode, backpack_Label   : backpack_Encode   , \
                    handbag_Label    : handbag_Encode    , umbrella_Label   : umbrella_Encode   , \
                    hat_Label        : hat_Encode, age_Label: age_Encode, gender_Label: gender_Encode})
                """if Finetune_Class == 1:
                    upperSleeve_Predict, upperSleeve_Cost, upperSleeve_Acc, upperSleeve_Confidence, upperSleeve_opt = \
                        sess.run([upperSleeve_Pred, upperSleeve_CrossEntropy_Loss, upperSleeve_Accuracy,              \
                        upperSleeve_Output, finetune_op], feed_dict={input_Image: Image_Reshaped,                  \
                        upperColor_Label : upperColor_Encode , upperSleeve_Label: upperSleeve_Encode, \
                        lowerColor_Label : lowerColor_Encode , lowerType_Label  : lowerType_Encode  , \
                        lowerLength_Label: lowerLength_Encode, backpack_Label   : backpack_Encode   , \
                        handbag_Label    : handbag_Encode    , umbrella_Label   : umbrella_Encode   , \
                        hat_Label        : hat_Encode, age_Label: age_Encode, gender_Label: gender_Encode})
                elif Finetune_Class == 3:
                    lowerType_Predict, lowerType_Cost, lowerType_Acc, lowerType_Confidence, lowerType_opt = sess.run([\
                        lowerType_Pred, lowerType_CrossEntropy_Loss, lowerType_Accuracy, lowerType_Output, finetune_op],\
                        feed_dict={input_Image: Image_Reshaped,                                                       \
                        upperColor_Label : upperColor_Encode , upperSleeve_Label: upperSleeve_Encode, \
                        lowerColor_Label : lowerColor_Encode , lowerType_Label  : lowerType_Encode  , \
                        lowerLength_Label: lowerLength_Encode, backpack_Label   : backpack_Encode   , \
                        handbag_Label    : handbag_Encode    , umbrella_Label   : umbrella_Encode   , \
                        hat_Label        : hat_Encode, age_Label: age_Encode, gender_Label: gender_Encode})
                elif Finetune_Class == 4:
                    lowerLength_Predict, lowerLength_Cost, lowerLength_Acc, lowerLength_Confidence, lowerLength_opt = \
                        sess.run([lowerLength_Pred, lowerLength_CrossEntropy_Loss, lowerLength_Accuracy,              \
                        lowerLength_Output, finetune_op], feed_dict={input_Image: Image_Reshaped,                  \
                        upperColor_Label : upperColor_Encode , upperSleeve_Label: upperSleeve_Encode, \
                        lowerColor_Label : lowerColor_Encode , lowerType_Label  : lowerType_Encode  , \
                        lowerLength_Label: lowerLength_Encode, backpack_Label   : backpack_Encode   , \
                        handbag_Label    : handbag_Encode    , umbrella_Label   : umbrella_Encode   , \
                        hat_Label        : hat_Encode, age_Label: age_Encode, gender_Label: gender_Encode})
                elif Finetune_Class == 5:
                    backpack_Predict, backpack_Cost, backpack_Acc, backpack_Confidence, backpack_opt = sess.run([     \
                        backpack_Pred, backpack_CrossEntropy_Loss, backpack_Accuracy, backpack_Output, finetune_op],  \
                        feed_dict={input_Image: Image_Reshaped,                                                       \
                        upperColor_Label : upperColor_Encode , upperSleeve_Label: upperSleeve_Encode, \
                        lowerColor_Label : lowerColor_Encode , lowerType_Label  : lowerType_Encode  , \
                        lowerLength_Label: lowerLength_Encode, backpack_Label   : backpack_Encode   , \
                        handbag_Label    : handbag_Encode    , umbrella_Label   : umbrella_Encode   , \
                        hat_Label        : hat_Encode, age_Label: age_Encode, gender_Label: gender_Encode})
                elif Finetune_Class == 6:
                    handbag_Predict, handbag_Cost, handbag_Acc, handbag_Confidence, handbag_opt = sess.run([          \
                        handbag_Pred, handbag_CrossEntropy_Loss, handbag_Accuracy, handbag_Output, finetune_op],       \
                        feed_dict={input_Image: Image_Reshaped,                                                       \
                        upperColor_Label : upperColor_Encode , upperSleeve_Label: upperSleeve_Encode, \
                        lowerColor_Label : lowerColor_Encode , lowerType_Label  : lowerType_Encode  , \
                        lowerLength_Label: lowerLength_Encode, backpack_Label   : backpack_Encode   , \
                        handbag_Label    : handbag_Encode    , umbrella_Label   : umbrella_Encode   , \
                        hat_Label        : hat_Encode, age_Label: age_Encode, gender_Label: gender_Encode})
                elif Finetune_Class == 7:
                    umbrella_Predict, umbrella_Cost, umbrella_Acc, umbrella_Confidence, umbrella_opt = sess.run([     \
                        umbrella_Pred, umbrella_CrossEntropy_Loss, umbrella_Accuracy, umbrella_Output, finetune_op],  \
                        feed_dict={input_Image: Image_Reshaped,                                                       \
                        upperColor_Label : upperColor_Encode , upperSleeve_Label: upperSleeve_Encode, \
                        lowerColor_Label : lowerColor_Encode , lowerType_Label  : lowerType_Encode  , \
                        lowerLength_Label: lowerLength_Encode, backpack_Label   : backpack_Encode   , \
                        handbag_Label    : handbag_Encode    , umbrella_Label   : umbrella_Encode   , \
                        hat_Label        : hat_Encode, age_Label: age_Encode, gender_Label: gender_Encode})
                elif Finetune_Class == 8:
                    hat_Predict, hat_Cost, hat_Acc, hat_Confidence, hat_opt = sess.run([                              \
                        hat_Pred, hat_CrossEntropy_Loss, hat_Accuracy, hat_Output, finetune_op], feed_dict={               \
                        input_Image: Image_Reshaped,                                                  \
                        upperColor_Label : upperColor_Encode , upperSleeve_Label: upperSleeve_Encode, \
                        lowerColor_Label : lowerColor_Encode , lowerType_Label  : lowerType_Encode  , \
                        lowerLength_Label: lowerLength_Encode, backpack_Label   : backpack_Encode   , \
                        handbag_Label    : handbag_Encode    , umbrella_Label   : umbrella_Encode   , \
                        hat_Label        : hat_Encode, age_Label: age_Encode, gender_Label: gender_Encode})
                elif Finetune_Class == 9:
                    age_Predict, age_Cost, age_Acc, age_Confidence, age_opt = sess.run([                              \
                        age_Pred, age_CrossEntropy_Loss, age_Accuracy, age_Output, finetune_op], feed_dict={               \
                        input_Image: Image_Reshaped,                                                  \
                        upperColor_Label : upperColor_Encode , upperSleeve_Label: upperSleeve_Encode, \
                        lowerColor_Label : lowerColor_Encode , lowerType_Label  : lowerType_Encode  , \
                        lowerLength_Label: lowerLength_Encode, backpack_Label   : backpack_Encode   , \
                        handbag_Label    : handbag_Encode    , umbrella_Label   : umbrella_Encode   , \
                        hat_Label        : hat_Encode, age_Label: age_Encode, gender_Label: gender_Encode})
                elif Finetune_Class == 10:
                    gender_Predict, gender_Cost, gender_Acc, gender_Confidence, gender_opt = sess.run([                              \
                        gender_Pred, gender_CrossEntropy_Loss, gender_Accuracy, gender_Output, finetune_op], feed_dict={               \
                        input_Image: Image_Reshaped,                                                  \
                        upperColor_Label : upperColor_Encode , upperSleeve_Label: upperSleeve_Encode, \
                        lowerColor_Label : lowerColor_Encode , lowerType_Label  : lowerType_Encode  , \
                        lowerLength_Label: lowerLength_Encode, backpack_Label   : backpack_Encode   , \
                        handbag_Label    : handbag_Encode    , umbrella_Label   : umbrella_Encode   , \
                        hat_Label        : hat_Encode, age_Label: age_Encode, gender_Label: gender_Encode})"""
                        
                if (step+1) % 50 == 0:
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH+MODEL_NAME+"/Finetune/", (MODEL_NAME + "_part" + str(partIndex) + "_final_finetune")))
            if (epoch+1) % 10 == 0:
                print("Finetune Class %d" % (Finetune_Class))
                if Finetune_Class == 1:
                    label_Batch = upperSleeve_Batch
                elif Finetune_Class == 3:
                    label_Batch = lowerType_Batch
                elif Finetune_Class == 4:
                    label_Batch = lowerLength_Batch
                elif Finetune_Class == 5:
                    label_Batch = backpack_Batch
                elif Finetune_Class == 6:
                    label_Batch = handbag_Batch
                elif Finetune_Class == 7:
                    label_Batch = umbrella_Batch
                elif Finetune_Class == 8:
                    label_Batch = hat_Batch
                elif Finetune_Class == 9:
                    label_Batch = age_Batch
                elif Finetune_Class == 10:
                    label_Batch = gender_Batch
                print("Loss %g, Accuracy %g" % (Cost, Acc))
                print("Real\t", label_Batch)
                print("Pred\t", Predict)
                """if Finetune_Class == 1:
                    #print("UpperSleeve Confidence")
                    #print(upperSleeve_Confidence)
                    print("UpperSleeve Loss %g, Accuracy %g" % (upperSleeve_Cost, upperSleeve_Acc))
                    print("UpperSleeve Real\t", upperSleeve_Batch)
                    print("UpperSleeve Pred\t", upperSleeve_Predict)
                elif Finetune_Class == 3:
                    #print("LowerType Confidence")
                    #print(lowerType_Confidence)
                    print("LowerType Loss %g, Accuracy %g" % (lowerType_Cost, lowerType_Acc))
                    print("LowerType Real\t", lowerType_Batch)
                    print("LowerType Pred\t", lowerType_Predict)
                elif Finetune_Class == 4:
                    #print("LowerLength Confidence")
                    #print(lowerLength_Confidence)
                    print("LowerLength Loss %g, Accuracy %g" % (lowerLength_Cost, lowerLength_Acc))
                    print("LowerLength Real\t", lowerLength_Batch)
                    print("LowerLength Pred\t", lowerLength_Predict)
                elif Finetune_Class == 5:
                    #print("Backpack Confidence")
                    #print(backpack_Confidence)
                    print("Backpack Loss %g, Accuracy %g" % (backpack_Cost, backpack_Acc))
                    print("Backpack Real\t", backpack_Batch)
                    print("Backpack Pred\t", backpack_Predict)
                elif Finetune_Class == 6:
                    #print("Handbag Confidence")
                    #print(handbag_Confidence)
                    print("Handbag Loss %g, Accuracy %g" % (handbag_Cost, handbag_Acc))
                    print("Handbag Real\t", handbag_Batch)
                    print("Handbag Pred\t", handbag_Predict)
                elif Finetune_Class == 7:
                    #print("Umbrella Confidence")
                    #print(umbrella_Confidence)
                    print("Umbrella Loss %g, Accuracy %g" % (umbrella_Cost, umbrella_Acc))
                    print("Umbrella Real\t", umbrella_Batch)
                    print("Umbrella Pred\t", umbrella_Predict)
                elif Finetune_Class == 8:
                    #print("Hat Confidence")
                    #print(hat_Confidence)
                    print("Hat Loss %g, Accuracy %g" % (hat_Cost, hat_Acc))
                    print("Hat Real\t", hat_Batch)
                    print("Hat Pred\t", hat_Predict)
                elif Finetune_Class == 10:
                    #print("Gender Confidence")
                    #print(gender_Confidence)
                    print("Gender Loss %g, Accuracy %g" % (gender_Cost, gender_Acc))
                    print("Gender Real\t", gender_Batch)
                    print("Gender Pred\t", gender_Predict)
                elif Finetune_Class == 9:
                    #print("Age Confidence")
                    #print(age_Confidence)
                    print("Age Loss %g, Accuracy %g" % (age_Cost, age_Acc))
                    print("Age Real\t", age_Batch)
                    print("Age Pred\t", age_Predict)"""

        print(BATCH_NUM)
        print("Optimization Finished!")
        saver.save(sess, os.path.join(MODEL_SAVE_PATH+MODEL_NAME+"/Finetune/", (MODEL_NAME + "_part" + str(partIndex) + "_final_finetune")))
        print("Save model...")

"""# Let One Class Data Balance
def cfg.getBalanceID(Label, ClassNumber, Index):
    BatchID = np.zeros(cfg.BATCH_SIZE, dtype=int)
    ClassMax = np.zeros(ClassNumber, dtype=int)
    # Set Each Class Max Number
    quo = cfg.BATCH_SIZE / ClassNumber
    rem = cfg.BATCH_SIZE % ClassNumber
    for ii in range(ClassNumber):
        ClassMax[ii] = quo
    for ii in range(rem):
        ClassMax[ii] += 1
    #print(np.shape(Label))
    jj = 0
    for ii in range(ClassNumber):
        getNumber = 0
        ID = Index[ii]
        while getNumber < ClassMax[ii]:
            if Label[ID] == ii:
                BatchID[jj] = ID
                jj += 1
                getNumber += 1
            ID += 1
            if ID == Label.shape[0]:
                ID = 0
        Index[ii] = ID

    return Index, BatchID"""

def main(argv):
    global cfg.LEARNING_RATE_BASE, cfg.LEARNING_RATE_DECAY, MODEL_NAME, MIN_KM
    if len(argv) >= 1:
        cfg.LEARNING_RATE_BASE = float(argv[0])
    if len(argv) >= 2:
        cfg.LEARNING_RATE_DECAY = float(argv[1])
    if len(argv) >= 3:
        MIN_KM = int(argv[2])
        
    print(USE_KM)
    if USE_KM:
        MODEL_NAME = "PETA_All_Atribute_MINKM" + str(MIN_KM)
    else:
        MODEL_NAME = "PETA_All_Atribute_NoWeighting"

    print(cfg.LEARNING_RATE_BASE, cfg.LEARNING_RATE_DECAY, MIN_KM, MODEL_NAME)

    if not os.path.isdir(MODEL_SAVE_PATH+MODEL_NAME):
        print("Mkdir")
        os.mkdir(MODEL_SAVE_PATH+MODEL_NAME)
        os.mkdir(MODEL_SAVE_PATH+MODEL_NAME+"/Finetune")
        os.mkdir(MODEL_SAVE_PATH+MODEL_NAME+"/Image")
        os.mkdir(MODEL_SAVE_PATH+MODEL_NAME+"/Txt")

    print("\n**********Model**********\n")
    print(model.MODEL_NAME)
    print("\n**********Model**********\n")

    # Make Confusion Matrix
    sum_confusion_matrix_upperColor = np.zeros([cfg.UPPER_COLOR_NUM,cfg.UPPER_COLOR_NUM], dtype=float)
    sum_confusion_matrix_upperSleeve = np.zeros([cfg.UPPER_SLEEVE_NUM,cfg.UPPER_SLEEVE_NUM], dtype=float)
    sum_confusion_matrix_lowerColor = np.zeros([cfg.LOWER_COLOR_NUM,cfg.LOWER_COLOR_NUM], dtype=float)
    sum_confusion_matrix_lowerType = np.zeros([cfg.LOWER_TYPE_NUM,cfg.LOWER_TYPE_NUM], dtype=float)
    sum_confusion_matrix_lowerLength = np.zeros([cfg.LOWER_LENGTH_NUM,cfg.LOWER_LENGTH_NUM], dtype=float)
    sum_confusion_matrix_backpack = np.zeros([cfg.BACKPACK_NUM,cfg.BACKPACK_NUM], dtype=float)
    sum_confusion_matrix_handbag = np.zeros([cfg.HANDBAG_NUM,cfg.HANDBAG_NUM], dtype=float)
    sum_confusion_matrix_umbrella = np.zeros([cfg.UMBRELLA_NUM,cfg.UMBRELLA_NUM], dtype=float)
    sum_confusion_matrix_hat = np.zeros([cfg.HAT_NUM,cfg.HAT_NUM], dtype=float)
    sum_confusion_matrix_age = np.zeros([cfg.AGE_NUM,cfg.AGE_NUM], dtype=float)
    sum_confusion_matrix_gender = np.zeros([cfg.GENDER_NUM,cfg.GENDER_NUM], dtype=float)

    # No Use Pretaing, Origin Data Split 5 Parts, 1 part for Testing, 4 parts for Training
    Index = 1
    totalTime = 0
    crossTimes = 0
    crossDataNumber = 0
    while Index < 2:
        TFRecord_Train_Name = TFRecord_Train_Path + str(Index+1) + ".tfrecords"
        TFRecord_Test_Name = TFRecord_Test_Path + str(Index+1) + ".tfrecords"
        train_Number = TFR.get_data_size(TFRecord_Train_Name)
        test_Number = TFR.get_data_size(TFRecord_Test_Name)

        # Training START
        print("*****************TRAINING START*****************")
        print("Training Data: %d" % (train_Number))
        with tf.device('/gpu:0'):
            trainStart = time.time()
            train(TFRecord_Train_Name, train_Number, Index, TFRecord_Test_Name, test_Number)
            trainEnd = time.time()
        train_Time = (trainEnd-trainStart)

        # Testing Start
        print("*****************TESTING START*****************")
        print("Training Data: %d" % (test_Number))
        with tf.device('/gpu:0'):
            testStart = time.time()
            test_confusion_matrix_upperColor, test_confusion_matrix_upperSleeve, test_confusion_matrix_lowerColor, test_confusion_matrix_lowerType,\
                test_confusion_matrix_lowerLength, test_confusion_matrix_backpack, test_confusion_matrix_handbag, test_confusion_matrix_umbrella,  \
                test_confusion_matrix_hat, test_confusion_matrix_age, test_confusion_matrix_gender \
                = evaluate(TFRecord_Test_Name, test_Number, Index, 1)
            testEnd = time.time()
        test_Time1 = (testEnd-testStart)
        totalTime += test_Time1

        # Set Confusion Matrix
        sum_confusion_matrix_upperColor = sum_confusion_matrix_upperColor + test_confusion_matrix_upperColor
        sum_confusion_matrix_upperSleeve = sum_confusion_matrix_upperSleeve + test_confusion_matrix_upperSleeve
        sum_confusion_matrix_lowerColor = sum_confusion_matrix_lowerColor + test_confusion_matrix_lowerColor
        sum_confusion_matrix_lowerType = sum_confusion_matrix_lowerType + test_confusion_matrix_lowerType
        sum_confusion_matrix_lowerLength = sum_confusion_matrix_lowerLength + test_confusion_matrix_lowerLength
        sum_confusion_matrix_backpack = sum_confusion_matrix_backpack + test_confusion_matrix_backpack
        sum_confusion_matrix_handbag = sum_confusion_matrix_handbag + test_confusion_matrix_handbag
        sum_confusion_matrix_umbrella = sum_confusion_matrix_umbrella + test_confusion_matrix_umbrella
        sum_confusion_matrix_hat = sum_confusion_matrix_hat + test_confusion_matrix_hat
        sum_confusion_matrix_age = sum_confusion_matrix_age + test_confusion_matrix_age
        sum_confusion_matrix_gender = sum_confusion_matrix_gender + test_confusion_matrix_gender

        # Show This Time Confusion Matrix
        cfg.print_confusion_matrix(test_confusion_matrix_upperColor)
        cfg.print_confusion_matrix(test_confusion_matrix_upperSleeve)
        cfg.print_confusion_matrix(test_confusion_matrix_lowerColor)
        cfg.print_confusion_matrix(test_confusion_matrix_lowerType)
        cfg.print_confusion_matrix(test_confusion_matrix_lowerLength)
        cfg.print_confusion_matrix(test_confusion_matrix_backpack)
        cfg.print_confusion_matrix(test_confusion_matrix_handbag)
        cfg.print_confusion_matrix(test_confusion_matrix_umbrella)
        cfg.print_confusion_matrix(test_confusion_matrix_hat)
        cfg.print_confusion_matrix(test_confusion_matrix_age)
        cfg.print_confusion_matrix(test_confusion_matrix_gender)

        print("*****************Look Value*****************")
        look_model_weight(1, Index)

        print("*****************FineTuned*****************")
        # Finetune Class 1: UpperSleeve, 3: LowerType, 4: LowerLength, 5: Backpack, 6: Handbag, 7: Umbrella, 8: Hat
        # Dont Need Finetune Class 0: UpperColor, 2: LowerColor
        with tf.device('/gpu:0'):
            finetuneStart = time.time()
            for finetune_index in range(11):
                if finetune_index == 0 or finetune_index == 2:
                    continue
                if finetune_index == 1:
                    FINETUNE_NUM = cfg.UPPER_SLEEVE_NUM
                    finetune_layer = "UpperSleeve_FullConnect"
                elif finetune_index == 3:
                    FINETUNE_NUM = cfg.LOWER_TYPE_NUM
                    finetune_layer = "LowerType_FullConnect"
                elif finetune_index == 4:
                    FINETUNE_NUM = cfg.LOWER_LENGTH_NUM
                    finetune_layer = "LowerLength_FullConnect"
                elif finetune_index == 5:
                    FINETUNE_NUM = cfg.BACKPACK_NUM
                    finetune_layer = "Backpack_FullConnect"
                elif finetune_index == 6:
                    FINETUNE_NUM = cfg.HANDBAG_NUM
                    finetune_layer = "Handbag_FullConnect"
                elif finetune_index == 7:
                    FINETUNE_NUM = cfg.UMBRELLA_NUM
                    finetune_layer = "Umbrella_FullConnect"
                elif finetune_index == 8:
                    FINETUNE_NUM = cfg.HAT_NUM
                    finetune_layer = "Hat_FullConnect"
                elif finetune_index == 9:
                    FINETUNE_NUM = cfg.AGE_NUM
                    finetune_layer = "Age_FullConnect"
                elif finetune_index == 10:
                    FINETUNE_NUM = cfg.GENDER_NUM
                    finetune_layer = "Gender_FullConnect"

                finetune(TFRecord_Train_Name, train_Number, Index, FINETUNE_NUM, finetune_index, finetune_layer)
            #finetune(TFRecord_Train_Name, train_Number, Index, TFRecord_Test_Name, test_Number)
            finetuneEnd = time.time()
            #return
        finetune_Time = (finetuneEnd - finetuneStart)

        print("*****************Look Value*****************")
        #look_model_weight(2, Index)

        # Testing Start
        print("*****************TESTING START*****************")
        print("Training Data: %d" % (test_Number))
        with tf.device('/gpu:0'):
            testStart = time.time()
            test_confusion_matrix_upperColor, test_confusion_matrix_upperSleeve, test_confusion_matrix_lowerColor, test_confusion_matrix_lowerType,\
                test_confusion_matrix_lowerLength, test_confusion_matrix_backpack, test_confusion_matrix_handbag, test_confusion_matrix_umbrella,  \
                test_confusion_matrix_hat, test_confusion_matrix_age, test_confusion_matrix_gender \
                = evaluate(TFRecord_Test_Name, test_Number, Index, 2)
            testEnd = time.time()
        test_Time2 = (testEnd-testStart)

        # Show This Time Confusion Matrix
        cfg.print_confusion_matrix(test_confusion_matrix_upperColor)
        cfg.print_confusion_matrix(test_confusion_matrix_upperSleeve)
        cfg.print_confusion_matrix(test_confusion_matrix_lowerColor)
        cfg.print_confusion_matrix(test_confusion_matrix_lowerType)
        cfg.print_confusion_matrix(test_confusion_matrix_lowerLength)
        cfg.print_confusion_matrix(test_confusion_matrix_backpack)
        cfg.print_confusion_matrix(test_confusion_matrix_handbag)
        cfg.print_confusion_matrix(test_confusion_matrix_umbrella)
        cfg.print_confusion_matrix(test_confusion_matrix_hat)
        cfg.print_confusion_matrix(test_confusion_matrix_age)
        cfg.print_confusion_matrix(test_confusion_matrix_gender)

        # Configuration
        print("Train Consume Time", train_Time)
        print("Before Fine-tune, Average Test Consume Time", test_Time1/test_Number)
        print("Fine-tune Consume Time", finetune_Time)
        print("After Fine-tune, Average Test Consume Time", test_Time2/test_Number)

        # Get This Time Each Class Number
        crossDataNumber += test_Number
        crossTimes += 1
        Index += 1
    
    # Show Final Confusion Matrix
    print("*****************Confusion Matrix*****************")
    print("UpperColor")
    cfg.print_confusion_matrix(sum_confusion_matrix_upperColor)
    print("UpperSleeve")
    cfg.print_confusion_matrix(sum_confusion_matrix_upperSleeve)
    print("LowerColor")
    cfg.print_confusion_matrix(sum_confusion_matrix_lowerColor)
    print("LowerType")
    cfg.print_confusion_matrix(sum_confusion_matrix_lowerType)
    print("LowerLength")
    cfg.print_confusion_matrix(sum_confusion_matrix_lowerLength)
    print("Backpack")
    cfg.print_confusion_matrix(sum_confusion_matrix_backpack)
    print("Handbag")
    cfg.print_confusion_matrix(sum_confusion_matrix_handbag)
    print("Umbrella")
    cfg.print_confusion_matrix(sum_confusion_matrix_umbrella)
    print("Hat")
    cfg.print_confusion_matrix(sum_confusion_matrix_hat)
    print("Age")
    cfg.print_confusion_matrix(sum_confusion_matrix_age)
    print("Gender")
    cfg.print_confusion_matrix(sum_confusion_matrix_gender)

    print("Total Test Number %d" % (crossDataNumber))
    print("Average Time %.4f (s)" % (totalTime/crossDataNumber))

    # Set Plot Axes
    plotX = []
    plotNum = int(cfg.TRAINING_STEPS/5)
    for ii in range(plotNum):
        plotX.append(ii+1)
    print("Plot Accuracy and Loss Figure")
    
    # Draw Loss Curve
    cfg.draw_learning_curve(plotX, UPPERCOLOR_TRAIN_LOSS, UPPERCOLOR_VAL_LOSS, MODEL_SAVE_PATH+MODEL_NAME+"/Image/UpperColor_Loss.png")
    cfg.draw_learning_curve(plotX, UPPERSLEEVE_TRAIN_LOSS, UPPERSLEEVE_VAL_LOSS, MODEL_SAVE_PATH+MODEL_NAME+"/Image/UpperSleeve_Loss.png")
    cfg.draw_learning_curve(plotX, LOWERCOLOR_TRAIN_LOSS, LOWERCOLOR_VAL_LOSS, MODEL_SAVE_PATH+MODEL_NAME+"/Image/LowerColor_Loss.png")
    cfg.draw_learning_curve(plotX, LOWERTYPE_TRAIN_LOSS, LOWERTYPE_VAL_LOSS, MODEL_SAVE_PATH+MODEL_NAME+"/Image/LowerType_Loss.png")
    cfg.draw_learning_curve(plotX, LOWERLENGTH_TRAIN_LOSS, LOWERLENGTH_VAL_LOSS, MODEL_SAVE_PATH+MODEL_NAME+"/Image/LowerLength_Loss.png")
    cfg.draw_learning_curve(plotX, BACKPACK_TRAIN_LOSS, BACKPACK_VAL_LOSS, MODEL_SAVE_PATH+MODEL_NAME+"/Image/Backpack_Loss.png")
    cfg.draw_learning_curve(plotX, HANDBAG_TRAIN_LOSS, HANDBAG_VAL_LOSS, MODEL_SAVE_PATH+MODEL_NAME+"/Image/Handbag_Loss.png")
    cfg.draw_learning_curve(plotX, UMBRELLA_TRAIN_LOSS, UMBRELLA_VAL_LOSS, MODEL_SAVE_PATH+MODEL_NAME+"/Image/Umbrella_Loss.png")
    cfg.draw_learning_curve(plotX, HAT_TRAIN_LOSS, HAT_VAL_LOSS, MODEL_SAVE_PATH+MODEL_NAME+"/Image/Hat_Loss.png")
    cfg.draw_learning_curve(plotX, AGE_TRAIN_LOSS, AGE_VAL_LOSS, MODEL_SAVE_PATH+MODEL_NAME+"/Image/Age_Loss.png")
    cfg.draw_learning_curve(plotX, GENDER_TRAIN_LOSS, GENDER_VAL_LOSS, MODEL_SAVE_PATH+MODEL_NAME+"/Image/Gender_Loss.png")
    
    # Draw Acc Curve
    cfg.draw_learning_curve(plotX, UPPERCOLOR_TRAIN_ACC, UPPERCOLOR_VAL_ACC, MODEL_SAVE_PATH+MODEL_NAME+"/Image/UpperColor_Acc.png")
    cfg.draw_learning_curve(plotX, UPPERSLEEVE_TRAIN_ACC, UPPERSLEEVE_VAL_ACC, MODEL_SAVE_PATH+MODEL_NAME+"/Image/UpperSleeve_Acc.png")
    cfg.draw_learning_curve(plotX, LOWERCOLOR_TRAIN_ACC, LOWERCOLOR_VAL_ACC, MODEL_SAVE_PATH+MODEL_NAME+"/Image/LowerColor_Acc.png")
    cfg.draw_learning_curve(plotX, LOWERTYPE_TRAIN_ACC, LOWERTYPE_VAL_ACC, MODEL_SAVE_PATH+MODEL_NAME+"/Image/LowerType_Acc.png")
    cfg.draw_learning_curve(plotX, LOWERLENGTH_TRAIN_ACC, LOWERLENGTH_VAL_ACC, MODEL_SAVE_PATH+MODEL_NAME+"/Image/LowerLength_Acc.png")
    cfg.draw_learning_curve(plotX, BACKPACK_TRAIN_ACC, BACKPACK_VAL_ACC, MODEL_SAVE_PATH+MODEL_NAME+"/Image/Backpack_Acc.png")
    cfg.draw_learning_curve(plotX, HANDBAG_TRAIN_ACC, HANDBAG_VAL_ACC, MODEL_SAVE_PATH+MODEL_NAME+"/Image/Handbag_Acc.png")
    cfg.draw_learning_curve(plotX, UMBRELLA_TRAIN_ACC, UMBRELLA_VAL_ACC, MODEL_SAVE_PATH+MODEL_NAME+"/Image/Umbrella_Acc.png")
    cfg.draw_learning_curve(plotX, HAT_TRAIN_ACC, HAT_VAL_ACC, MODEL_SAVE_PATH+MODEL_NAME+"/Image/Hat_Acc.png")
    cfg.draw_learning_curve(plotX, AGE_TRAIN_ACC, AGE_VAL_ACC, MODEL_SAVE_PATH+MODEL_NAME+"/Image/Age_Acc.png")
    cfg.draw_learning_curve(plotX, GENDER_TRAIN_ACC, GENDER_VAL_ACC, MODEL_SAVE_PATH+MODEL_NAME+"/Image/Gender_Acc.png")

if __name__ == '__main__':
    main(sys.argv[1:])

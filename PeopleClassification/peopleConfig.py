## Parameters
import numpy as np
import cv2
import random

import peopleModel as model

# Image Size
IMG_WIDTH = 200
IMG_HEIGHT = 400
NUM_CHANNELS = 3

# People Attribute Class Number
UPPER_COLOR_NUM = 11
UPPER_SLEEVE_NUM = 2
LOWER_COLOR_NUM = 11
LOWER_TYPE_NUM = 2
LOWER_LENGTH_NUM = 2
BACKPACK_NUM = 2
HANDBAG_NUM = 2
UMBRELLA_NUM = 2
HAT_NUM = 2
AGE_NUM = 3
GENDER_NUM = 2

# Training And Fine-tuning Parameters
BATCH_SIZE = 30
LEARNING_RATE_BASE = 0.0001
FINETUNE_LEARNING_RATE_BASE = 0.0001
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 60
FINETUNE_STEPS = 50
FINETUNE_Epoch = 30

# Print Confusion Matrix
def print_confusion_matrix(confusion_matrix):
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
    print("\n")

def DataAugmentation(batch_x, train):
    # Data Augmentation
    ranID = np.random.permutation(batch_x.shape[0])
    augmetation_xs = np.zeros((batch_x.shape[0], model.INPUT_WIDTH*model.INPUT_HEIGHT*NUM_CHANNELS), dtype=float)
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
        augmetation_xs[index] = np.reshape(imgBBOG, (-1, model.INPUT_WIDTH*model.INPUT_HEIGHT*NUM_CHANNELS))
        index += 1
    return augmetation_xs

# Let One Class Data Balance
def getBalanceID(Label, ClassNumber, Index):
    BatchID = np.zeros(BATCH_SIZE, dtype=int)
    ClassMax = np.zeros(ClassNumber, dtype=int)
    # Set Each Class Max Number
    quo = BATCH_SIZE / ClassNumber
    rem = BATCH_SIZE % ClassNumber
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
    return Index, BatchID

# Draw Curve And Save
def draw_learning_curve(plotX, train, val, plotNum, save_path):
    plt.plot(plotX, train, plotX, val)
    plt.text(plotX[plotNum-1], train[plotNum-1], "Train"+str(train[plotNum-1]))
    plt.text(plotX[plotNum-1], val[plotNum-1], "Val"+str(val[plotNum-1]))
    plt.savefig(save_path)
    plt.close()

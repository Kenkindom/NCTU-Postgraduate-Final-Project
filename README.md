#Data Preprocess, People Attribute Training, Testing Step
1.  Convert Mat File Or PETA Datast
    Data Process Folder
    Using Relative Matlab Code Convert File
    (1) ConvertLabel.m, Conver All Pedestrian Mat File, Every File Have Image, Label, Mask, ImageMask, In Code:
        DATA_NAME Is Ouput Folder Name
	      MAT_PATH Is Pedestrian Mat File Folder Location
	      Time Is "morning" Or "night", Attention, When Time Is Night, Skip Color Judgement

    (2) ConvertPETALabel.m, Convert PETA Dataset Subfolder, Every File Have Image, Label, In Code:
        FolderName Is PETA's Subset Folder

2.  Convert Data To TFRecord File
    PeopleClassification Folder
    Use TFRecord_write.py Convert First Step(1.) Output Data To TFRecord Data Format
    Instruction: python3 TFRecord_write.py
    Use First Step Output Folder Name, Set Program's DATA_NAME And PETA_NAME, In Main Function Call get_data_list() Pass Parameter DATA_PATH Or PETA_PATH
    In main function, TFDir Is Output TFRecord Data Folder
    Be Attention, Because PETA Dataset Don't Have Mask Image, In Main Function Must Comment Or Delete Mask Relatived Code
    
    End Of Program Output MinMaxScaler Converter When Training And Testing, Fine-tuning Will Be Used

3.  People Attribute Training And Testing
    PeopleClassification Folder
    Use peopleClassificationAll.py For Training And Testing
    Instruction: python3 peopleClassificationAll.py learning_rate decay_rate MIN_KM
                 python3 peopleClassificationAll.py 0.0001 0.99 2
                 python3 peopleClassificationAll.py (Will Use Default Parameter)
    
    Can Adjustment Parameter
    INPUT_WIDTH、INPUT_HEIGHT、NUM_CHANNELS Is Input Model Size
    IMG_WIDTH、IMG_HEIGHT、NUM_CHANNELS Is Input Image Size
    BATCH_SIZE Is Batch Size
    LEARNING_RATE_BASE Is Training Default Learning Rate
    FINETUNE_LEARNING_RATE_BASE Is Fine-tuned Default Learning Rate
    LEARNING_RATE_DECAY Is Training Default Decay Rate
    TRAINING_STEPS Is Training Epoch
    TRANSFER Is Whether Continue Training 
    MIN_KM Is Loss Function Will Use Parameter

    Before Training Must Set
    MODEL_NAME Is Save Model Name, Will Save In './Model' Folder
    TFDATA Is TFRecord Name From Second Step

    In main function Will Do train(training), evaluate(Testing), finetune(fine-tuning), evaluate(Testing) Function
    In This Function Use read_and_decode function. If In TFRecord Have Saving Mask, Can Read "mask_string". Otherwise Make Error
    In train function Will Output Loss Curve Image And Accuracy Curve Image In "./Model/Image" Folder 
    In evaluate function Will Output Txt File For Each Attribute, And In Txt File Will Record Confidence For Each Testing Data, And Return Confusion Matrix
    When Doing Fine-tune Will Use Loop Set Attribute Parameter, Number Of Class, Name Of Fine-tune, And Call finetune function
    
    End of main function Draw learning curve

#MaskRCNN偵測
    Mask R-CNN Detect And Predict Attribute
    程式執行前須先將
        人工標計的屬性label檔案放置指定的人工label檔案之位置(mat_Dir)
        所有原始影像放置指定的輸入影像位置(filename)

    執行demo.py，會將指定的輸入資料夾中所有影像作行人偵測，並且對應到指定的label資料夾去比對Bounding Box IOU>=0.5，保留期原本的Label資訊，將所有偵測到的行人影像、原Label串起來後，放入指定的模型中進行預測，最後輸出Confusion Matrix

    MODEL_SAVE_PATH 為指定的模行路徑
    MODEL_NAME 為指定的模型名稱
    TFDIR + TFDATA 為指定的MinMaxScaler轉換器之路徑
    MMS_PATH = TFDIR + TFDATA + MinMaxScaler轉換器之名稱
    Time 為白天(morning)或晚上(night)

    程式中
    filename 為指定的輸入影像位置
    mat_Dir 為指定的人工label檔案之位置
    當對應的IOU >= 0.5時，會將偵測到的行人影像、Mask影像、Label存起來至指定的輸出位置，因無設定輸出變數須，因此必須要自行修改

    須注意當Time = night時，因label檔案中顏色標記幾乎皆為Unknown，因此必須跳過顏色的比對

    若要利用偵測到的行人進行轉成TFrecord檔，則在輸出的資料夾中工執行TFRecord_write.py
    並將輸出後的TFData中資料夾移至上面第3點的TFDir，並設定好TFDATA即可進行訓練


#將人工標記之Label轉成YOLO或MaskRCNN可讀之格式
1.  For YOLO
    (1) 將superpixel後的Label利用MatToXml.m轉檔
        matFolder 為指定的輸入label資料夾
        xmlFolder 為指定的輸出xml資料夾
        jpgFolder 為指定的輸出jpg資料夾 #可不需要
        txtName 為指定的txt檔案名稱，內容為所有的影像名稱 #可不需要
    
    (2) 將已標記行人屬性的Label利用LabelToXml.m轉檔
        matFolder 為指定的輸入Label資料夾
        xmlFolder 為指定的輸出xml資料夾

    兩程式皆輸出所有影像的xml檔案至指定的xml資料夾
    xml檔案中包含影像名稱、影像長寬、影像中所有行人的bbox位置長寬

2.  For Mask R-CNN
    (1) 將superpixel後的Label利用MatToJson.m轉檔
        matFolder 為指定的輸入label資料夾
        jsonTrainFile 為指定的訓練用Json檔
        jsonValFile 為指定的驗證用Json檔

        會利用annSegmentation.m計算出行人輪廓之座標，若連續三點成一線，則只保留兩端點座標

    (2) 將已標記行人屬性的Label利用LabelToJson.m轉檔
        matFolder 為指定的輸入label資料夾
        jsonFolder 為指定的輸出Json檔案之資料夾，會輸出以下四個檔案
        jsonTrainFile 為指定的訓練用Json檔
        jsonValFile 為指定的驗證用Json檔
        jsonMiniFile 為指定的迷你用Json檔
        jsonVal35kFile 為指定的Json檔

        會利用LabelGetSeg.m計算出行人輪廓之座標，若連續三點成一線，則只保留兩端點座標

    兩程式皆輸出所json檔案至指定的資料夾
    Json檔案中包含{info, license, image, annotation, category}
    info 為資料集的敘述，包含{description: 自行賦與, contributir: 自行賦與, data_create: 自行賦與}
    license 為該資料集license敘述，包含{id: 自行賦與, name: 自行賦與}
    image 為所有原始影像的敘述，包含{id: 自行賦與, width: 原始影像寬, height: 原始影像高, file_name: 原始影像名稱, license: 對應到上述的license id}
    annotation 為影像中行人的相關資訊，包含{id: 自行賦與, image_id: 對應到image中之id, category_id, 對應到category中之id, segmentation: 為行人輪廓之座標, area: 所佔之面積, bbox: Bounding Box資訊, iscrowd: 自行賦與}
    categories 為影像中使用之類別，包含{id: 自行賦與, name: 類別名稱, supercategory: 自行賦與}

#評估
    Evaluation
    (1) PRcurve.m利用Bounding Box IOU >= 0.5進行評估
        將MaskR-CNN偵測到的行人之Bounding Box資訊存至txt檔案中
        txtFile 為指定的Bounding Box txt檔

    (2) PRcuve_Pixel.m利用Pixel IOU >= 0.5進行評估
        將MaskR-CNN偵測到的行人之Mask資訊存至txt檔案中
        txtFile 為指定的Bounding Box txt檔

    最後皆會輸出PRcurve與AP

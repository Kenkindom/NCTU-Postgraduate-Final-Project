#資料前處理、行人屬性訓練、測試步驟
1.  將Mat檔或PETA資料夾進行轉檔
    Data Process
    利用相關的Matlab程式進行轉檔
    
    (1).ConvertLabel.m將所有行人mat檔案，轉出各自的行人image、label、mask、maskImage，程式中
        DATA_NAME為指定的輸出資料夾名稱
	MAT_PATH為指定的行人Mat檔資料夾之位置
	程式中必須利用Mat檔案的Time來判斷使用白天或晚上的資料，須注意在進行晚上資料的轉檔時，必須跳過顏色判斷的部分

    (2).ConvertPETALabel.m將PETA資料集中的指定資料夾之行人影像與Label，轉出各自的行人image、label，程式中
        FolderName為指定的PETA子資料夾

2.  將資料轉為TFRecord檔案
    PeopleClassification
    利用TFRecord_write.py將第1點的資料轉出TFRecord資料格式
    指令 python3 TFRecord_write.py
    利用第1點指定的輸出資料夾名稱，設定程式中的DATA_NAME或PETA_NAME，並在main function中呼叫get_data_list時將DATA_PATH或PETA_PATH做為參數傳送
    在main function中TFDir為設定的輸出TFRecord檔案的資料夾
    須注意，因為PETA沒有mask影像，因此在main function的部分需將與mask有關係的部分註解或刪除
    
    程式尾端會另外輸出MinMaxScaler的轉換器，會在訓練、測試、Fine-tune時使用

3.  行人屬性訓練、測試
    PeopleClassification
    使用peopleClassificationAll.py進行訓練、測試
    指令 python3 peopleClassificationAll.py learning_rate decay_rate MIN_KM
    參數部分若不輸入則會以預設的為主
    
    可調之預設參數
    INPUT_WIDTH、INPUT_HEIGHT、NUM_CHANNELS 為進入模型時的輸入大小
    IMG_WIDTH、IMG_HEIGHT、NUM_CHANNELS 為指定的影像輸入大小
    BATCH_SIZE 為指定的BATCH大小
    LEARNING_RATE_BASE 為訓練時預設的Learning Rate
    FINETUNE_LEARNING_RATE_BASE 為Fine-tune時預設的Learning Rate
    LEARNING_RATE_DECAY 為訓練時預設的Decay Rate
    TRAINING_STEPS 為訓練次數
    TRANSFER 為是否要接續訓練
    MIN_KM 為MIN_KM

    訓練前須先指定
    MODEL_NAME 為在./Model資料夾中，儲存模型的名稱
    TFDATA 為利用第2點得到的指定的TFDir設定要取的TFRecord Data名稱

    在main function中會依序執行train(訓練)、evaluate(測試)、finetune(fine-tune)、evaluate(測試)
    須注意在這些function中，皆有使用read_and_decode function，在TFRecord檔中有存mask時，才能讀取mask_string，否則會報錯
    在train function中，會在./Model/Image輸出Loss Curve、Acc Curve
    在evaluate function中，會./Model/Txt輸出所有屬性的每筆測試資料之confidence，並回傳所有屬性的Confusion Matrix
    在做Fine-tune時，會利用迴圈對指定的屬性，設定類別數量、Fine-tune的名稱，並呼叫finetune function
    
    在main function尾端為畫learning curve的部分

#MaskRCNN偵測
Mask R-CNN Detect And Predict Attribute
1.  程式執行前須先將
        人工標計的屬性label檔案放置指定的人工label檔案之位置(mat_Dir)
        所有原始影像放置指定的輸入影像位置(filename)

2.  執行demo.py，會將指定的輸入資料夾中所有影像作行人偵測，並且對應到指定的label資料夾去比對Bounding Box IOU>=0.5，保留期原本的Label資訊，將所有偵測到的行人影像、原Label串起來後，放入指定的模型中進行預測，最後輸出Confusion Matrix

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

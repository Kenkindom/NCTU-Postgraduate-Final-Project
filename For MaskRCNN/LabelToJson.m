clear all;
clc;

%% Read label folder
matFolder = 'C:\Users\Lab031\Desktop\CHT_2017\Label\';
jsonFolder = 'C:\Users\Lab031\Desktop\CHT_2017\LabelToJson\';
Dir = dir([matFolder, '*.mat']);
matNumber = size(Dir, 1);
fprintf("Have %d Labels\n", matNumber);

%% Write Train & Val json Str
info = '"info": {"description": "CHT 2017 dataset", "contributor": "CHT", "date_create": "2018-03-01"}, ';
license = '"license": [{"id": 1, "name": "CHT License"}], ';
image = '"images": [';
annotation = '"annotations": [';
category = '"categories": [{"id": 1, "name": "person", "supercategory": "person"}]';

imageVal = image;
imageTrain = image;
imageMini = image;
imageVal35k = image;
imageId = 0;
annotationTrain = annotation;
annotationVal = annotation;
annotationMini = annotation;
annotationVal35k = annotation;
oldName = [];
smallSegNumber = 0;

%% Get Label Information
for matIndex = 1:matNumber
    MAT = load([matFolder Dir(matIndex).name]);
    matName = Dir(matIndex).name;
    fprintf("%5d-%d %s\n", matIndex, matNumber, matName);
    matLen = size(matName, 2);
    
    %% Get Name And Label Index
    jpgLen = 0;
    labelIndex = 0;
    jpgName = [];
    for index = matLen-4:-1:1
        if matName(index) == '_'
            labelIndex = str2num(matName(index+1: matLen-4));
            jpgLen = index - 1;
            break;
        end
    end
    jpgName = [matName(1:jpgLen) '.jpg'];
    
    %% Get Image
    if ~strcmp(oldName, jpgName)
        imageId = imageId + 1;
        if mod(imageId, 10) >= 0 && mod(imageId, 10) <= 1
            imageVal = [imageVal '{"id": ' num2str(imageId) ', "width": 1280, "height": 720, "file_name": "' jpgName '", "license": 1}'];
        elseif mod(imageId, 10) >=2 && mod(imageId, 10) <= 7
            imageTrain = [imageTrain '{"id": ' num2str(imageId) ', "width": 1280, "height": 720, "file_name": "' jpgName '", "license": 1}'];
        elseif mod(imageId, 10) == 8  
            imageMini = [imageMini '{"id": ' num2str(imageId) ', "width": 1280, "height": 720, "file_name": "' jpgName '", "license": 1}'];
        elseif mod(imageId, 10) == 9  
            imageVal35k = [imageVal35k '{"id": ' num2str(imageId) ', "width": 1280, "height": 720, "file_name": "' jpgName '", "license": 1}'];
        end
    end
    
    %% Get Annotation
    mask = imresize(MAT.label.mask, [MAT.label.height, MAT.label.width]);
    mask(find(mask >= 0.5)) = 1;
    mask(find(mask < 0.5)) = 0;
    seg = LabelGetSeg(mask);
    area = size(find(mask == 1), 1);
    [r, w] = find(mask == 1);
    xmin = min(min(w)) + MAT.label.x;
    xmax = max(max(w)) + MAT.label.x;
    ymin = min(min(r)) + MAT.label.y;
    ymax = max(max(r)) + MAT.label.y;
    
    segmentation = '[[';
    haveSeg = 0;
    for outIndex = 1:size(seg, 2)
        % If This Cell Point Number Fewer Than 4 Point, Produce Point By
        % Self
        if size(seg{outIndex}, 1) <= 4
            smallSegNumber = smallSegNumber + 1;
            if size(seg{outIndex}, 1) == 1
                segmentation = [segmentation num2str(seg{outIndex}(1,1)+MAT.label.x) ',' num2str(seg{outIndex}(1,2)+MAT.label.y) ','];
                segmentation = [segmentation num2str(seg{outIndex}(1,1)+MAT.label.x) ',' num2str(seg{outIndex}(1,2)+MAT.label.y) ','];
                segmentation = [segmentation num2str(seg{outIndex}(1,1)+MAT.label.x) ',' num2str(seg{outIndex}(1,2)+MAT.label.y) ','];
                segmentation = [segmentation num2str(seg{outIndex}(1,1)+MAT.label.x) ',' num2str(seg{outIndex}(1,2)+MAT.label.y) ','];
                segmentation = [segmentation num2str(seg{outIndex}(1,1)+MAT.label.x) ',' num2str(seg{outIndex}(1,2)+MAT.label.y) ']'];
            elseif size(seg{outIndex}, 1) == 2
                segmentation = [segmentation num2str(seg{outIndex}(1,1)+MAT.label.x) ',' num2str(seg{outIndex}(1,2)+MAT.label.y) ','];
                segmentation = [segmentation num2str(seg{outIndex}(1,1)+MAT.label.x) ',' num2str(seg{outIndex}(1,2)+MAT.label.y) ','];
                segmentation = [segmentation num2str(seg{outIndex}(1,1)+MAT.label.x) ',' num2str(seg{outIndex}(1,2)+MAT.label.y) ','];
                segmentation = [segmentation num2str(seg{outIndex}(2,1)+MAT.label.x) ',' num2str(seg{outIndex}(2,2)+MAT.label.y) ','];
                segmentation = [segmentation num2str(seg{outIndex}(2,1)+MAT.label.x) ',' num2str(seg{outIndex}(2,2)+MAT.label.y) ']'];
            elseif size(seg{outIndex}, 1) == 3
                segmentation = [segmentation num2str(seg{outIndex}(1,1)+MAT.label.x) ',' num2str(seg{outIndex}(1,2)+MAT.label.y) ','];
                segmentation = [segmentation num2str((seg{outIndex}(1,1)+seg{outIndex}(2,1))/2+MAT.label.x) ',' num2str((seg{outIndex}(1,2)+seg{outIndex}(2,2))/2+MAT.label.y) ','];
                segmentation = [segmentation num2str(seg{outIndex}(2,1)+MAT.label.x) ',' num2str(seg{outIndex}(2,2)+MAT.label.y) ','];
                segmentation = [segmentation num2str((seg{outIndex}(2,1)+seg{outIndex}(3,1))/2+MAT.label.x) ',' num2str((seg{outIndex}(2,2)+seg{outIndex}(3,2))/2+MAT.label.y) ','];
                segmentation = [segmentation num2str(seg{outIndex}(3,1)+MAT.label.x) ',' num2str(seg{outIndex}(3,2)+MAT.label.y) ']'];
            else
                segmentation = [segmentation num2str(seg{outIndex}(1,1)+MAT.label.x) ',' num2str(seg{outIndex}(1,2)+MAT.label.y) ','];
                segmentation = [segmentation num2str(seg{outIndex}(2,1)+MAT.label.x) ',' num2str(seg{outIndex}(2,2)+MAT.label.y) ','];
                segmentation = [segmentation num2str(seg{outIndex}(3,1)+MAT.label.x) ',' num2str(seg{outIndex}(3,2)+MAT.label.y) ','];
                segmentation = [segmentation num2str(seg{outIndex}(4,1)+MAT.label.x) ',' num2str(seg{outIndex}(4,2)+MAT.label.y) ','];
                segmentation = [segmentation num2str((seg{outIndex}(1,1)+seg{outIndex}(4,1))/2+MAT.label.x) ',' num2str((seg{outIndex}(1,2)+seg{outIndex}(4,2))/2+MAT.label.y) ']'];
            end
            if outIndex ~= size(seg, 2)
                segmentation = [segmentation ', ['];
            end
            continue;
        end
        for pointIndex = 1:size(seg{outIndex}, 1)
            segmentation = [segmentation num2str(seg{outIndex}(pointIndex,1)+MAT.label.x) ',' num2str(seg{outIndex}(pointIndex,2)+MAT.label.y)];
            if pointIndex ~= size(seg{outIndex}, 1)
                segmentation = [segmentation ','];
            end
        end
        segmentation = [segmentation ']'];
        if outIndex ~= size(seg, 2)
            segmentation = [segmentation ', ['];
        end
    end
    segmentation = [segmentation ']'];
    
    if mod(imageId, 10) >= 0 && mod(imageId, 10) <= 1
        annotationVal = [annotationVal '{"id": ' num2str(matIndex) ', "image_id": ' num2str(imageId) ', "category_id": 1, "segmentation": ' segmentation ', "area": ' num2str(area) ', "bbox": [' num2str(xmin) '.0, ' num2str(ymin) '.0, ' num2str(xmax-xmin) '.0, ' num2str(ymax-ymin) '.0], "iscrowd": 0}'];
    elseif mod(imageId, 10) >=2 && mod(imageId, 10) <= 7
        annotationTrain = [annotationTrain '{"id": ' num2str(matIndex) ', "image_id": ' num2str(imageId) ', "category_id": 1, "segmentation": ' segmentation ', "area": ' num2str(area) ', "bbox": [' num2str(xmin) '.0, ' num2str(ymin) '.0, ' num2str(xmax-xmin) '.0, ' num2str(ymax-ymin) '.0], "iscrowd": 0}'];
    elseif mod(imageId, 10) ==8  
        annotationMini = [annotationMini '{"id": ' num2str(matIndex) ', "image_id": ' num2str(imageId) ', "category_id": 1, "segmentation": ' segmentation ', "area": ' num2str(area) ', "bbox": [' num2str(xmin) '.0, ' num2str(ymin) '.0, ' num2str(xmax-xmin) '.0, ' num2str(ymax-ymin) '.0], "iscrowd": 0}'];
    elseif mod(imageId, 10) ==9  
        annotationVal35k = [annotationVal35k '{"id": ' num2str(matIndex) ', "image_id": ' num2str(imageId) ', "category_id": 1, "segmentation": ' segmentation ', "area": ' num2str(area) ', "bbox": [' num2str(xmin) '.0, ' num2str(ymin) '.0, ' num2str(xmax-xmin) '.0, ' num2str(ymax-ymin) '.0], "iscrowd": 0}'];
    end
    
    if matIndex == matNumber
        if ~strcmp(oldName, jpgName)
            if imageVal(1, size(imageVal, 2)-1) == ','
                imageVal = imageVal(1, 1:size(imageVal, 2)-2);
            end
            if imageTrain(1, size(imageTrain, 2)-1) == ','
                imageTrain = imageTrain(1, 1:size(imageTrain, 2)-2);
            end
            if imageMini(1, size(imageMini, 2)-1) == ','
                imageMini = imageMini(1, 1:size(imageMini, 2)-2);
            end
            if imageVal35k(1, size(imageVal35k, 2)-1) == ','
                imageVal35k = imageVal35k(1, 1:size(imageVal35k, 2)-2);
            end
            imageVal = [imageVal '], '];
            imageTrain = [imageTrain '], '];
            imageMini = [imageMini '], '];
            imageVal35k = [imageVal35k '], '];
        end
        if annotationTrain(1, size(annotationTrain, 2)-1) == ','
            annotationTrain = annotationTrain(1, 1:size(annotationTrain, 2)-2);
        end
        if annotationVal(1, size(annotationVal, 2)-1) == ','
            annotationVal = annotationVal(1, 1:size(annotationVal, 2)-2);
        end
        if annotationMini(1, size(annotationMini, 2)-1) == ','
            annotationMini = annotationMini(1, 1:size(annotationMini, 2)-2);
        end
        if annotationVal35k(1, size(annotationVal35k, 2)-1) == ','
            annotationVal35k = annotationVal35k(1, 1:size(annotationVal35k, 2)-2);
        end
        annotationTrain = [annotationTrain '], '];
        annotationVal = [annotationVal '], '];
        annotationMini = [annotationMini '], '];
        annotationVal35k = [annotationVal35k '], '];
    else
        if ~strcmp(oldName, jpgName)
            if mod(imageId, 10) >= 0 && mod(imageId, 10) <= 1
                imageVal = [imageVal ', '];
            elseif mod(imageId, 10) >=2 && mod(imageId, 10) <= 7
                imageTrain = [imageTrain ', '];
            elseif mod(imageId, 10) ==8  
                imageMini = [imageMini ', '];
            elseif mod(imageId, 10) ==9  
                imageVal35k = [imageVal35k ', '];
            end
        end
        
        if mod(imageId, 10) >= 0 && mod(imageId, 10) <= 1
            annotationVal = [annotationVal ', '];
        elseif mod(imageId, 10) >=2 && mod(imageId, 10) <= 7
            annotationTrain = [annotationTrain ', '];
        elseif mod(imageId, 10) ==8  
            annotationMini = [annotationMini ', '];
        elseif mod(imageId, 10) ==9  
            annotationVal35k = [annotationVal35k ', '];
        end
    end
    oldName = jpgName;
end

%% Write Train & Val json File

jsonTrainFile = 'C:\Users\Lab031\Desktop\CHT_2017\LabelToJson\instances_train2014.json';
jsonValFile = 'C:\Users\Lab031\Desktop\CHT_2017\LabelToJson\instances_val2014.json';
jsonMiniFile = 'C:\Users\Lab031\Desktop\CHT_2017\LabelToJson\instances_minival2014.json';
jsonVal35kFile = 'C:\Users\Lab031\Desktop\CHT_2017\LabelToJson\instances_valminusminival2014.json';
jsonTrainFid = fopen(jsonTrainFile, 'w');
if jsonTrainFid == -1
    error('Cannt open Train json File');
end
jsonValFid = fopen(jsonValFile, 'w');
if jsonValFid == -1
    error('Cannt open Val json File');
end
jsonMiniFid = fopen(jsonMiniFile, 'w');
if jsonMiniFid == -1
    error('Cannt open Minival json File');
end
jsonVal35kFid = fopen(jsonVal35kFile, 'w');
if jsonVal35kFid == -1
    error('Cannt open Val35k json File');
end

fprintf(jsonTrainFid, "{%s %s %s %s %s}", info, imageTrain, license, annotationTrain, category);
fprintf(jsonValFid, "{%s %s %s %s %s}", info, imageVal, license, annotationVal, category);
fprintf(jsonMiniFid, "{%s %s %s %s %s}", info, imageMini, license, annotationMini, category);
fprintf(jsonVal35kFid, "{%s %s %s %s %s}", info, imageVal35k, license, annotationVal35k, category);

fclose(jsonTrainFid);
fclose(jsonValFid);
fclose(jsonMiniFid);
fclose(jsonVal35kFid);
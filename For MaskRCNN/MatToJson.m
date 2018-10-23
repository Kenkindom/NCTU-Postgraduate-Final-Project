clear all;
clc;

% Read label folder
matFolder = 'C:\Users\Lab031\Desktop\CHT_2017\Mat\';

% Write Train & Val json File
jsonTrainFile = 'C:\Users\Lab031\Desktop\CHT_2017\Annotation\instances_instance2014.json';
jsonValFile = 'C:\Users\Lab031\Desktop\CHT_2017\Annotation\instances_val2014.json';
jsonTrainFid = fopen(jsonTrainFile, 'w');
if jsonTrainFid == -1 
    error('Cannt open Train json File');
end
jsonValFid = fopen(jsonValFile, 'w');
if jsonValFid == -1
    error('Cannt open Val json File');
end

% Write Train & Val json Str
info = '"info": {"description": "CHT 2017 dataset", "contributor": "CHT", "date_create": "2018-03-01"}, ';
license = '"license": [{"id": 1, "name": "CHT License"}], ';
image = '"images": [';
annotation = '"annotations": [';
category = '"categories": [{"id": 1, "name": "person", "supercategory": "person"}]';

Dir = dir([matFolder, '*.mat']);
matNumber = size(Dir, 1);
fprintf("Have %d Labels\n", matNumber);

annotationId = 1;
imageTrain = image;
imageVal = image;
annotationTrain = annotation;
annotationVal = annotation;
% Read all label file
%matNumber = 15;
for matIndex = 1:matNumber
    % Read label file
    MAT = load([matFolder Dir(matIndex).name]);
    fprintf("%4d %s\n", matIndex, Dir(matIndex).name);
    labelNumber = size(MAT.label.region, 2);
    if labelNumber == 0
        delete([matFolder Dir(matIndex).name]);
        totalDeleteNumber = totalDeleteNumber + 1;
        fptintf("\n");
        continue;
    end
    
    jpgName = [Dir(matIndex).name(1:length(Dir(matIndex).name)-4) '.jpg'];
    if mod(matIndex, 5) ~= 1
        imageTrain = [imageTrain '{"id": ' num2str(matIndex) ', "width": 1280, "height": 720, "file_name": "' jpgName '", "license": 1}'];
    else
        imageVal = [imageVal '{"id": ' num2str(matIndex) ', "width": 1280, "height": 720, "file_name": "' jpgName '", "license": 1}'];
    end
    
    for labelIndex = 1:labelNumber
        xmin = 1280;
        ymin = 720;
        xmax = 0;
        ymax = 0;
        
        for segNumber = 1:size(MAT.label.region{1,labelIndex},1)
            [r, w] = find(MAT.label.table == MAT.label.region{1, labelIndex}(segNumber));
            if max(max(r)) > ymax
                ymax = max(max(r));
            end
            if min(min(r)) < ymin
                ymin = min(min(r));
            end
            if max(max(w)) > xmax
                xmax = max(max(w));
            end
            if min(min(w)) < xmin
                xmin = min(min(w));
            end
        end
        [out, area] = annSegmentation(MAT, labelIndex);
        segmentation = '[[';
        for outIndex = 1:size(out, 2)
            for pointIndex = 1:size(out{outIndex}, 1)
                segmentation = [segmentation num2str(out{outIndex}(pointIndex,1)) ',' num2str(out{outIndex}(pointIndex,2))];
                if pointIndex ~= size(out{outIndex}, 1)
                    segmentation = [segmentation ','];
                end
            end
            segmentation = [segmentation ']'];
            if outIndex ~= size(out, 2)
                segmentation = [segmentation ', ['];
            end
        end
        segmentation = [segmentation ']'];
        if mod(matIndex, 5) ~= 1
            annotationTrain = [annotationTrain '{"id": ' num2str(annotationId) ', "image_id": ' num2str(matIndex) ', "category_id": 1, "segmentation": ' segmentation ', "area": ' num2str(area) ', "bbox": [' num2str(xmin) '.0, ' num2str(ymin) '.0, ' num2str(xmax-xmin) '.0, ' num2str(ymax-ymin) '.0], "iscrowd": 0}'];
            annotationId = annotationId + 1;
        else
            annotationVal = [annotationVal '{"id": ' num2str(annotationId) ', "image_id": ' num2str(matIndex) ', "category_id": 1, "segmentation": ' segmentation ', "area": ' num2str(area) ', "bbox": [' num2str(xmin) '.0, ' num2str(ymin) '.0, ' num2str(xmax-xmin) '.0, ' num2str(ymax-ymin) '.0], "iscrowd": 0}'];
            annotationId = annotationId + 1;
        end
            
        if labelIndex ~= labelNumber
            if mod(matIndex, 5) ~= 1
                annotationTrain = [annotationTrain ', '];
            else
                annotationVal = [annotationVal ', '];
            end
        end
    end
    
    if matIndex == matNumber 
        imageTrain = [imageTrain '], '];
        imageVal = [imageVal '], '];
        annotationTrain = [annotationTrain '], '];
        annotationVal = [annotationVal '], '];
    else
        if mod(matIndex, 5) ~= 1
            imageTrain = [imageTrain ', '];
            annotationTrain = [annotationTrain ', '];
        else
            imageVal = [imageVal ', '];
            annotationVal = [annotationVal ', '];
        end
    end
    
end

fprintf(jsonTrainFid, "{%s %s %s %s %s}", info, imageTrain, license, annotationTrain, category);
fprintf(jsonValFid, "{%s %s %s %s %s}", info, imageVal, license, annotationVal, category);
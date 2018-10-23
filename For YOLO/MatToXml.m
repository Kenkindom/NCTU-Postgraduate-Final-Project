clear all;
clc;

% Read label folder
matFolder = 'C:\Users\Lab031\Desktop\CHT_2017\Mat\morning\';
xmlFolder = 'C:\Users\Lab031\Desktop\CHT_2017\Annotation\morning\';
jpgFolder = 'C:\Users\Lab031\Desktop\CHT_2017\JPEGImages\';
txtName = 'morningfile.txt';
txtFid = fopen(txtName, 'w');
Dir = dir([matFolder, '*.mat']);
matNumber = size(Dir, 1);
fprintf("Have %d Labels\n", matNumber);

totalDeleteNumber = 0;
totalLabelNumber = 0;
% Read all label file
for matIndex = 1:matNumber
    fprintf("%s ", Dir(matIndex).name);
    
    % Read label file
    MAT = load([matFolder Dir(matIndex).name]);
    labelNumber = size(MAT.label.region, 2);
    if labelNumber == 0
        delete([matFolder Dir(matIndex).name]);
        totalDeleteNumber = totalDeleteNumber + 1;
        fptintf("\n");
        continue;
    end
    
    % Create xml file for each label file
    xmlName = [Dir(matIndex).name(1:length(Dir(matIndex).name)-4) '.xml'];
    jpgName = [Dir(matIndex).name(1:length(Dir(matIndex).name)-4) '.jpg'];
    fprintf(txtFid, "%s\n", Dir(matIndex).name(1:length(Dir(matIndex).name)-4));
    
    xmlFid = fopen([xmlFolder xmlName], 'w');
    if xmlFid < 0 
        fprintf("Open File Error!\n");
        break;
    end
    
    % Write xml file
    fprintf(xmlFid, "<annotation>\n");
    fprintf(xmlFid, "\t<folder>JPEGImages</folder>\n");
    fprintf(xmlFid, "\t<filename>%s</filename>\n", jpgName);
    fprintf(xmlFid, "\t<source>\n");
    fprintf(xmlFid, "\t\t<database>CHT 2017 Database</database>\n");
    fprintf(xmlFid, "\t</source>\n");
    fprintf(xmlFid, "\t<size>\n");
    fprintf(xmlFid, "\t\t<width>1280</width>\n");
    fprintf(xmlFid, "\t\t<height>720</height>\n");
    fprintf(xmlFid, "\t\t<depth>3</depth>\n");
    fprintf(xmlFid, "\t</size>\n");
    fprintf(xmlFid, "\t<segmented>0</segmented>\n"); % ???????
    
    % Get label detail
    fprintf("Have %d Label(s)", labelNumber);
    for labelIndex = 1:labelNumber
        fprintf(", %d", labelIndex);
        
        xmin = 1280;
        ymin = 720;
        xmax = 0;
        ymax = 0;
        fprintf(" Have %d segmentation", size(MAT.label.region{1,labelIndex},1));
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
        
        fprintf(xmlFid, "\t<object>\n");
        fprintf(xmlFid, "\t\t<name>person</name>\n");
        fprintf(xmlFid, "\t\t<pose>Unspecified</pose>\n");
        fprintf(xmlFid, "\t\t<truncated>0</truncated>\n");
        fprintf(xmlFid, "\t\t<difficult>0</difficult>\n");
        fprintf(xmlFid, "\t\t<bndbox>\n");
        fprintf(xmlFid, "\t\t\t<xmin>%d</xmin>\n", xmin);
        fprintf(xmlFid, "\t\t\t<ymin>%d</ymin>\n", ymin);
        fprintf(xmlFid, "\t\t\t<xmax>%d</xmax>\n", xmax);
        fprintf(xmlFid, "\t\t\t<ymax>%d</ymax>\n", ymax);
        fprintf(xmlFid, "\t\t</bndbox>\n");
        fprintf(xmlFid, "\t</object>\n");
        totalLabelNumber = totalLabelNumber + 1;
    end
    fprintf(xmlFid, "</annotation>\n");
    fprintf("\n");
    
    % Close xml file
    status = fclose(xmlFid);
    if status == -1
        fprintf("Close file error!\n");
        break;
    end
end

fprintf("Delete Number : %d\nLabel Number : %d\n", totalDeleteNumber, totalLabelNumber);
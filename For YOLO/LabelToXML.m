function LabelToXML
    %% Read label folder
    matFolder = 'C:\Users\Lab031\Desktop\CHT_2017\Label\';
    xmlFolder = 'C:\Users\Lab031\Desktop\CHT_2017\LabelToAnnotation\';
    Dir = dir([matFolder, '*.mat']);
    matNumber = size(Dir, 1);
    fprintf("Have %d Labels\n", matNumber);
    %matNumber = 1;

    %% Process Every Mat File
    oldName = [];
    for matIndex = 1:matNumber
        matName = Dir(matIndex).name;
        MAT = load([matFolder matName]);
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

        %% Open XML File
        jpgName = [matName(1:jpgLen) '.jpg'];
        xmlName = [matName(1:jpgLen) '.xml'];
        if matIndex == 1
            xmlFid = fopen([xmlFolder xmlName], 'w');
        elseif ~strcmp(oldName, jpgName)
            fprintf(xmlFid, "</annotation>\n");
            status = fclose(xmlFid);
            if status == -1
                error("Close file error!\n");
            end
            xmlFid = fopen([xmlFolder xmlName], 'w');
        end
        fprintf("Name : %s, Label : %d\n", jpgName, labelIndex);
        if xmlFid < 0 
            fprintf("Open XML File Error!");
        end

        %% Resize Mask & Find Bounding Box's Four Corner
        mask = imresize(MAT.label.mask, [MAT.label.height, MAT.label.width]);
        mask(find(mask > 0.5)) = 1;
        mask(find(mask <= 0.5)) = 0;
        [r, w] = find(mask == 1);
        xmin = min(min(w)) + MAT.label.x;
        xmax = max(max(w)) + MAT.label.x;
        ymin = min(min(r)) + MAT.label.y;
        ymax = max(max(r)) + MAT.label.y;
        if xmin < 1 
            xmin = 1;
        end
        if ymin < 1
            ymin = 1;
        end
        if xmax > 1280
            xmax = 1280;
        end
        if ymax > 1280
            ymax = 720;
        end

        %% Write XML File
        if labelIndex == 1
            % First Write This XML File
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
            fprintf(xmlFid, "\t<segmented>0</segmented>\n");
        end
        writeFile(xmlFid, xmin, xmax, ymin, ymax);
%         fprintf(xmlFid, "\t<object>\n");
%         fprintf(xmlFid, "\t\t<name>person</name>\n");
%         fprintf(xmlFid, "\t\t<pose>Unspecified</pose>\n");
%         fprintf(xmlFid, "\t\t<truncated>0</truncated>\n");
%         fprintf(xmlFid, "\t\t<difficult>0</difficult>\n");
%         fprintf(xmlFid, "\t\t<bndbox>\n");
%         fprintf(xmlFid, "\t\t\t<xmin>%d</xmin>\n", xmin);
%         fprintf(xmlFid, "\t\t\t<ymin>%d</ymin>\n", ymin);
%         fprintf(xmlFid, "\t\t\t<xmax>%d</xmax>\n", xmax);
%         fprintf(xmlFid, "\t\t\t<ymax>%d</ymax>\n", ymax);
%         fprintf(xmlFid, "\t\t</bndbox>\n");
%         fprintf(xmlFid, "\t</object>\n");

        %% Down Sampling
        width = xmax - xmin;
        height = ymax - ymin;
        if (width*2)/3 > 20
            if (height*2)/3 > 20
                writeFile(xmlFid, xmin, xmin + (width*2/3), ymin, ymin + (height*2)/3);
                writeFile(xmlFid, xmax - (width*2/3), xmax, ymin, ymin + (height*2)/3);
                writeFile(xmlFid, xmin, xmin + (width*2/3), ymax - (height*2)/3, ymax);
                writeFile(xmlFid, xmax - (width*2/3), xmax, ymax - (height*2)/3, ymax);
            end
            writeFile(xmlFid, xmin, xmin + (width*2/3), ymin, ymax);
            writeFile(xmlFid, xmax - (width*2/3), xmax, ymin, ymax);
        else
            if (height*2)/3 > 20
                writeFile(xmlFid, xmin, xmax, ymin, ymin + (height*2)/3);
                writeFile(xmlFid, xmin, xmax, ymax - (height*2)/3, ymax);
            end
        end

        oldName = jpgName;
    end
    fprintf(xmlFid, "</annotation>\n");
    fclose(xmlFid);
end

function writeFile(xmlFid, xmin, xmax, ymin, ymax)
    fprintf(xmlFid, "\t<object>\n");
    fprintf(xmlFid, "\t\t<name>person</name>\n");
    fprintf(xmlFid, "\t\t<pose>Unspecified</pose>\n");
    fprintf(xmlFid, "\t\t<truncated>0</truncated>\n");
    fprintf(xmlFid, "\t\t<difficult>0</difficult>\n");
    fprintf(xmlFid, "\t\t<bndbox>\n");
    fprintf(xmlFid, "\t\t\t<xmin>%d</xmin>\n", round(xmin));
    fprintf(xmlFid, "\t\t\t<ymin>%d</ymin>\n", round(ymin));
    fprintf(xmlFid, "\t\t\t<xmax>%d</xmax>\n", round(xmax));
    fprintf(xmlFid, "\t\t\t<ymax>%d</ymax>\n", round(ymax));
    fprintf(xmlFid, "\t\t</bndbox>\n");
    fprintf(xmlFid, "\t</object>\n");
end
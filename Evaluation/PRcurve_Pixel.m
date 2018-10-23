function PRcurve_Pixel
    txtFile = ['./BoundingBox/person_backpack_boundingbox_Pixel.txt'];
    txtFid = fopen(txtFile, 'r');

    test = {};
    con = [];
    jpgName = [];
    testNumber = 0;
    rowIndex = 1;
    while 1
        % Read File
        line = fgetl(txtFid);
        if ~isstr(line) || line(1) == '1'
            % New Image
            if ~isstr(line)
                % File End
                break;
            end
            jpgName = line(3: size(line, 2));
            if strcmp(jpgName, ".") || strcmp(jpgName, "..")
                % Skip Strange Line
                line = fgetl(txtFid);
                continue;
            end
        elseif line(1) == '2'
            % Get Person Line
            testNumber = testNumber + 1;
            % Split Line By Blank Or \t
            split = regexp(line(3: size(line, 2)), '\t', 'split');
            bound = regexp(split{1,2}, ' ', 'split');
            confidence = str2num(bound{1,5});
            test{1,testNumber}.name = jpgName;
            test{1,testNumber}.x = str2num(bound{1,1}) + 1;
            test{1,testNumber}.y = str2num(bound{1,2}) + 1;
            test{1,testNumber}.width = str2num(bound{1,3});
            test{1,testNumber}.height =  str2num(bound{1,4});
            test{1,testNumber}.confidence =  str2num(bound{1,5});
            con = [con, confidence];
            test{1,testNumber}.mask = zeros(720, 1280);
            rowIndex = 1;
            count = 0;
        elseif line(1) == '[' || line(2) == '['
            canBreak = 0;
            num = 0;
            colIndex = 1;
            while(1)
                if line(size(line, 2)) == ']'
                    if line(size(line, 2)-1) == ']'
                        line = line(1: size(line, 2)-2);
                    else
                        line = line(1: size(line, 2)-1);
                    end
                    canBreak = 1;
                end
                split = regexp(line(3: size(line, 2)), ' ', 'split');
                for index = 1:size(split, 2)
                    test{1,testNumber}.mask(rowIndex, colIndex) = str2num(split{1, index});
                    if str2num(split{1, index}) == 1
                        count = count + 1;
                    end
                    colIndex = colIndex + 1;
                end
                num = num + size(split, 2);
                if canBreak == 1
                    rowIndex = rowIndex + 1;
                    break;
                end
                line = fgetl(txtFid);
            end
        end
    end
    
    [~, index] = sort(con, 'descend');
    temp = test(index);
    save 'PixelArray.mat' 'temp';
    
%     temp = load('PixelArrayMorning.mat');
%     temp = temp.temp;
%     testNumber = size(temp, 2);

    same = [];
    re = [];
    pre = [];
    sameNumber = 0;
    dontcarePre = 0;
    dontNumber = 0;
    for tempIndex = 1:testNumber
        fprintf("%4d/%d, Same %d\n", tempIndex, testNumber, sameNumber);
        use = temp{1, tempIndex};
        isDont = 0;
        DIR = dir(['./Output/Morning/' use.name(1:size(use.name,2)-4) '_*.mat']);
        for dirIndex = 1:size(DIR, 1)
            MAT = load(['./Output/Morning/' DIR(dirIndex).name]);
            mask = imresize(MAT.label.mask, [MAT.label.height, MAT.label.width]);
            mask(find(mask > 0.5)) = 1;
            mask(find(mask <= 0.5)) = 0;
            
            [rate] = sameMask(use, mask, MAT.label.x, MAT.label.y, 1280, 720);
            if rate >= 0.5
                if MAT.label.DontCare == 1
                    if isDont == 0
                        isDont = 1;
                        dontcarePre = dontcarePre + 1;
                    end
                    continue;
                end
                if isDont == 1
                    isDont = 0;
                    dontcarePre = dontcarePre - 1;
                end
                if MAT.label.DontCare == 1
                    dontNumber = dontNumber + 1;
                end
                sameNumber = sameNumber + 1;
                break;
            end
        end
        if isDont == 0
            same = [same, sameNumber];
            pre = [pre, sameNumber/(tempIndex )];
        end
    end
    
    matNumber = 0;
    matFolder = './Output/Morning/';
    Dir = dir([matFolder, '*.mat']);
    for matIndex = 1:size(Dir, 1)
        MAT = load([matFolder Dir(matIndex).name]);
        if MAT.label.DontCare == 1
            continue;
        end
        matNumber = matNumber + 1;
    end
    
    fprintf("Mat Bounding Box Number is %d\n", matNumber);
    fprintf("Size recall %d, precision %d\n", size(same, 2), size(pre, 2));
    re = same./(matNumber);
    
    area = 0;
    old = 0;
    for areaIndex = 1:size(pre, 2)
        if re(areaIndex) == old
            continue;
        end
        if areaIndex == size(pre, 2) || areaIndex == 1
            area = area + pre(areaIndex) / (2 * matNumber);
        else
            area = area + pre(areaIndex) / matNumber;
        end
        old = re(areaIndex);
    end
    fprintf("%f %f\n", area, VOCap(re', pre'));
    
    table.precision = pre;
    table.recall = re;
    table.labelNumber = matNumber;
    table.sameNumber = sameNumber;
    table.area = area;
    
    if size(same, 2) ~= 0
        hold on;
        plot(re, pre, '-');
        axis([0 1 0 1]);
        title('MaskRCNN Morning Pixel');
        xlabel('recall');
        ylabel('precision');
    end
    save(['./Evaluation/MaskRCNN_Morning_Pixel.mat'], 'table');
end

function [rate] = sameMask(test, mask, x, y, width, height)
    rate = 0;
    
    if size(mask, 1) > height || size(mask, 2) > width
        return;
    end
    
    board = zeros(height, width);
    boardA = zeros(height, width);
    boardB = zeros(height, width);
    boardA = test.mask;
    if x < 1
        x = 1;
    end
    if y < 1
        y = 1;
    end
    boardy = y + size(mask,1)-1;
    boardx = x + size(mask,2)-1;
    if y+size(mask, 1) > 720
        boardy = 720;
        y = 720 - size(mask, 1) + 1;
    end
    if x+size(mask, 2) > 1280
        boardx = 1280;
        x = 1280 - size(mask, 2) + 1;
    end
    boardB(y:boardy, x:boardx) = mask;
    board = boardA + boardB;
    clear tempBoardA; clear tempBoardB; clear boardA; clear boardB;

    insectionMask = size(find(board == 2), 1);
    collectionMask = size(find(board ~= 0), 1);

    if insectionMask/collectionMask > rate
        rate = insectionMask/collectionMask;
    end
end

function ap = VOCap(rec,prec)
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
        mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
end
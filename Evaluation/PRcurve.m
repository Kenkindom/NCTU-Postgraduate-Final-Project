function PRcurve
    epoch = '10000';
    txtFile = ['./BoundingBox/YOLO_Morning_boundingbox.txt'];
    txtFid = fopen(txtFile, 'r');

    test = {};
    con = [];
    testNumber = 0;
    while 1
        line = fgetl(txtFid);
        if ~isstr(line) || line(1) == '1'
            if ~isstr(line)
                break;
            end
            jpgName = line(3: size(line, 2));
            if strcmp(jpgName, ".") || strcmp(jpgName, "..")
                line = fgetl(txtFid);
                continue;
            end
        elseif line(1) == '2'
            testNumber = testNumber + 1;
%             bound = regexp(line(3: size(line, 2)), ' ', 'split');
%             confidence = str2num(bound{1,6});
%             test{1,testNumber}.name = jpgName;
%             test{1,testNumber}.x = str2num(bound{1,2}) + 1;
%             test{1,testNumber}.y = str2num(bound{1,3}) + 1;
%             test{1,testNumber}.width = str2num(bound{1,4});
%             test{1,testNumber}.height =  str2num(bound{1,5});
%             test{1,testNumber}.confidence =  str2num(bound{1,6});
%             con = [con, confidence];
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
        end
    end
%     close(txtFid);
    
    [~, index] = sort(con, 'descend');
    temp = test(index);
    
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
            if ymax > 720
                ymax = 720;
            end
            [rate] = sameMask(use, xmin, xmax, ymin, ymax, 1280, 720);
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
        %plot(re, pre, '-');
        plot(re, pre, '-');
        axis([0 1 0 1]);
        title('YOLO Morning');
        xlabel('recall');
        ylabel('precision');
    end
    save(['./Evaluation/YOLO_Morning.mat'], 'table');
end

function [rate] = sameMask(test, xmin, xmax, ymin, ymax, width, height)
    rate = 0;
        
    board = zeros(height, width);
    boardA = zeros(height, width);
    boardB = zeros(height, width);
    if test.y+test.height > 720
        test.height = 720 - test.y;
    end
    if test.x+test.width > 1280
        test.width = 1280 - test.x;
    end
    tempBoardA = boardA(test.y:test.y+test.height, test.x:test.x+test.width);
    tempBoardA = 1;
    boardA(test.y:test.y+test.height, test.x:test.x+test.width) = tempBoardA;
    tempBoardB =  boardB(ymin:ymax, xmin:xmax);
    tempBoardB = 2;
    boardB(ymin:ymax, xmin:xmax) = tempBoardB;
    board = boardA + boardB;
    clear tempBoardA; clear tempBoardB; clear boardA; clear boardB;

    insectionMask = size(find(board == 3), 1);
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
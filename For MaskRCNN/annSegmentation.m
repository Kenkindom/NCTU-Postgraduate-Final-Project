function [out, area] = annSegmentation(in, index)
    height = size(in.label.table, 1);
    width = size(in.label.table, 2);
    table = zeros(height, width);
    segNumber = size(in.label.region{1, index}, 1);
    for segIndex = 1:segNumber
        [r, w] = find(in.label.table == in.label.region{1, index}(segIndex));
        table((w-1) * height + r) = 1;
        clear r; clear w;
    end
    
    [ri, wi] = find(table == 1);
    area = size(ri, 1);
    
    % find margin
    tempW = 2:width-1;
    tempH = 2:height-1;
    [row, wid] = find(table(tempH, tempW) == 1 & table(tempH-1, tempW) == 1 & table(tempH+1, tempW) == 1 & table(tempH, tempW-1) == 1 & table(tempH, tempW+1) == 1);
    ind = (tempW(wid) - 1) * height + tempH(row);
    table(ind) = 0;
    
    % find margin's corner
    finishFlag = 0;
    isFind = 0;
    circle = 0;
    path = zeros(height, width);
    pointIndex = 0;
    next2 = [];
    next2Index = 0;
    isPush = 0;
    while pointIndex <= size(find(table == 1),1)
        % Is first loop or have another margin don't find
        if finishFlag == 0 && isFind == 0
            circle = circle + 1;
            isFind = 1;
            temp = table - path;
            if isempty(find(temp == -1, 1)) == 0
                error('Why Have Negative!');
            end
            [r, w] = find(temp == 1);
            preX = w(1);    preY = r(1);
            nowX = 0;       nowY = 0;
            path(preY, preX) = 1;
            out{circle} = [preX, preY];
            if preY+1 <= height && table(preY+1, preX) == 1 && path(preY+1, preX) == 0
                nowX = preX;    nowY = preY+1;
            elseif preX+1 <= width && preY+1 <= height && table(preY+1, preX+1) == 1 && path(preY+1, preX+1) == 0
                nowX = preX+1;  nowY = preY+1;
            end
            
            if nowX == 0 || nowY == 0
                if size(out{circle}, 1 == 1)
                    isFind = 0;
                    pointIndex = pointIndex + 1;
                    if size(find(table == 1), 1) == size(find(path == 1), 1)
                        finishFlag = 1;
                    end
                    continue;
                end
                error('X and Y is Error!');
            end
            path(nowY, nowX) = 1;
            pointIndex = pointIndex + 2;
        end
        
        % makesure direction is uncounter clock
        site = getScore(preX, preY, nowX, nowY);
        if site == -1 && isPush == 0
            error('Error Site');
        end
        
        for stepIndex = 1:8
            site = mod(site+1, 8);
            % find next point
            [nextX, nextY] = findNext(nowX, nowY, site);
            % out of boundary
            if nextX<1 || nextX>width || nextY<1 || nextY>height
                continue;
            end
            % this point 
            if table(nextY, nextX) == 1 && path(nextY, nextX) ~= 1
                break;
            end
            nextX = 0;
            nextY = 0;
        end
        
        for stepIndex = 1:7
            site = mod(site+1, 8);
            [tempX, tempY] = findNext(nowX, nowY, site);
            % out of boundary
            if tempX<1 || tempX>width || tempY<1 || tempY>height
                continue;
            end
            % this point 
            if table(tempY, tempX) == 1 && path(tempY, tempX) ~= 1
                if next2Index == 0
                    next2 = [next2; [tempX, tempY]];
                    next2Index = next2Index + 1;
                elseif next2Index > 0 && ~isempty(find((next2(next2Index, :) == [tempX, tempY]) == 0))
                    next2 = [next2; [tempX, tempY]];
                    next2Index = next2Index + 1;
                end
                break;
            end
        end
        
        % make sure next2 point will be walk
        if next2Index > 0 && nextX == next2(next2Index, 1) && nextY == next2(next2Index, 2)
            if next2Index == 1
                next2 = [];
                next2Index = 0;
            else
                next2Index = next2Index - 1;
                next2 = next2(1:next2Index, :);
            end
        end
        
        if nextX == 0 && nextY == 0
            if getScore(nowX, nowY, out{circle}(1, 1), out{circle}(1, 2)) >= 0
                out{circle} = [out{circle}; [nowX, nowY]];
                pointIndex = pointIndex + 1;
                isFind = 0;
                if size(find(table == 1), 1) == size(find(path == 1), 1)
                    finishFlag = 1;
                end
                continue;
            elseif next2Index ~= 0
                nextX = next2(next2Index, 1);
                nextY = next2(next2Index, 2);
                if next2Index == 1
                    next2 = [];
                    next2Index = 0;
                else
                    next2Index = next2Index - 1;
                    next2 = next2(1:next2Index, :);
                end
                isPush = 1;
                path(nextY, nextX) = 1;
                out{circle} = [out{circle}; [nowX, nowY]];
            end
        else
            if nextY == 0 || nextX == 0
                error('Next (x, y) Have 0');
            end
            path(nextY, nextX) = 1;
            if isPush == 1
                isPush = 0;
                out{circle} = [out{circle}; [nowX, nowY]];
            elseif isLine(preX, preY, nowX, nowY, nextX, nextY) == 0
                out{circle} = [out{circle}; [nowX, nowY]];
            end
        end
        pointIndex = pointIndex + 1;
        preX = nowX;    preY = nowY;
        nowX = nextX;   nowY = nextY;
    end
    
    for cellIndex = 1:size(out, 2)
        table((out{cellIndex}(:,1)-1) * height + out{cellIndex}(:,2)) = 2;
    end
    
end

% calculate where are pre point at now point
function site = getScore(preX, preY, nowX, nowY)
    site = -1;
    if nowY-1 == preY && nowX-1 == preX
        site = 0;
    elseif nowY == preY && nowX-1 == preX
        site = 1;
    elseif nowY+1 == preY && nowX-1 == preX
        site = 2;
    elseif nowY-1 == preY && nowX == preX
        site = 7;
    elseif nowY+1 == preY && nowX == preX
        site = 3;
    elseif nowY-1 == preY && nowX+1 == preX
        site = 6;
    elseif nowY == preY && nowX+1 == preX
        site = 5;
    elseif nowY+1 == preY && nowX+1 == preX
        site = 4;
    end
end

% find uncounterclock next point
function [nextX, nextY] = findNext(X, Y, score)
    if score == 0   
        nextX = X-1;    nextY = Y-1;
    elseif score == 1
        nextX = X-1;    nextY = Y;
    elseif score == 2
        nextX = X-1;    nextY = Y+1;
    elseif score == 3
        nextX = X;    nextY = Y+1;
    elseif score == 4
        nextX = X+1;    nextY = Y+1;
    elseif score == 5
        nextX = X+1;    nextY = Y;
    elseif score == 6
        nextX = X+1;    nextY = Y-1;
    elseif score == 7
        nextX = X;    nextY = Y-1;
    end
end

% check this three point is at same line
function out = isLine(preX, preY, nowX, nowY, nextX, nextY)
    if preX == nowX && nowX == nextX
        out = 1;
    else
        m1 = (nowY - preY) / (nowX - preX);
        m2 = (nextY - nowY) / (nextX - nowX);
        if m1 == m2
            out = 1;
        else
            out = 0;
        end
    end
end
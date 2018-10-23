matFolder = './Mat/ALL/';
%matFolder = './Mat/20180505/';
Dir = dir([matFolder, '*.mat']);
matNumber = size(Dir, 1);

dontcare = 0;
unlabeled = 0;

% age
ageC = zeros(1,2); ageT = zeros(1,2); ageA = zeros(1,2); ageM = zeros(1,2); ageS = zeros(1,2); ageUn = zeros(1,2);
% gender
genderM = zeros(1,2); genderF = zeros(1,2); genderUn = zeros(1,2);
% upper - color
upperCRed = zeros(1,2); upperCOrange = zeros(1,2); upperCYellow = zeros(1,2); upperCGreen = zeros(1,2);
upperCBlue = zeros(1,2); upperCPink = zeros(1,2); upperCPurple = zeros(1,2); upperCBrown = zeros(1,2);
upperCGray = zeros(1,2); upperCBlack = zeros(1,2); upperCWhite = zeros(1,2); upperCUn = zeros(1,2);
% upper - sleeve
upperSL = zeros(1,2); upperSS = zeros(1,2); upperSUn = zeros(1,2);
% upper - design
upperDPlain = zeros(1,2); upperDPlaid = zeros(1,2); upperDLogo = zeros(1,2); upperDStriped = zeros(1,2);
upperDDot = zeros(1,2); upperDUn = zeros(1,2);
% lower - color
lowerCRed = zeros(1,2); lowerCOrange = zeros(1,2);  lowerCYellow = zeros(1,2); lowerCGreen = zeros(1,2);
lowerCBlue = zeros(1,2); lowerCPink = zeros(1,2); lowerCPurple = zeros(1,2); lowerCBrown = zeros(1,2);
lowerCGray = zeros(1,2); lowerCBlack = zeros(1,2); lowerCWhite = zeros(1,2); lowerCUn = zeros(1,2);
% lower - type
lowerTT = zeros(1,2); lowerTP = zeros(1,2);  lowerTS = zeros(1,2); lowerTL = zeros(1,2);lowerTUn = zeros(1,2);
% carry
carryBack = zeros(1,2); carrySat = zeros(1,2); carryHand = zeros(1,2); carrySuit = zeros(1,2);
carryUmO = zeros(1,2); carryUmF = zeros(1,2);  carryPram = zeros(1,2); carryBaby = zeros(1,2); carryWheel = zeros(1,2);
CarryNo = zeros(1,2);
% hat
hat = zeros(1,2); hood = zeros(1,2); nohat = zeros(1,2);

nighttime = 0; daytime = 0; timeIndex = 0; count = 0;

fprintf("mat Number %d\n", matNumber);
randomID = randperm(matNumber);
dayNoHave = 0; dayHave = 0; nightNoHave = 0; nightHave = 0;

for matIndex = 1:matNumber
    fileName = Dir(randomID(matIndex)).name(1:size(Dir(randomID(matIndex)).name, 2)-4);
    MAT = load([matFolder fileName '.mat']);
    %fprintf("%s ", Dir(matIndex).name);
    if MAT.label.DontCare == 1
        dontcare = dontcare + 1;
        continue;
    end
    
    haveUmbrella = 0;
    %%%%%%%%%%%%time%%%%%%%%%%%%
    if strcmp(MAT.label.Time, 'TimeNighttime')
        nighttime = nighttime + 1;
        timeIndex = 1;
    elseif strcmp(MAT.label.Time, 'TimeDaytime')
        daytime = daytime + 1;
        timeIndex = 2;
    end
    
    if strcmp(MAT.label.Gender, 'GenderUnlabeled') == 1 || strcmp(MAT.label.Age, 'AgeUnlabeled') == 1 || strcmp(MAT.label.Ridding, 'RiddingUnlabeled') == 1
        fprintf("%s have unlabeled label\n", Dir(randomID(matIndex)).name);
        %copyfile([matFolder Dir(matIndex).name], 'C:/Users/Lab031/Desktop/LABEL_ATTRIBUTE/input');
        unlabeled = unlabeled + 1;
        continue;
    elseif strcmp(MAT.label.UpperBodyColor, 'UpperBodyColorUnlabeled') == 1 || strcmp(MAT.label.UpperBodyDesign, 'UpperBodyDesignUnlabeled') == 1 || strcmp(MAT.label.UpperBodySleeve, 'UpperBodySleeveUnlabeled') == 1 
        fprintf("%s have unlabeled label\n", Dir(randomID(matIndex)).name);
        %copyfile([matFolder Dir(matIndex).name], 'C:/Users/Lab031/Desktop/LABEL_ATTRIBUTE/input');
        unlabeled = unlabeled + 1;
        continue;
    elseif strcmp(MAT.label.LowerBodyColor, 'LowerBodyColorUnlabeled') == 1 || strcmp(MAT.label.LowerBodyType, 'LowerBodyTypeUnlabeled') == 1
        fprintf("%s have unlabeled label\n", Dir(randomID(matIndex)).name);
        %copyfile([matFolder Dir(matIndex).name], 'C:/Users/Lab031/Desktop/LABEL_ATTRIBUTE/input');
        unlabeled = unlabeled + 1;
        continue;
    end
    
    %%%%%%%%%%%%age%%%%%%%%%%%%
    if strcmp(MAT.label.Age, 'AgeChild') == 1
        ageC(1, timeIndex) = ageC(1, timeIndex) + 1;
    elseif strcmp(MAT.label.Age, 'AgeTeen') == 1
        ageT(1, timeIndex) = ageT(1, timeIndex) + 1;
    elseif strcmp(MAT.label.Age, 'AgeAdult') == 1
        ageA(1, timeIndex) = ageA(1, timeIndex) + 1;
    elseif strcmp(MAT.label.Age, 'AgeMiddle') == 1
        ageM(1, timeIndex) = ageM(1, timeIndex) + 1;
    elseif strcmp(MAT.label.Age, 'AgeSenior') == 1
        ageS(1, timeIndex) = ageS(1, timeIndex) + 1;
    elseif strcmp(MAT.label.Age, 'AgeUnknown') == 1
        ageUn(1, timeIndex) = ageUn(1, timeIndex) + 1;
    else
        fprintf("%s have error button\n", Dir(randomID(matIndex)).name);
        %copyfile([matFolder Dir(matIndex).name], 'C:/Users/Lab031/Desktop/LABEL_ATTRIBUTE/input');
    end
    %%%%%%%%%%%%gender%%%%%%%%%%%%
    if strcmp(MAT.label.Gender, 'GenderMale') == 1
        genderM(1, timeIndex) = genderM(1, timeIndex) + 1;
    elseif strcmp(MAT.label.Gender, 'GenderFemale') == 1
        genderF(1, timeIndex) = genderF(1, timeIndex) + 1;
    elseif strcmp(MAT.label.Gender, 'GenderUnknown') == 1
        genderUn(1, timeIndex) = genderUn(1, timeIndex) + 1;
    else
        fprintf("%s have error button\n", Dir(randomID(matIndex)).name);
        %copyfile([matFolder Dir(matIndex).name], 'C:/Users/Lab031/Desktop/LABEL_ATTRIBUTE/input');
    end
    %%%%%%%%%%%%upper%%%%%%%%%%%%
    %%%%color%%%%
    if strcmp(MAT.label.UpperBodyColor, 'UpperBodyColorRed') == 1
        upperCRed(1, timeIndex) = upperCRed(1, timeIndex) + 1;
    elseif strcmp(MAT.label.UpperBodyColor, 'UpperBodyColorOrange') == 1
        upperCOrange(1, timeIndex) = upperCOrange(1, timeIndex) + 1;
    elseif strcmp(MAT.label.UpperBodyColor, 'UpperBodyColorYellow') == 1
        upperCYellow(1, timeIndex) = upperCYellow(1, timeIndex) + 1;
    elseif strcmp(MAT.label.UpperBodyColor, 'UpperBodyColorGreen') == 1
        upperCGreen(1, timeIndex) = upperCGreen(1, timeIndex) + 1;
    elseif strcmp(MAT.label.UpperBodyColor, 'UpperBodyColorBlue') == 1
        upperCBlue(1, timeIndex) = upperCBlue(1, timeIndex) + 1;
    elseif strcmp(MAT.label.UpperBodyColor, 'UpperBodyColorPink') == 1
        upperCPink(1, timeIndex) = upperCPink(1, timeIndex) + 1;
    elseif strcmp(MAT.label.UpperBodyColor, 'UpperBodyColorPurple') == 1
        upperCPurple(1, timeIndex) = upperCPurple(1, timeIndex) + 1;
    elseif strcmp(MAT.label.UpperBodyColor, 'UpperBodyColorBrown') == 1
        upperCBrown(1, timeIndex) = upperCBrown(1, timeIndex) + 1;
    elseif strcmp(MAT.label.UpperBodyColor, 'UpperBodyColorGray') == 1
        upperCGray(1, timeIndex) = upperCGray(1, timeIndex) + 1;
    elseif strcmp(MAT.label.UpperBodyColor, 'UpperBodyColorBlack') == 1
        upperCBlack(1, timeIndex) = upperCBlack(1, timeIndex) + 1;
    elseif strcmp(MAT.label.UpperBodyColor, 'UpperBodyColorWhite') == 1
        upperCWhite(1, timeIndex) = upperCWhite(1, timeIndex) + 1;
    elseif strcmp(MAT.label.UpperBodyColor, 'UpperBodyColorUnknown') == 1
        upperCUn(1, timeIndex) = upperCUn(1, timeIndex) + 1;
    else
        fprintf("%s have error button\n", Dir(randomID(matIndex)).name);
        %copyfile([matFolder Dir(matIndex).name], 'C:/Users/Lab031/Desktop/LABEL_ATTRIBUTE/input');
    end
    %%%%sleeve%%%%
    if strcmp(MAT.label.UpperBodySleeve, 'UpperBodySleeveLong') == 1
        upperSL(1, timeIndex) = upperSL(1, timeIndex) + 1;
    elseif strcmp(MAT.label.UpperBodySleeve, 'UpperBodySleeveShort') == 1
        upperSS(1, timeIndex) = upperSS(1, timeIndex) + 1;
    elseif strcmp(MAT.label.UpperBodySleeve, 'UpperBodySleeveUnknown') == 1
        upperSUn(1, timeIndex) = upperSUn(1, timeIndex) + 1;
    else
        fprintf("%s have error button\n", Dir(randomID(matIndex)).name);
        %copyfile([matFolder Dir(matIndex).name], 'C:/Users/Lab031/Desktop/LABEL_ATTRIBUTE/input');
    end
    %%%%design%%%%
    if strcmp(MAT.label.UpperBodyDesign, 'UpperBodyDesignPlain') == 1
        upperDPlain(1, timeIndex) = upperDPlain(1, timeIndex) + 1;
    elseif strcmp(MAT.label.UpperBodyDesign, 'UpperBodyDesignPlaid') == 1
        upperDPlaid(1, timeIndex) = upperDPlaid(1, timeIndex) + 1;
    elseif strcmp(MAT.label.UpperBodyDesign, 'UpperBodyDesignLogo') == 1
        upperDLogo(1, timeIndex) = upperDLogo(1, timeIndex) + 1;
    elseif strcmp(MAT.label.UpperBodyDesign, 'UpperBodyDesignStriped') == 1
        upperDStriped(1, timeIndex) = upperDStriped(1, timeIndex) + 1;
    elseif strcmp(MAT.label.UpperBodyDesign, 'UpperBodyDesignDot') == 1
        upperDDot(1, timeIndex) = upperDDot(1, timeIndex) + 1;
    elseif strcmp(MAT.label.UpperBodyDesign, 'UpperBodyDesignUnknown') == 1
        upperDUn(1, timeIndex) = upperDUn(1, timeIndex) + 1;
    else
        fprintf("%s have error button\n", Dir(randomID(matIndex)).name);
        %copyfile([matFolder Dir(matIndex).name], 'C:/Users/Lab031/Desktop/LABEL_ATTRIBUTE/input');
    end
    %%%%%%%%%%%%lower%%%%%%%%%%%%
    %%%%color%%%%
    if strcmp(MAT.label.LowerBodyColor, 'LowerBodyColorRed') == 1
        lowerCRed(1, timeIndex) = lowerCRed(1, timeIndex) + 1;
    elseif strcmp(MAT.label.LowerBodyColor, 'LowerBodyColorOrange') == 1
        lowerCOrange(1, timeIndex) = lowerCOrange(1, timeIndex) + 1;
    elseif strcmp(MAT.label.LowerBodyColor, 'LowerBodyColorYellow') == 1
        lowerCYellow(1, timeIndex) = lowerCYellow(1, timeIndex) + 1;
    elseif strcmp(MAT.label.LowerBodyColor, 'LowerBodyColorGreen') == 1
        lowerCGreen(1, timeIndex) = lowerCGreen(1, timeIndex) + 1;
    elseif strcmp(MAT.label.LowerBodyColor, 'LowerBodyColorBlue') == 1
        lowerCBlue(1, timeIndex) = lowerCBlue(1, timeIndex) + 1;
    elseif strcmp(MAT.label.LowerBodyColor, 'LowerBodyColorPink') == 1
        lowerCPink(1, timeIndex) = lowerCPink(1, timeIndex) + 1;
    elseif strcmp(MAT.label.LowerBodyColor, 'LowerBodyColorPurple') == 1
        lowerCPurple(1, timeIndex) = lowerCPurple(1, timeIndex) + 1;
    elseif strcmp(MAT.label.LowerBodyColor, 'LowerBodyColorBrown') == 1
        lowerCBrown(1, timeIndex) = lowerCBrown(1, timeIndex) + 1;
    elseif strcmp(MAT.label.LowerBodyColor, 'LowerBodyColorGray') == 1
        lowerCGray(1, timeIndex) = lowerCGray(1, timeIndex) + 1;
    elseif strcmp(MAT.label.LowerBodyColor, 'LowerBodyColorBlack') == 1
        lowerCBlack(1, timeIndex) = lowerCBlack(1, timeIndex) + 1;
    elseif strcmp(MAT.label.LowerBodyColor, 'LowerBodyColorWhite') == 1
        lowerCWhite(1, timeIndex) = lowerCWhite(1, timeIndex) + 1;
    elseif strcmp(MAT.label.LowerBodyColor, 'LowerBodyColorUnknown') == 1
        lowerCUn(1, timeIndex) = lowerCUn(1, timeIndex) + 1;
    else
        fprintf("%s have error button\n", Dir(randomID(matIndex)).name);
        %copyfile([matFolder Dir(matIndex).name], 'C:/Users/Lab031/Desktop/LABEL_ATTRIBUTE/input');
    end
    %%%%type%%%%
    if strcmp(MAT.label.LowerBodyType, 'LowerBodyTypeTrousers') == 1
        lowerTT(1, timeIndex) = lowerTT(1, timeIndex) + 1;
    elseif strcmp(MAT.label.LowerBodyType, 'LowerBodyTypePants') == 1
        lowerTP(1, timeIndex) = lowerTP(1, timeIndex) + 1;
    elseif strcmp(MAT.label.LowerBodyType, 'LowerBodyTypeShortSkirt') == 1
        lowerTS(1, timeIndex) = lowerTS(1, timeIndex) + 1;
    elseif strcmp(MAT.label.LowerBodyType, 'LowerBodyTypeLongSkirt') == 1
        lowerTL(1, timeIndex) = lowerTL(1, timeIndex) + 1;
    elseif strcmp(MAT.label.LowerBodyType, 'LowerBodyTypeUnknown') == 1
        lowerTUn(1, timeIndex) = lowerTUn(1, timeIndex) + 1;
    else
        fprintf("%s have error button\n", Dir(randomID(matIndex)).name);
        %copyfile([matFolder Dir(matIndex).name], 'C:/Users/Lab031/Desktop/LABEL_ATTRIBUTE/input');
    end
    %%%%%%%%%%%%carrying%%%%%%%%%%%%
    if size(MAT.label.Carrying, 2) == 0
        CarryNo(1, timeIndex) = CarryNo(1, timeIndex) + 1;
    end
    haveBackpack = 0;
    for carryIndex = 1:size(MAT.label.Carrying, 2)
        if strcmp(MAT.label.Carrying{1, carryIndex}, 'CarryingBackpack') == 1
            carryBack(1, timeIndex) = carryBack(1, timeIndex) + 1;
            haveBackpack = 1;
            %imwrite(MAT.label.image, ['./Mat/Backpack/' fileName '.jpg']);
            %copyfile([matFolder Dir(randomID(matIndex)).name], './Mat/Backpack/mat');
        elseif strcmp(MAT.label.Carrying{1, carryIndex}, 'CarryingSatchel') == 1
            carrySat(1, timeIndex) = carrySat(1, timeIndex) + 1;
            %copyfile([matFolder Dir(matIndex).name], './Mat/Backpack_Satchel_Handbag');
        elseif strcmp(MAT.label.Carrying{1, carryIndex}, 'CarryingHandbag') == 1
            carryHand(1, timeIndex) = carryHand(1, timeIndex) + 1;
            %imwrite(MAT.label.image, ['./Mat/Handbag/' fileName '.jpg']);
            %copyfile([matFolder Dir(randomID(matIndex)).name], './Mat/Handbag/mat');
        elseif strcmp(MAT.label.Carrying{1, carryIndex}, 'CarryingSuitcase') == 1
            carrySuit(1, timeIndex) = carrySuit(1, timeIndex) + 1;
        elseif strcmp(MAT.label.Carrying{1, carryIndex}, 'CarryingUmbrellaOpen') == 1
            carryUmO(1, timeIndex) = carryUmO(1, timeIndex) + 1;
            haveUmbrella = 1;
        elseif strcmp(MAT.label.Carrying{1, carryIndex}, 'CarryingUmbrellaFold') == 1
            carryUmF(1, timeIndex) = carryUmF(1, timeIndex) + 1;
        elseif strcmp(MAT.label.Carrying{1, carryIndex}, 'CarryingPram') == 1
            carryPram(1, timeIndex) = carryPram(1, timeIndex) + 1;
        elseif strcmp(MAT.label.Carrying{1, carryIndex}, 'CarryingBaby') == 1
            carryBaby(1, timeIndex) = carryBaby(1, timeIndex) + 1;
        elseif strcmp(MAT.label.Carrying{1, carryIndex}, 'CarryingWheelchair') == 1
            carryWheel(1, timeIndex) = carryWheel(1, timeIndex) + 1;
        else
            fprintf("%s have error button\n", Dir(randomID(matIndex)).name);
        end
    end
    
    %%%%%%%%%%%%Hat%%%%%%%%%%%%
    try
        if MAT.label.Hat == 1
            hat(1, timeIndex) = hat(1, timeIndex) + 1;
            %copyfile([matFolder Dir(matIndex).name], 'C:/Users/Lab031/Desktop/CHT_2017/Clothing Recognition/Mat/HatHood/Hat/mat');
        elseif MAT.label.Hood == 1
            hood(1, timeIndex) = hood(1, timeIndex) + 1;
            %copyfile([matFolder Dir(matIndex).name], 'C:/Users/Lab031/Desktop/CHT_2017/Clothing Recognition/Mat/HatHood/Hood/mat');
        elseif MAT.label.Hat == 0 && MAT.label.Hood == 0
            nohat(1, timeIndex) = nohat(1, timeIndex) + 1;
        else
            fprintf("%s no Hat Label\n", Dir(randomID(matIndex)).name);
        end
    catch
        %fprintf("%s\n", fileName)
        copyfile([matFolder Dir(matIndex).name], 'C:/Users/Lab031/Desktop/LABEL_ATTRIBUTE/input');
    end
end

fprintf("\n\n\n")
fprintf("DontCare %d, Unlabeled %d\n", dontcare, unlabeled);

for timeIndex = 1:2
    fprintf("*******************************\n");
    if timeIndex == 1
        fprintf("--------NightTIme %5d--------\n", nighttime);
    else
        fprintf("---------DayTime %5d---------\n", daytime);
    end
    fprintf("*******************************\n");
    fprintf("Gender: Male %d, Female %d, Unknown %d\n", genderM(1, timeIndex), genderF(1, timeIndex), genderUn(1, timeIndex));
    fprintf("Age: Child %d, Teen %d, Adult %d, Middle %d, Senior %d, Unknown %d\n", ageC(1, timeIndex), ageT(1, timeIndex), ageA(1, timeIndex), ageM(1, timeIndex), ageS(1, timeIndex), ageUn(1, timeIndex));
    fprintf("UpperColor: Red %d, Orange %d, Yellow %d, Green %d, Blue %d, Pink %d, Purple %d, Brown %d, Gray %d, Black %d, White %d, Unknown %d\n", upperCRed(1, timeIndex), upperCOrange(1, timeIndex), upperCYellow(1, timeIndex), upperCGreen(1, timeIndex), upperCBlue(1, timeIndex), upperCPink(1, timeIndex), upperCPurple(1, timeIndex), upperCBrown(1, timeIndex), upperCGray(1, timeIndex), upperCBlack(1, timeIndex), upperCWhite(1, timeIndex), upperCUn(1, timeIndex));
    fprintf("UpperSleeve: Long %d, Short %d, Unknown %d\n", upperSL(1, timeIndex), upperSS(1, timeIndex), upperSUn(1, timeIndex));
    fprintf("UpperDesign: Plain %d, Plaid %d, Logo %d, Striped %d, Dot %d, Unknown %d\n", upperDPlain(1, timeIndex), upperDPlaid(1, timeIndex), upperDLogo(1, timeIndex), upperDStriped(1, timeIndex), upperDDot(1, timeIndex), upperDUn(1, timeIndex));
    fprintf("LowerColor:  Red %d, Orange %d, Yellow %d, Green %d, Blue %d, Pink %d, Purple %d, Brown %d, Gray %d, Black %d, White %d, Unknown %d\n", lowerCRed(1, timeIndex), lowerCOrange(1, timeIndex), lowerCYellow(1, timeIndex), lowerCGreen(1, timeIndex), lowerCBlue(1, timeIndex), lowerCPink(1, timeIndex), lowerCPurple(1, timeIndex), lowerCBrown(1, timeIndex), lowerCGray(1, timeIndex), lowerCBlack(1, timeIndex), lowerCWhite(1, timeIndex), lowerCUn(1, timeIndex));
    fprintf("LowerType: Trousers %d, Pants %d, ShortSkirt %d, LongSkirt %d, Unknown %d\n", lowerTT(1, timeIndex), lowerTP(1, timeIndex), lowerTS(1, timeIndex), lowerTL(1, timeIndex), lowerTUn(1, timeIndex));
    fprintf("Carry: Backpack %d, Satchel %d, Handbag %d, Suitcase %d, Umbrella(Open) %d, Umbrella(fold) %d, Pram %d, Baby %d, Wheelchair %d NoCarry %d\n", carryBack(1, timeIndex), carrySat(1, timeIndex), carryHand(1, timeIndex), carrySuit(1, timeIndex), carryUmO(1, timeIndex), carryUmF(1, timeIndex), carryPram(1, timeIndex), carryBaby(1, timeIndex), carryWheel(1, timeIndex), CarryNo(1, timeIndex));
    fprintf("Hat: Hat %d, Hood %d, NoHat %d\n", hat(1, timeIndex), hood(1, timeIndex), nohat(1, timeIndex));
end

clear all;
function ConvertPETALabel
    FolderName = 'VIPeR';
    imageFolder = ['./DataPETA/PETA dataset/' FolderName '/archive/'];
    imageDir = [dir([imageFolder '*.bmp']) dir([imageFolder '*.jpg']) dir([imageFolder '*.jpeg']) dir([imageFolder '*.png'])];
    imageNumber = size(imageDir, 1);
    
    labelFid = fopen(['./DataPETA/PETA dataset/' FolderName '/archive/Label.txt'], 'r');
    
    label = [];
    labelNumber = 0;
    upc = 0;
    lpc = 0;
    ups = 0;
    lpt = 0;
    a = 0;
    g = 0;
    number = 0;
    while 1
        line = fgetl(labelFid);
        if ~isstr(line)
            break;
        end
        split = regexp(line, " ", 'split');
        
        upperColor = [];
        upperSleeve = [];
        lowerColor = [];
        lowerType = [];
        umbrella = 'NoUmbrella';
        backpack = 'NoBackpack';
        handbag = 'NoHandbag';
        hat = 'CarryNoHat';
        
        for splitIndex = 2:size(split, 2)
            % upperColor 11 Class
            if strcmp(split{1, splitIndex}, 'upperBodyRed') == 1 
                upperColor = 'UpperBodyColorRed';
                upc = upc + 1;
            elseif strcmp(split{1, splitIndex}, 'upperBodyOrange') == 1
                upperColor = 'UpperBodyColorOrange';
                upc = upc + 1;
            elseif strcmp(split{1, splitIndex}, 'upperBodyYellow') == 1
                upperColor = 'UpperBodyColorYellow';
                upc = upc + 1;
            elseif strcmp(split{1, splitIndex}, 'upperBodyGreen') == 1
                upperColor = 'UpperBodyColorGreen';
                upc = upc + 1;
            elseif strcmp(split{1, splitIndex}, 'upperBodyBlue') == 1
                upperColor = 'UpperBodyColorBlue';
                upc = upc + 1;
            elseif strcmp(split{1, splitIndex}, 'upperBodyPink') == 1
                upperColor = 'UpperBodyColorPink';
                upc = upc + 1;
            elseif strcmp(split{1, splitIndex}, 'upperBodyBrown') == 1
                upperColor = 'UpperBodyColorBrown';
                upc = upc + 1;
            elseif strcmp(split{1, splitIndex}, 'upperBodyGrey') == 1
                upperColor = 'UpperBodyColorGray';
                upc = upc + 1;
            elseif strcmp(split{1, splitIndex}, 'upperBodyBlack') == 1
                upperColor = 'UpperBodyColorBlack';
                upc = upc + 1;
            elseif strcmp(split{1, splitIndex}, 'upperBodyWhite') == 1
                upperColor = 'UpperBodyColorWhite';
                upc = upc + 1;
            end
            
            % lowerColor 11 Class
            if strcmp(split{1, splitIndex}, 'lowerBodyRed') == 1 
                lowerColor = 'LowerBodyColorRed';
                lpc = lpc + 1;
            elseif strcmp(split{1, splitIndex}, 'lowerBodyOrange') == 1
                lowerColor = 'LowerBodyColorOrange';
                lpc = lpc + 1;
            elseif strcmp(split{1, splitIndex}, 'lowerBodyYellow') == 1
                lowerColor = 'LowerBodyColorYellow';
                lpc = lpc + 1;
            elseif strcmp(split{1, splitIndex}, 'lowerBodyGreen') == 1
                lowerColor = 'LowerBodyColorGreen';
                lpc = lpc + 1;
            elseif strcmp(split{1, splitIndex}, 'lowerBodyBlue') == 1
                lowerColor = 'LowerBodyColorBlue';
                lpc = lpc + 1;
            elseif strcmp(split{1, splitIndex}, 'lowerBodyPink') == 1
                lowerColor = 'LowerBodyColorPink';
                lpc = lpc + 1;
            elseif strcmp(split{1, splitIndex}, 'lowerBodyBrown') == 1
                lowerColor = 'LowerBodyColorBrown';
                lpc = lpc + 1;
            elseif strcmp(split{1, splitIndex}, 'lowerBodyGrey') == 1
                lowerColor = 'LowerBodyColorGray';
                lpc = lpc + 1;
            elseif strcmp(split{1, splitIndex}, 'lowerBodyBlack') == 1
                lowerColor = 'LowerBodyColorBlack';
                lpc = lpc + 1;
            elseif strcmp(split{1, splitIndex}, 'lowerBodyWhite') == 1
                lowerColor = 'LowerBodyColorWhite';
                lpc = lpc + 1;
            end
            
            % upperSleeve
            if strcmp(split{1, splitIndex}, 'upperBodyLongSleeve') == 1
                upperSleeve = 'UpperBodySleeveLong';
                ups = ups + 1;
            elseif strcmp(split{1, splitIndex}, 'upperBodyShortSleeve') == 1 || strcmp(split{1, splitIndex}, 'upperBodyNoSleeve') == 1
                upperSleeve = 'UpperBodySleeveShort';
                ups = ups + 1;
            end
            
            % lowerType
            if strcmp(split{1, splitIndex}, 'lowerBodyTrousers') == 1
                lowerType = 'LowerBodyTypeTrousers';
                lpt = lpt + 1;
            elseif strcmp(split{1, splitIndex}, 'lowerBodyHotPants') == 1 || strcmp(split{1, splitIndex}, 'lowerBodyShorts') == 1
                lowerType = 'LowerBodyTypePants';
                lpt = lpt + 1;
            elseif strcmp(split{1, splitIndex}, 'lowerBodyShortSkirt') == 1
                lowerType = 'LowerBodyTypeShortSkirt';
                lpt = lpt + 1;
            elseif strcmp(split{1, splitIndex}, 'lowerBodyLongSkirt') == 1
                lowerType = 'LowerBodyTypeLongSkirt';
                lpt = lpt + 1;
            end
            
            if strcmp(split{1, splitIndex}, 'carryingBackpack') == 1
                % Backpack
                backpack = backpack(3:end);
            elseif strcmp(split{1, splitIndex}, 'carryingPlasticBags') == 1
                % Handbag
                handbag = handbag(3:end);
            elseif strcmp(split{1, splitIndex}, 'carryingUmbrella') == 1
                % Umbrella
                umbrella = umbrella(3:end);
            elseif strcmp(split{1, splitIndex}, 'accessoryHat') == 1
                % Hat
                hat = 'CarryHat';
            end
            
            
            if strcmp(split{1, splitIndex}, 'personalMale') == 1
                gender = 'GenderMale';
                g = g + 1;
            elseif strcmp(split{1, splitIndex}, 'personalFemale') == 1
                gender = 'GenderFemale';
                g = g + 1;
            end
            
            
            if strcmp(split{1, splitIndex}, 'personalLess15') == 1
                age = 'AgeChild';
                a = a + 1;
            elseif strcmp(split{1, splitIndex}, 'personalLess30') == 1 || strcmp(split{1, splitIndex}, 'personalLess45') == 1 || strcmp(split{1, splitIndex}, 'personalLess60') == 1 
                age = 'AgeAdult';
                a = a + 1;
            elseif strcmp(split{1, splitIndex}, 'personalLarger60') == 1
                age = 'AgeSenior';
                a = a + 1;
            end
        end
        
        if size(upperColor, 1) ~= 0 && size(upperSleeve, 1) ~= 0 && size(lowerColor, 1) ~= 0 && size(lowerType, 1) ~= 0 && size(gender, 1) ~= 0 && size(age, 1) ~= 0
            labelNumber = labelNumber + 1;
            label(labelNumber).index = split{1, 1};
            label(labelNumber).upperColor = upperColor;
            label(labelNumber).upperSleeve = upperSleeve;
            label(labelNumber).lowerColor = lowerColor;
            label(labelNumber).lowerType = lowerType;
            label(labelNumber).backpack = backpack;
            label(labelNumber).handbag = handbag;
            label(labelNumber).umbrella = umbrella;
            label(labelNumber).hat = hat;
            label(labelNumber).gender = gender;
            label(labelNumber).age = age;
        end
        clear upperColor upperSleeve lowerColor lowerType;
    end
    fclose(labelFid);
    fprintf("Label Number %d\n", labelNumber);
    fprintf("%d %d %d %d %d %d\n", upc, lpc, ups, lpt, a, g);
    
    matchNumber = 0;
    for imageIndex = 1:imageNumber
        imageName = imageDir(imageIndex).name;
        img = imread([imageFolder imageName]);
        match = regexp(imageName, "_", 'split');
        matchName = match{1, 1};
        fprintf("%s %s\n", imageName, matchName);
        
        for labelIndex = 1:labelNumber
            if strcmp(matchName, label(labelIndex).index) == 1
                fprintf("%s Match\n", imageName);
                split = regexp(imageName, "\.", 'split');
                txtName = [split{1,1} '.txt'];
                txtFid = fopen(['./Data/PETA_All_Morning/label/' txtName], 'w');
                fprintf(txtFid, "UpperBodyColor : %s\n", label(labelIndex).upperColor);
                fprintf(txtFid, "UpperBodySleeve : %s\n", label(labelIndex).upperSleeve);
                fprintf(txtFid, "LowerBodyColor : %s\n", label(labelIndex).lowerColor);
                fprintf(txtFid, "LowerBodyType : %s\n", label(labelIndex).lowerType);
                fprintf(txtFid, "Umbrella : %s\n", label(labelIndex).umbrella);
                fprintf(txtFid, "Hat : %s\n", label(labelIndex).hat);
                fprintf(txtFid, "Backpack : %s\n", label(labelIndex).backpack);
                fprintf(txtFid, "Handbag : %s\n", label(labelIndex).handbag);
                fprintf(txtFid, "Age : %s\n", label(labelIndex).age);
                fprintf(txtFid, "Gender : %s\n", label(labelIndex).gender);
                
                matchNumber = matchNumber + 1;
                imwrite(img, ['./Data/PETA_All_Morning/image/' imageName]);
                fclose(txtFid);
                continue;
            end
        end
    end
    fprintf("Match Number %d %d\n", matchNumber, imageNumber);
end
function ConvertLabel
    DATA_NAME = 'CHT_ALL_Atribute_Night_New';
    DATA_PATH = ['./DATA/' DATA_NAME];
    IMG_PATH = [DATA_PATH '/image/'];
    MASK_PATH = [DATA_PATH '/mask/'];
    MASKIMG_PATH = [DATA_PATH '/maskimage/'];
    LABEL_PATH = [DATA_PATH '/label/'];
    
    MAT_PATH = './Mat/ALL/';
    MAT_DIR = dir(MAT_PATH);
    MAT_NUMBER = size(MAT_DIR, 1);
    
    for MAT_INDEX = 1:MAT_NUMBER
        matName = MAT_DIR(MAT_INDEX).name;
        if matName == "." || matName == ".."
            continue;
        end
        
        load([MAT_PATH matName]);
        
        if label.DontCare == 1
            continue;
        end
        
        % Skip Unknown Label
%         if strcmp(label.Time, 'TimeNighttime') == 1
%             continue;
%         end
        if strcmp(label.Time, 'TimeDaytime') == 1
            continue;
        end
        
        if strcmp(label.Gender, 'GenderUnknown') == 1 || strcmp(label.Age, 'AgeUnknown') == 1
            continue;
        end
        
%         if strcmp(label.UpperBodyColor, 'UpperBodyColorUnknown') == 1 || strcmp(label.LowerBodyColor, 'LowerBodyColorUnknown') == 1
%             continue;
%         end
        
        if strcmp(label.UpperBodySleeve, 'UpperBodySleeveUnknown') == 1 || strcmp(label.LowerBodyType, 'LowerBodyTypeUnknown') == 1
            continue;
        end
        
        % Skip Hat Type is Hood
        try
            if label.Hood == 1
                continue;
            end
        catch
            continue
        end
        fprintf("%s\n", matName);
        
        fileName = matName(1:end-4);
        jpgName = [fileName '.jpg'];
        labelName = [fileName '.txt'];
        
        % Save Image, Mask, MaskImage
        imwrite(label.image, [IMG_PATH jpgName]);
        imwrite(label.mask, [MASK_PATH jpgName]);
        maskimg(:,:,1) = label.image(:,:,1) .* uint8(label.mask);
        maskimg(:,:,2) = label.image(:,:,2) .* uint8(label.mask);
        maskimg(:,:,3) = label.image(:,:,3) .* uint8(label.mask);
        imwrite(maskimg, [MASKIMG_PATH jpgName]);
        
        umbrella = 'NoUmbrella';
        backpack = 'NoBackpack';
        handbag = 'NoHandbag';
        % Carry
        carryNumber = size(label.Carrying, 2);
        for carryIndex = 1:carryNumber
            if strcmp(label.Carrying{1, carryIndex}, 'CarryingBackpack') == 1
                backpack = backpack(3:end);
            elseif strcmp(label.Carrying{1, carryIndex}, 'CarryingHandbag') == 1
                handbag = handbag(3:end);
            elseif strcmp(label.Carrying{1, carryIndex}, 'CarryingUmbrellaOpen') == 1
                umbrella = umbrella(3:end);
            end
        end
        
        % Hat
        try
            hat = 'CarryNoHat';
            if label.Hat == 1
                hat = 'CarryHat';
            elseif label.Hood == 1
                hat = 'CarryHood';
            end
        catch
            continue
        end
        
        % Write Txt
        labelFid = fopen([LABEL_PATH labelName], 'w');
        
        % Clothes
        fprintf(labelFid, "UpperBodyColor : %s\n", label.UpperBodyColor);
        fprintf(labelFid, "UpperBodySleeve : %s\n", label.UpperBodySleeve);
        fprintf(labelFid, "LowerBodyColor : %s\n", label.LowerBodyColor);
        fprintf(labelFid, "LowerBodyType : %s\n", label.LowerBodyType);
        
        % Carry
        fprintf(labelFid, "Umbrella : %s\n", umbrella);
        fprintf(labelFid, "Hat : %s\n", hat);
        fprintf(labelFid, "Backpack : %s\n", backpack);
        fprintf(labelFid, "Handbag : %s\n", handbag);
        
        % Age
        fprintf(labelFid, "Age : %s\n", label.Age);
        % Gender
        fprintf(labelFid, "Gender : %s\n", label.Gender);
        
        fclose(labelFid);
        clear label;
    end
end
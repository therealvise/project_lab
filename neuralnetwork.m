folder = '/Users/lucadisimone/Desktop/Lab_Multimedialità/54339_104884_bundle_archive/HAM10000_images_part_2';
myImages = imageSet(folder);

imds = imageDatastore(folder, 'LabelSource', 'foldernames', 'IncludeSubfolders',true);

% Find the first instance of an image for each category
nv = find(imds.Labels == 'HAM10000_images_part_2', 1);

figure
imshow(readimage(imds,nv))

numTrainFiles = 8012;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');

inputSize = [450 600 3];
numClasses = 1;

layers = [
    imageInputLayer(inputSize)
    convolution2dLayer(5,20)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'MaxEpochs',4, ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(imdsTrain,layers,options);



% Importing Dataset after download it by Kaggle Website
data = readtable('HAM10000_metadata.csv');
head(data,5)

% Statistics about counting for each sensible column
G1 = groupcounts(data,'age');
disp(G1);

G2 = groupcounts(data,'sex');
disp(G2);


G3 = groupcounts(data,'localization');
disp(G3);

% Look about missing values after scan rows
vars = data.Properties.VariableNames;
figure
imagesc(ismissing(data))
ax = gca;
ax.XTick = 1:12;
ax.XTickLabel = vars;
ax.XTickLabelRotation = 90;
title('Missing Rows Values')

avgAge = mean(data.age);
disp(avgAge)

figure
histogram(data.age);

data.sex = categorical(data.sex, {'male' 'female'});

figure
histogram(data.sex)
title('Gender Percentage')

data.localization = categorical(data.localization, {'abdomen' 'acral' 'back' 'chest' 'ear' 'face' 'foot' 'genital' 'hand' 'lower extremity' 'neck' 'scalp' 'trunk' 'unknown' 'upper extremity'});
                                                    
                                                    
                                                    
figure
histogram(data.localization)
title('Localization Percentage')   

[m,n] = size(data);
P = 0.80 ;
idx = randperm(m)  ;
Training = data(idx(1:round(P*m)),:) ; 
Testing = data(idx(round(P*m)+1:end),:) ;

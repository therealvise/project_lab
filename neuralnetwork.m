net = alexnet;
analyzeNetwork(net)
inputSize = net.Layers(1).InputSize;

folder_train = '/Users/gianlucavisentin/Desktop/monkeys_project/data_train';
myImages_train = imageSet(folder_train);

folder_validation = '/Users/gianlucavisentin/Desktop/monkeys_project/data_val';
myImages_val = imageSet(folder_validation);

imdsTrain = imageDatastore(folder_train, 'LabelSource', 'foldernames', 'IncludeSubfolders',true);
imdsValidation = imageDatastore(folder_validation, 'LabelSource', 'foldernames', 'IncludeSubfolders',true);
%imdsValidation = imageDatastore('validation', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
[trainingImages, testImages] = splitEachLabel(imdsTrain, 0.8, 'randomize');
layers = net.Layers;

layersTransfer = net.Layers(1:end-3);


numClasses = 10;

layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),trainingImages, ...
    'DataAugmentation',imageAugmenter);

augimdsValidation = augmentedImageDatastore(inputSize(1:2),testImages);

options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');

netTransfer = trainNetwork(augimdsTrain,layers,options);

[YPred,scores] = classify(netTransfer,augimdsValidation);

YValidation = testImages.Labels;
accuracy = mean(YPred == YValidation)


idx = randperm(numel(testImages.Files),12);
figure
for i = 1:12
    subplot(4,4,i)
    I = readimage(testImages,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label));
end

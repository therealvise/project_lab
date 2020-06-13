% Script con lo scopo di addestrare una rete neurale sulla base di un
% dataset contenente circa 1300 immagini, divise in base alla classe di
% appartenenza. Il dataset è disponibile presso il link:
% https://www.kaggle.com/slothkong/10-monkey-species

% La scelta è ricaduta su una rete AlexNet, di tipo convoluzionale,
% ampiamente utilizzata nel riconoscimento delle immagini
net = alexnet;
analyzeNetwork(net)
inputSize = net.Layers(1).InputSize;

% Import immagini da locale. In caso di esecuzione su una macchina diversa,
% cambiare il percorso
folder_train = '/Users/nomeutente/Desktop/monkeys_project/data_train';
myImages_train = imageSet(folder_train);

folder_validation = '/Users/nomeutente/Desktop/monkeys_project/data_val';
myImages_val = imageSet(folder_validation);

imdsTrain = imageDatastore(folder_train, 'LabelSource', 'foldernames', 'IncludeSubfolders',true);
imdsValidation = imageDatastore(folder_validation, 'LabelSource', 'foldernames', 'IncludeSubfolders',true);

% Training:Test split 80-20
[trainingImages, testImages] = splitEachLabel(imdsTrain, 0.8, 'randomize');
layers = net.Layers;

layersTransfer = net.Layers(1:end-3);


numClasses = 10;
% Definizione dei layers convoluzionali 
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

pixelRange = [-30 30];

% Data augmentation viene introdotto per migliorare i risultati ed evitare
% il fenomeno dell'overfitting, ovvero quando un modello statistico
% complesso si adatta ai dati di campione
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),trainingImages, ...
    'DataAugmentation',imageAugmenter);

augimdsValidation = augmentedImageDatastore(inputSize(1:2),testImages);

% Nel caso di un'esecuzione più veloce si può diminuire il numero di
% epoche, riducendo le iterazioni
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

% Print di un esempio per ogni immagine, appartenente ad una classe
idx = randperm(numel(testImages.Files),12);
figure
for i = 1:12
    subplot(4,4,i)
    I = readimage(testImages,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label));
end

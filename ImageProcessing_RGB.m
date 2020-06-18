
% In questo script sono state eseguite delle operazione sul Dataset "Monkeys species".
% Le immagini del Dataset di partenza sono modificate applicando ridimensionamento, rumore Gaussiano e 
% una rete neurale pre-allenata per la riduzione del rumore.
% Nei vari casi le immagini sono state salvate in tre directory distinte per poter essere analizzate separatamente.

% Percorso della cartella dove sono presenti le immagini della i-esima specie di scimmia del Dataset 
path_directory='/Users/nomeUtente/Desktop/monkeys_project/training/training/n0';
original_files=dir([path_directory '/*.jpg']);

% Ciclo per scorrere tutte le immagini della cartella per k che va da 1 fino all'ultimo elemento della cartella
% Ad ogni immagine è stato applicato nell'ordine: ridimensionamento, rumore Gaussiano e 
% rete neurale pre-allenata per la riduzione del rumore.
for k=1:length(original_files)
    filename=[path_directory '/' original_files(k).name];
    pristineRGB = imread(filename);
    pristineRGB = im2double(pristineRGB);
    J = imresize(pristineRGB, 0.5);
    destination='/Users/nomeUtente/Desktop/ResizeRGB/ResizeRGB';
    imwrite(J,[destination,num2str(k),'.jpg']);
    noisyRGB = imnoise(J,'gaussian',0,0.01);
    destination='/Users/nomeUtente/Desktop/NoiseRGBr/NoiseRGBr';
    imwrite(noisyRGB,[destination,num2str(k),'.jpg']);
    [noisyR,noisyG,noisyB] = imsplit(noisyRGB);
    net = denoisingNetwork('dncnn');
    denoisedR = denoiseImage(noisyR,net);
    denoisedG = denoiseImage(noisyG,net);
    denoisedB = denoiseImage(noisyB,net);
    denoisedRGB = cat(3,denoisedR,denoisedG,denoisedB);
    noisyPSNR = psnr(noisyRGB,J);
    denoisedPSNR = psnr(denoisedRGB,J);
    noisySSIM = ssim(noisyRGB,J);
    denoisedSSIM = ssim(denoisedRGB,J);
    destination='/Users/nomeUtente/Desktop/RemoveNoiseRGBr/RemoveNoiseRGBr';
    imwrite(denoisedRGB,[destination,num2str(k),'.jpg']);
end


% In questo script sono state eseguite delle operazione sul Dataset "Monkeys species".
% Le immagini del Dataset di partenza sono modificate applicando rumore di tipo Salt and Pepper e 
% un filtro di Wiener per la riduzione del rumore.
% In entrambi i casi le immagini sono state salvate in due directory distinte per poter essere analizzate separatamente.

% Percorso della cartella dove sono presenti le immagini della i-esima specie di scimmia del Dataset 
path_directory='/Users/nomeUtente/Desktop/monkeys_project/training/training/n0';
original_files=dir([path_directory '/*.jpg']);

% Ciclo per scorrere tutte le immagini della cartella per k che va da 1 fino all'ultimo elemento della cartella
% Ad ogni immagine è stato applicato nell'ordine: rumore Salt and Pepper e 
% filtro di Wiener per la riduzione del rumore.
for k=1:length(original_files)
    filename=[path_directory '/' original_files(k).name];
    I = imread(filename)
    J = imnoise(I,'salt & pepper', 0.005);
    [noisyR,noisyG,noisyB] = imsplit(J);
    noisedRGB = cat(3,noisyR,noisyG,noisyB);
    destination='/Users/nomeUtente/Desktop/SaltPepperRGB/n0/';
    imwrite(noisedRGB,[destination,num2str(k),'.jpg']);
    I = rgb2gray(I);
    x = wiener2(J(:,:,1), [8 8]);
    y = wiener2(J(:,:,2), [8 8]);
    z = wiener2(J(:,:,3), [8 8]);
    denoisedRGB = cat(3,x,y,z);
    destination='/Users/nomeUtente/Desktop/DenoisedSaltPepperRGB/n0/';
    imwrite(denoisedRGB,[destination,num2str(k),'.jpg']);
end


% Questo script è relativo alla modifica del Dataset.
% Le immagini del Dataset di partenza sono modificate applicando rumore Gaussiano e filtro di Wiener.
% In entrambi i casi le immagini sono state salvate in due directory distinte per poter essere analizzate separatamente.

% Percorso della cartella dove sono presenti le immagini della i-esima specie di scimmia del Dataset
% In caso di esecuzione su una macchina diversa, cambiare il percorso
path_directory='/Users/nomeUtente/Desktop/monkeys_project/training/training/n0'; 
original_files=dir([path_directory '/*.jpg']);

% Ciclo per scorrere tutte le immagini della cartella per k che va da 1 fino all'ultimo elemento della cartella
% Ad ogni immagine è stato applicato nell'ordine: conversione da RGB a scala di grigi, rumore Gaussiano e filtro di Wiener.
for k=1:length(original_files)
    filename=[path_directory '/' original_files(k).name];
    R = imread(filename)
    I = rgb2gray(R);
    J = imnoise(I,'gaussian',0,0.025);
    destination='/Users/nomeUtente/Desktop/NoiseGrayScale/n0/'; 
    imwrite(J,[destination,num2str(k),'.jpg']);
    K = wiener2(J,[5 5]);
    destination='/Users/nomeUtente/Desktop/WienerGrayScale/n0/'; 
    imwrite(K,[destination,num2str(k),'.jpg']);
end

path_directory='/Users/bigdata/Desktop/monkeys_project/data_train/n9'; 
original_files=dir([path_directory '/*.jpg']);
for k=1:length(original_files)
    filename=[path_directory '/' original_files(k).name];
    I = imread(filename)
    J = imnoise(I,'salt & pepper', 0.005);
    [noisyR,noisyG,noisyB] = imsplit(J);
    I = rgb2gray(I);
    x = wiener2(J(:,:,1), [8 8]);
    y = wiener2(J(:,:,2), [8 8]);
    z = wiener2(J(:,:,3), [8 8]);
    grayscale = cat(3,x,y,z);
    denoisedRGB = cat(3,noisyR,noisyG,noisyB);
    destination='/Users/bigdata/Desktop/prova1/n9_removenoise/';
    imwrite(grayscale,[destination,num2str(k),'.jpg']);
end

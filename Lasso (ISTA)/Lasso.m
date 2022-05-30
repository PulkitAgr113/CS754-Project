% Seed set for reproducibility
rng(1);

% Reading the image file, and converting the values to double
image = imread("celebAtest/182638.jpg");
image = rgb2gray(image);

targetSize = [64 64];
r = centerCropWindow2d(size(image),targetSize);
x = imcrop(image, r);

x = double(reshape(x, [], 1));
n = size(x,1);
m = 1000;
A = normrnd(0, 1/m, m, n);

y = A * x;
lambda = 0.1;
tar_gap = 1e-6;
x_recon = l1_ls(A, y, lambda, tar_gap, false);
imshow(uint8(reshape(x_recon, 64, 64)));

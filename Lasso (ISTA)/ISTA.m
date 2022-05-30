% Seed set for reproducibility
rng(4);

% Reading the image file, and converting to grayscale
image = imread("celebAtest/182650.jpg");
image = rgb2gray(image);

% Cropping central 64x64 square, and converting to double
targetSize = [64 64];
r = centerCropWindow2d(size(image),targetSize);
x = double(imcrop(image, r));

% Dimension of one side of the image
N = 64;

% 1D DCT matrix of size 8x8
DCT = dctmtx(8);

% Taking kronecker product and its transpose to find the 2D 64x64 DCT matrix
DCT = kron(DCT, DCT);
DCT = DCT';

% Matrix to store the reconstructed value for the NxN original image
recon = zeros(N, N);

% Matrix to store the number of times a reconstructed value for each
% index was calculated using the ISTA algorithm
occur = zeros(N, N);

% Measurement matrix of size 32x64
phi = randn (32, 64);

% The measurement matrix is a 32x64 random matrix
% So the compressed measurements are of size 32x1
A = phi * DCT;

% Parameters used in the ISTA algorithm (set optimally)
alpha = 1 + eigs(A' * A, 1);
lambda = 1;

% Dividing the image into 8x8 patches, and reconstructing each patch
for i = 1:N-7
    for z = 1:N-7
        
        % Theta can be randomly initialized, and so it is set to 0
        theta = zeros(64, 1);
        
        % Corresponding vectorized patch of the measured image
        y_small = phi * reshape(x(i: i+7, z: z+7), 64, 1);
        
        % Running the ISTA algorithm for 50 iterations (gives good results)
        for iterations = 1:50
            % soft function has been implemented at the end of this file
            theta = soft(theta + (1/alpha) * A' * (y_small - A * theta), lambda / (2 * alpha));
        end
        
        % Using the found theta to make reconstructed 8x8 patch
        x_recon = DCT * theta;
        x_recon = reshape(x_recon, 8, 8);

        % Adding the reconstructed patch over the patch found till now
        % Average will be taken after all patches have been worked upon
        for p = 0:7
            for q = 0:7
                recon(i+p, z+q) = recon(i+p, z+q) + x_recon(p+1, q+1);
                occur(i+p, z+q) = occur(i+p, z+q) + 1;
            end
        end

    end
end

% Taking average of reconstructions over all overlapping pixels
recon = recon ./ occur;

% Relative mean squared error between reconstructed and original image
mse = sqrt(sum((recon(:) - x(:)).^2, 'all') / sum(x(:).^2, 'all'));
disp('Root Mean Squared Error is ' + string(mse));

% Saving the reconstructed images
fig = figure();
hold on

subplot(1,2,1);
imshow(x,[]);
title('Original Image');

subplot(1,2,2);
imshow(recon,[]);
title('Reconstructed Image');

saveas(fig, 'reconstructed_images/182650.png');

% soft function (used in ISTA algorithm)
% Gives solution to x in y = x + lambda * sgn(x)
% where sgn(x) is the signum function, applied element-wise
function x = soft(y, lambda)

    x = zeros(size(y));
    
    for i = 1:size(y)
        if y(i)>=lambda
            x(i) = y(i)-lambda;
        elseif y(i)<=-lambda
            x(i) = y(i)+lambda;
        else
            x(i) = 0;
        end
    end
    
end

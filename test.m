% test.m

clear;close all;clc

load('mnist.mat');
load('w.mat');

test_img = (test_img-0.5)./2;
test_img = reshape(test_img, 28, 28, []);

solver = caffe.Solver('ipsolver.prototxt');
solver.net.copy_from('tmp.caffemodel');

solver.net.reshape_as_input({zeros(28,28,1,1)});
test_size = 10000;
right = 0;
wnorm = sqrt(sum(w.^2, 1)); % 1x10
for ii = 1:test_size
    solver.net.forward({test_img(:, :, ii)});
    label = test_label(ii);
    x = solver.net.blobs('relu3').get_data();
    xnorm = sqrt(sum(x.^2, 1)); % 1xn
    theta = (w'*x)./(wnorm'*xnorm); % 10xn
    k = floor(theta.*2./pi); % 10xn
    
    f = (-1).^k.*2.*(w'*x).^2./(wnorm'*xnorm) - ...
        (2.*k+(-1).^k).*(wnorm'*xnorm); % 10xn
    
    [~, ind] = max(f);
    if ind == label+1
        right = right+1;
    end
end

acc = right./test_size
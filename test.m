% test.m

clear;close all;clc

caffe.reset_all
caffe.set_mode_gpu
caffe.set_device(0)

load('mnist.mat');
load('w.mat');

test_img = (test_img-0.5).*2;
test_img = reshape(test_img, 28, 28, []);

net = caffe.Net('test.prototxt', 'train');
net.copy_from('tmp.caffemodel');

net.reshape_as_input({zeros(28,28,1,1)});
test_size = 10000;
right = 0;
wnorm = sqrt(sum(w.^2, 1)); % 1x10
for ii = 1:test_size
    net.forward({test_img(:, :, ii)});
    label = test_label(ii);
    x = net.blobs('bn_ip').get_data();
    xnorm = sqrt(sum(x.^2, 1)); % 1xn
    theta = acos((w'*x)./(wnorm'*xnorm)); % 10xn
    
    f = wnorm'*xnorm.*fi(theta, 2);
    
    [~, ind] = max(f);
    if ind == label+1
        right = right+1;
    end
end

acc = right./test_size
% train.m
% Author: Taotao
% Time: 20180102

clear;close all;clc
dbstop if error

load('mnist.mat');

train_img = (train_img-0.5)./2;
train_img = reshape(train_img, 28, 28, []);

rng('shuffle');

batch_size = 20;
max_iter = 50000;
lr_rate = 0.003;
gamma = 0.0003;
power = 0.75;

solver = caffe.Solver('ipsolver.prototxt');
w = rand(1024, 10)./100;

figure(1);
hold on

for ii = 1:max_iter
    inputs = processinputs(train_img, train_label, batch_size);
    solver.net.forward({inputs.batch_img});
    label = inputs.batch_label;
    x = solver.net.blobs('relu3').get_data();
    [loss, dw, dx] = large_margin_softmax(x, w, label);
    if mod(ii, 10) == 0
        fprintf('%d / %d loss: %g \n', ii, max_iter, loss);
        plot(ii, loss, 'rx');
        pause(0.0001);
    end
    rate_now = lr_rate*(1+ii*gamma).^(-power);
    w = w - dw*rate_now;
    solver.net.blobs('relu3').set_diff(dx);
    solver.net.backward_prefilled();
    solver.update(single(rate_now));
end
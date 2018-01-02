% train.m
% Author: Taotao
% Time: 20180102

clear;close all;clc

load('mnist.mat');

train_img = (train_img-0.5)./2;
train_img = reshape(train_img, 28, 28, []);

rng('shuffle');

batch_size = 20;
max_iter = 50000;
lr_rate = 0.001;
gamma = 0.0003;
power = 0.75;

m = 2;


solver = caffe.Solver('ipsolver.prototxt');


for ii = 1:max_iter
    inputs = processinputs(train_img, train_label, batch_size);
    solver.net.forward({inputs.batch_img});
    label = inputs.batch_label;
    
end
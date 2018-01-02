% process.m

function inputs = processinputs(image, label, batch_size)
    rdp = randperm(length(label), batch_size);
    batch_img = image(:, :, rdp);
    inputs.batch_img = permute(batch_img, [1, 2, 4, 3]);
    batch_label = label(rdp);
    inputs.batch_label = zeros(batch_size, 10);
    for ii = 1:batch_size
        inputs.batch_label(ii, batch_label(ii)+1) = 1;
    end
    
end
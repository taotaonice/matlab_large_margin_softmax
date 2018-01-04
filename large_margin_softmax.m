% large_margin_softmax.m

function [loss, dw, dx] = large_margin_softmax(x, w)
    % margin is 2
    % x:1024xn  w:1024x2
    dw = zeros(size(w));
    dx = zeros(size(x));
    
    xnorm = sqrt(sum(x.^2, 1)); % 1xn
    wnorm = sqrt(sum(w.^2, 1)); % 1x2
    theta = (w'*x)./(wnorm'*xnorm); % 2xn
    k = floor(theta.*2./pi); % 2xn
    
    f = (-1).^k.*2.*(w'*x).^2./(wnorm'*xnorm) - ...
        (2.*k+(-1).^k).*(wnorm'*xnorm); % 2xn
    
    s = sum(exp(w'*x), 1);
    s = repmat(s, 2, 1);
    s = s - exp(w'*x);
    loss = exp(f)./(exp(f) + s);
    loss = sum(loss(:)) ./ size(w, 2);
    
    dfdx = 4.*w*((-1).^k.*wnorm*xnorm./(wnorm'*xnorm)) - ...
            2.*((-1).^k.*(w'*x).^2./)
    
end
% fi.m

function res = fi(theta, m)
    k = floor(theta*m./pi);
    res = (-1).^k * cos(m*theta) - 2*k;
end
function [QQ] = fixedRank(AA, kk, pp)
    ll = kk + pp;
    [mm, nn] = size(AA);

    WW = randn(nn, ll) + 1i * randn(nn, ll);      % draw from gaussian distribution
    YY = AA * WW;                               % samples from range(AA)
    [QQ, ~] = qr(YY, 'econ');                    % orthonormalize range
end
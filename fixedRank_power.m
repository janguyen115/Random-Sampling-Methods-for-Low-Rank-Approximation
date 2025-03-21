function [QQ] = fixedRank_power(AA, kk, pp, qq)
    ll = kk + pp;
    [mm, nn] = size(AA);

    WW = randn(nn, ll) + 1i * randn(nn, ll);      % draw from gaussian distribution
    YY = AA * WW;                                 % samples from range(AA)
    [QQ, ~] = qr(YY, 'econ');

    for i = 0:qq
        [QQ, ~] = qr(AA' * QQ, 'econ');
        [QQ, ~] = qr(AA * QQ, 'econ');      % orthonormalize, exponentiate singular value decay
    end
end
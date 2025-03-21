function [QQ] = fixedPrecision(AA, rr, tol)
    [mm, nn] = size(AA);
    II = eye(mm, nn);

    WW = randn(nn, rr) + 1i * randn(nn, rr);     % draw from gaussian distribution
    YY = AA * WW;                               % samples from range(AA)
    j = 0;                                      
    QQ = zeros(mm, 0);                          % initialize QQ

    while max(vecnorm(YY(:,j+1:j+rr), 2, 1)) > (tol / 10 * sqrt(2 / pi))     % bound current approximation
        j = j + 1;
        YY(:, j) = (II - QQ * QQ') * YY(:,j);               % orthogonalize sample to range(QQ'A)
        qq = YY(:,j) / norm(YY(:, j));                      % normalize current range sample
        QQ = [QQ qq];                                       % append basis vector
        WW = [WW (randn(nn, 1) + 1i * randn(nn, 1))];       % draw additional gaussian vector
        YY = [YY ((II - QQ * QQ') * AA * WW(:, j + rr))];   % add to range samples 
        for i = j+1 : j+rr-1
            YY(:, i) = YY(:, i) - qq * dot(qq, YY(:, i));   % orthogonalize next rr many samples to range(QQ'A)
        end
    end   
end
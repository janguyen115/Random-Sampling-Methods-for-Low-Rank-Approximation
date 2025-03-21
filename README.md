To demo randomizedLowRank.m, simply run the following command to reproduce the experiments in the paper:

randomizedLowRank % may take ~10 seconds


To test each function, initialize dimensions and generate a 200x200 discretized Laplace integral:

e.g. % fixed-rank problem
    mm = 200; nn = 200; kk = 20; pp = 5;
    AA = discretizeLaplace(mm, nn);
    QQ = fixedRank(AA, kk, pp);  
    fprintf("Fixed-Rank Approximation Error: %.3e\n", norm(AA - QQ * QQ' * AA)) % output approximation error

e.g. % fixed-rank with power iteration
    mm = 200; nn = 200; kk = 20; pp = 5; qq = 5;
    AA = discretizeLaplace(mm, nn);    % can be substituted for another matrix with slower singular value decay rate
    QQ = fixedRank_power(AA, kk, pp, qq);
    fprintf("Fixed-Rank Approximation Error: %.3e\n", norm(AA - QQ * QQ' * AA))

e.g. % fixed-precision problem
    mm = 200; nn = 200; rr = 10; tol = 1e-10;
    AA = discretizeLaplace(mm, nn);
    QQ = fixedPrecision(AA, rr, tol);
    fprintf("Fixed-Precision Approximation Error: %.3e\n", norm(AA - QQ * QQ' * AA))
    fprintf("Rank of Approximate Basis Q: %i\n", size(QQ, 2))

e.g. % compare errors
    [errA, errSVD] = compareQQA(AA, QQ);
    fprintf("Approximation Error ||A - QQ*A||: %.3e \n SVD Approximation Error ||A - QUSV'||: %.3e\n", errA, errSVD)
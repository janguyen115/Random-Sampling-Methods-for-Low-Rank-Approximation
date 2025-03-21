function [] = main()
    % set seed for reproducibility
    rng(444)

    % set parameters for experimentation
    mm = 200; nn = 200; kk = 20; pp = 5; qq = 5; rr = 10; tol = 1e-10;

    % generate input matrix
    AA = discretizeLaplace(mm, nn);

    %%% fixed-rank problem
    % QQ_fr = fixedRank(AA, kk, pp);

    pp = 2;
    errAA = zeros(150,0);
    errSVD = zeros(150, 0);
    XX = 1:150;
    for kk = XX
        QQ_fr = fixedRank(AA, kk, pp);
        [errAA(kk), errSVD(kk)] = compareQQA(AA, QQ_fr);
    end
    
    colors = [0, 114, 178;  % Blue
          230, 159, 0] / 255;  % Orange

    figure; 
    subplot(1, 2, 1);
    hold on;
    plot(XX, errAA, '*', 'Color', colors(1,:), 'LineWidth', 1, 'MarkerSize', 5, 'DisplayName', "||A - QQ*A||");
    plot(XX, errSVD, 'Color', colors(2,:), 'LineWidth', 1, 'MarkerSize', 5, 'DisplayName', "||A - QUSV*||");
    xlabel('l = k + p, "Desired Rank(Q) + Oversampling"');
    ylabel("Log Scale Approximation Error");
    set(gca, 'Yscale', 'log');
    title("Fixed Rank k vs. Approximation Error");
    legend('Location', 'best')
    grid on;

    [~, s, ~] = svd(AA);
    sigmas = diag(s);

    subplot(1, 2, 2);
    plot(1:150, sigmas(1:150,:))
    set(gca, 'Yscale', 'log')
    xlabel('Index (i)')
    ylabel('\Sigma_i_i');
    ylim([1e-17 Inf]); 
    title('Singular Value Decay (\Sigma)')
    grid on;

    %%% fixed-precision problem
    % QQ_fp = fixedPrecision; 

    tol = 1e-10;
    errAA_fp = zeros(15, 0);
    errSVD_fp = zeros(15, 0);

    XX = 1:50;
    for rr = XX
        QQ_fp = fixedPrecision(AA, rr, tol);
        [errAA_fp(rr), errSVD_fp(rr)] = compareQQA(AA, QQ_fp);
    end

    colors = [0, 158, 115;  % Green
          204, 121, 167] / 255; % Purple

    figure; 
    subplot(1, 2, 1);
    hold on;
    plot(XX, errAA_fp, '*', 'Color', colors(1,:), 'LineWidth', 1, 'MarkerSize', 5, 'DisplayName', "||A - QQ*A||");
    plot(XX, errSVD_fp, 'Color', colors(2,:), 'LineWidth', 1, 'MarkerSize', 5, 'DisplayName', "||A - QUSV*||");
    xlabel('r, "Reliability Parameter"');
    ylabel("Log Scale Approximation Error");
    set(gca, 'Yscale', 'log');
    title("Reliability Parameter r vs. Approximation Error");
    legend('Location', 'best')
    grid on;

    [~, s, ~] = svd(AA);
    sigmas = diag(s);

    subplot(1, 2, 2);
    plot(1:150, sigmas(1:150,:))
    set(gca, 'Yscale', 'log')
    xlabel('Index (i)')
    ylabel('\Sigma_i_i');
    ylim([1e-17 Inf]); 
    title('Singular Value Decay (\Sigma)')
    grid on;

    %%% experiments with power iteration to counter slow-singular value decay
    s = (1:nn).^(-0.5); % Power-law decay
    AAg = (randn(mm, nn) + 1i * randn(mm, nn)) * diag(s);
    % QQ_q0 = fixedRank(AAg, kk, pp);               % special case of qq = 0
    % QQ_q5 = fixedRank_power(AAg, kk, pp, qq);

    kk = 60; pp = 5;
    errAA_pi = zeros(5, 0);
    errSVD_pi = zeros(5, 0);

    XX = 0:60;
    for qq = XX
        QQ_pi = fixedRank_power(AAg, kk, pp, qq);
        [errAA_pi(qq+1), errSVD_pi(qq+1)] = compareQQA(AAg, QQ_pi);

    end

    figure; 
    subplot(1, 2, 1);
    hold on;
    plot(XX, errAA_pi, '*', 'Color', colors(1,:), 'LineWidth', 1, 'MarkerSize', 5, 'DisplayName', "||A - QQ*A||");
    plot(XX, errSVD_pi, 'Color', colors(2,:), 'LineWidth', 1, 'MarkerSize', 5, 'DisplayName', "||A - QUSV*||");
    xlabel('q, "Power Iteration Applications"');
    ylabel("Log Scale Approximation Error");
    set(gca, 'Yscale', 'log');
    title("Power Iteration Applications vs. Approximation Error");
    grid on;
    legend('Location', 'best')

    
    [~, s, ~] = svd(AAg);
    sigmas = diag(s);

    subplot(1, 2, 2);
    plot(1:150, sigmas(1:150,:))
    set(gca, 'Yscale', 'log')
    xlabel('Index (i)')
    ylabel('\Sigma_i_i');
    ylim([1e-17 Inf]); 
    title('Singular Value Decay (\Sigma)')
    grid on;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


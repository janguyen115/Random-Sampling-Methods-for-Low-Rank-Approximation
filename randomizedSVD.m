function [] = main() % Generate Matrix, Setup Problem
    rng(444)

    m = 200; n = 200; k = 20; p = 5; r = 10; tol = 1e-10;
    l = k + p; % desired rank + oversampling
    fprintf("Parameters: m = %i ; n = %i, l = k + p = %i + %i ; r = %i ; tol = 1e-10\n", m, n, k, p, r)

    A = discretizeLaplace(m);  

    Q = getQ(A, 'adaptive_gaussian', 60, 10, 1e-10);
    
    plotByL(A, 'gaussian', 'direct')
    getLoRA(A, 'gaussian', 'direct', l, r, tol)
    getLoRA(A, 'adaptive_gaussian', 'direct', l, r, tol)
end

function [] = getLoRA(A, sample, typeSVD, l, r, tol)
    [m, n] = size(A);
    I = eye(m);
    Q = getQ(A, sample, l, r, tol);
    fprintf("\nSize of Q (%s): %i x %i\n", sample, size(Q));
    fprintf("Low-Rank Approximation (%s): norm((I - Q * Q') * A) = %.3e\n", sample, norm((I - Q * Q') * A))
    [U, S, V] = getSVD(A, Q, typeSVD);
    fprintf("%s SVD: norm(A - U * S * V') = %.3e\n", typeSVD, norm(A - U * S * V'))
end

function [Q] = getQ(A, type, l, r, tol) 
    [m, n] = size(A);
    I = eye(m);
    if nargin < 4
        r = 10; tol = 1e-10;
    end

    if strcmp(type, "gaussian")
        W = randn(n, l) + 1i * randn(n, l);
        Y = A * W; % initialize samples from range(A) m x l
        [Q, ~] = qr(Y, 'econ');
    elseif strcmp(type, "adaptive_gaussian")
        W = randn(n, r) + 1i * randn(n, r);
        Y = A * W; % initialize samples from range(A) m x r
        j = 0;
        Q = zeros(m, 0);
        while max(vecnorm(Y(:,j+1:j+r), 2, 1)) > (tol / 10 * sqrt(2 / pi))
            j = j + 1;
            Y(:, j) = (I - Q * Q') * Y(:,j);
            qj = Y(:,j) / norm(Y(:, j));
            Q = [Q qj];
            W = [W (randn(n, 1) + 1i * randn(n, 1))];
            Y = [Y ((I - Q * Q') * A * W(:, j + r))];
            for i = j+1 : j+r-1
                Y(:, i) = Y(:, i) - qj * dot(qj, Y(:, i));
            end
        end   
    elseif strcmp(type, "SRFT")
        theta = unifrnd(0, 2*pi, [n, 1]);
        D = diag(exp(1i * theta));

        [p, q] = meshgrid(0:m-1, 0:m-1);
        F = m^(-1/2) * exp(-2 * pi * 1i .* (p) .* (q) / m);
        
        cols = randperm(n, l);
        R = I(:,cols);
        
        SRFT = sqrt(n/l) * D * F * R;
        Y = A * SRFT;

        [Q, ~] = qr(Y, 'econ');
    end
end

function [U, S, V] = getSVD(A, Q, type)
    [m, n] = size(A);
    [~, l] = size(Q);

    if strcmp(type, "direct") % recommended when epsilon not so small or ln large
        B = Q' * A;
        [u, S, V] = svd(B);
        U = Q * u;
        [q, w, e] = svd(Q * Q' *A);
    elseif (strcmp(type, 'row_extraction')) % faster, less accurate
        l = min([m, n, l]);
        [~, ~, J] = optim_id(Q, l);
        X = Q * pinv(Q(J,:));
        [R, T] = qr(A(J,:));
        Z = X * R;
        [U, S, v] = svd(Z);
        V = T' * v;
    else
        fprintf("Error: Improper `type` input, please enter one of the following: `direct`, `row_extraction.`")
    end
end

function [A] = discretizeLaplace(m)
    t = linspace(0, 2*pi, m)';

    x1 = cos(t);
    y1 = sin(t);

    x2 = 2 + cos(t);
    y2 = sin(t);

    A = zeros(m, m);
    for i = 1:m
        for j = 1:m
            % Compute distance between points (x2(i), y2(i)) and (x1(j), y1(j))
            r = sqrt((x2(i) - x1(j))^2 + (y2(i) - y1(j))^2);
            
            % Logarithmic kernel
            A(i, j) = log(r);
        end
    end

    L1 = 2 * pi;
    weights = (L1 / m) * ones(1, m);

    A = A .* weights;

    sigma_max = norm(A, 2 );
    A = A / sigma_max;
end

function [] = plotByL(A, sample, typeSVD)
    [m, n] = size(A);
    I = eye(m, n);
    [u, s, v] = svd(A);
    err = zeros(1, 30); errSVD = zeros(1, 30); diffSVD = zeros(1, 30);

    count = 1;
    X = linspace(5, 150, 30);
    for l = X
        Q = getQ(A, sample, l);
        err(count) = norm((I - Q*Q')*A);

        [U, S, V] = getSVD(A, Q, typeSVD);

        errSVD(count) = norm(U*S*V' - A);
        diffSVD(count) = norm(U*S*V' - u*s*v');
        count = count + 1;
    end

    colors = [0, 114, 178;  % Blue
          230, 159, 0;  % Orange
          0, 158, 115;  % Green
          204, 121, 167] / 255; % Purple

    figure; 
    subplot(1, 2, 1)
    hold on;
    plot(X, err, '-*', 'Color', colors(1, :), 'LineWidth', 1, 'MarkerSize', 5, 'DisplayName', "||A - QQ*A||")
    xlabel('l = k + p, "Desired Rank(Q) + Oversampling"')
    ylabel("Log Scale Approximation Error")
    title(sprintf('Approximation Rank vs. Approximation Error | sampling = %s | typeSVD = %s', sample, typeSVD))
    set(gca, 'Yscale', 'log');

    plot(X, errSVD, 'Color', colors(3, :), 'LineWidth', 1, 'MarkerSize', 5, 'DisplayName', sprintf("||A - svd(QQ*A)|| %s SVD", typeSVD))
    plot(X, diffSVD, 'Color', colors(2, :), 'LineWidth', 1, 'MarkerSize', 5, 'DisplayName', sprintf("||svd(A) - svd(QQ*A)|| %s SVD", typeSVD))
    
    legend('Location', 'best')
    grid on;
    hold off;

    sigmas = diag(s);     
    subplot(1, 2, 2)
    plot(1:200, sigmas)
    set(gca, 'Yscale', 'log')
    xlabel('Index (i)')
    ylabel('\Sigma_i_i');
    title('Singular Value Decay (\Sigma)')
    grid on;
end

%%% Not my code, used to get Interpolative Decomposition (ID) matrix for
%%% SVD via row_extraction
function [approx, Z, cols] = optim_id(A, l)
    % Perform QR decomposition with column pivoting
    [Q, R, P] = qr(A, 'vector');
    
    % Extract the top l columns and rows
    R_l = R(1:l, 1:l);
    cols = P(1:l);
    C = A(:, cols);
    
    % Solve the least squares problem
    Z = (R_l' * R_l) \ (C' * A);
    
    % Compute the approximation
    approx = C * Z;
end
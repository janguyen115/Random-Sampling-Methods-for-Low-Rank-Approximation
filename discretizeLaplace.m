function [AA] = discretizeLaplace(mm, nn)
    tt = linspace(0, 2*pi, mm)';

    xx1 = cos(tt);
    yy1 = sin(tt);

    xx2 = 2 + cos(tt);
    yy2 = sin(tt);

    AA = zeros(mm, nn);
    for i = 1:mm
        for j = 1:mm
            % Compute distance between points (x2(i), y2(i)) and (x1(j), y1(j))
            rr = sqrt((xx2(i) - xx1(j))^2 + (yy2(i) - yy1(j))^2);
            
            % Logarithmic kernel
            AA(i, j) = log(rr);
        end
    end

    L1 = 2 * pi;
    weights = (L1 / mm) * ones(1, mm);

    AA = AA .* weights;

    sigma_max = norm(AA, 2 );
    AA = AA / sigma_max;
end
function [errAA, errSVD] = compareQQA(AA, QQ)

    [uu, SS, VV] = svd(QQ' * AA);
    UU = QQ * uu;

    errAA = norm(AA - QQ*QQ'*AA);
    errSVD = norm( AA - UU*SS*VV');    
end
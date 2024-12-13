function antonio(x, y, SCOPE)   
    % 1/x, 1/y 관계 피팅
    u = 1 ./ (x+eps);
    v = 1 ./ (log10(y+eps)+eps);
    
    function rrs = restsquare(A)
        p = polyfit(u, v, 1);
    
        % p1과 p2 추출
        p1 = p(1);
        p2 = p(2);
        
        % a, b, c 계산
        b = 1 / p1;
        c = b * p2;
        f = @(x)(10.^(A-b./(c+x)));
        rrs = (y-f(x)).^2;
    end
    fzero(@restsquare,max(y));
    % 결과 출력
    % fprintf('a = %.4f, b = %.4f, c = %.4f\n', a, b, c);
    fplot(restsquare,[min(y), max(y)])
end


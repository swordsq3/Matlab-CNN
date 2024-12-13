Times = double(50);  % 타이머 반복 횟수를 double로 변환
x = magic(1000);     % 1000x1000 매직 스퀘어 행렬 생성

tic;  % 타이머 시작
for i = 1:Times
    y = cnnPool([3 3], x, 'meanpool');
end
fprintf('meanpool time used:%s\n', toc);

% 두 번째 풀링 테스트
for i = 1:Times
    y = cnnPool([3 3], x, 'newpool');
end
fprintf('newpool time used:%s\n', toc);
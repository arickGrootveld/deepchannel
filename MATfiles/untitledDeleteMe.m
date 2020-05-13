test1.one = 1;
test1.two = 2;

test2.one = 1;
test2.two = 2;

test3.one = 1;
test3.two = 2;
test3.three = 3;

test.test1 = test1;
test.test2 = test2;
test.test3 = test3;

% matSave('data', 'GETestDataAK', test);

save('data/AK1.mat', '-struct', 'test');
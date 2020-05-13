% Converting the data file passed to it to be 
function AGtoAKDataConversionVlad(filename, ls_coeffs)

load(filename, 'observedStates', 'finalStateValues');

obsStComplex = squeeze(observedStates(1, 2:end,:) + 1j*observedStates(2,2:end,:));

fnlStComplex = squeeze(finalStateValues(2,:) + 1j*finalStateValues(4,:));
deleteMeComp = squeeze(finalStateValues(1,:) + 1j*finalStateValues(3,:));

% correcting the LS coeffs so that they correctly match up against the size
% of the data to be loaded
ls_shape = size(ls_coeffs);

% If its not a column vector, make it into a column vector
if(ls_shape(1) ~= 1)
    ls_coeffs = transpose(ls_coeffs);

end

% x_pred = ls_coeffs * obsStComplex;
% 
% sq_err = abs((x_pred - fnlStComplex).^2);
% 
% mean(sq_err)


% a_ls_pred = [1.29526954081751 + 3.93126983016584e-05i;-0.172420799862244 + 0.000119151856212871i;-0.107957390649879 - 0.000228777965743445i;-0.00495884316176761 - 7.49261029625185e-05i;-0.000221006223159027 + 5.78631116978062e-05i;-0.000670301706677115 - 7.43782032480310e-05i;-0.000309187929465809 + 0.000378061378379583i;0.000276117988988933 - 0.000388145692365941i;0.00308909482050926 + 9.49291430518389e-05i;-0.0123153374337684 + 8.64443831362551e-05i];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% VLADS ATTEMPT BELOW %%%%%%%%%%%%%%%%%%%%%%%

% obsStComplex - z 
% fnlStComplex - x

% Convert into long observation array
longObsStComplex = zeros(1, size(obsStComplex, 2)-1 + size(obsStComplex, 1));
for i = 1:size(obsStComplex, 2)
    if (i == 1)
        longObsStComplex(1, 1:9) = obsStComplex(:, 1)';
    else
        longObsStComplex(1, 9+i) = obsStComplex(9, i);
    end
end

longObsStComplex = longObsStComplex(1, 1:100000);

a = 5
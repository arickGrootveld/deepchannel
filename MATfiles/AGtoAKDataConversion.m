% Converting the data file passed to it to be 
function reformattedData = AGtoAKDataConversion(filename, ls_coeffs)

load(filename, 'observedStates', 'finalStateValues', 'systemStates');

obsStComplex = transpose(flipud(squeeze(observedStates(1, :,:) + 1j*observedStates(2,:,:))));
fnlStComplex = squeeze(finalStateValues(2,:) + 1j*finalStateValues(4,:));
trueStatesComplex = squeeze(systemStates(1,:,:) + 1j*systemStates(2,:,:));

% correcting the LS coeffs so that they correctly match up against the size
% of the data to be loaded
ls_shape = size(ls_coeffs);

% If its not a column vector, make it into a column vector
if(ls_shape(1) == 1)
    ls_coeffs = transpose(ls_coeffs);

end

x_pred = obsStComplex * ls_coeffs;

sq_err = abs((x_pred - fnlStComplex.')).^2;

mean(sq_err)

reformattedData = {};
reformattedData.obsStates = obsStComplex;
reformattedData.trueStates = trueStatesComplex;
reformattedData.finalStates = fnlStComplex;

end


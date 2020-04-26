function toepedData = teoplitizifyData(data, seqLength)
    % Input size (2, N)
    [~, N] = size(data);
    % Outputsize (2, seqLength, N-seqLength+1)
    toepedData = zeros(2, seqLength, N-seqLength+1);
    
    
    if(N-seqLength+1 > 1)
       for m = 1:(N-seqLength+1)
          toepedData(:, :, m) = data(:, m:m+seqLength-1);
       end
    else
       disp('This is not gonna work');
    end

end
clear
close
% Set up the Import Options and import the data
opts = spreadsheetImportOptions("NumVariables", 17);

opts.Sheet = "Sheet1";
opts.DataRange = "C3:S16";

opts.VariableNames = ["city1", "city1_1", "city1_2", "city2", "city2_1", "city2_2", "city3", "city3_1", "city3_2", "city4", "city4_1", "city4_2", "fd", "VarName16", "VarName17", "VarName18", "X"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double"];

% Input MRIOT
MRIOT_test = readtable("Data\MRIOT_test.xlsx", opts, "UseExcel", false);
MRIOT_test = table2array(MRIOT_test);
clear opts

Data = MRIOT_test;
NCity = 4; % the number of countries and regions
for idx_city = 1:NCity
    close all
    Shock_rate = 0.01; % the  shock rate of disaster
    
    NSector = 3; % the number of sectors in each region
    N = NSector*NCity;
    
    [Rs,Cs] = size(Data);
    I = eye(N);
    Z = Data(1:N,1:N);
    X = Data(Rs, 1:N)';
    A = Z ./ ( repmat(X', [N, 1]));
    A(isnan(A)) = 0;
    L = (I-A)^-1;
    TVA = Data(N+1, 1:N);
    FD_temp = Data(1:N, N+1:Cs - 1);
    FD = sum(FD_temp,2);
    
    % Calculate Shock
    Shock_region_num = 1;
    Shock = zeros(1,N);
    Shock_str = 1+(idx_city-1)*NSector;
    Shock_end = NSector+(idx_city-1)*NSector;
    Shock(Shock_str:Shock_end) = TVA(Shock_str:Shock_end) .* Shock_rate;
      
    % Simulation
    TFD = FD + Shock';
    TFD(isnan(TFD)) = 0;
    Z(isnan(Z)) = 0;
    delta_TFD = Shock';    
    delta_Y0 = (I-A)^-1 * delta_TFD;
    Indirectloss = sum(delta_Y0,1);
    Directloss = sum(Shock);
    Rate = Indirectloss ./ Directloss;   
end
clc
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

NCity = 4;
NSector = 3;
N = NCity * NSector;
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
Output =  Data(1:N, Cs);

% ==========================  Calculate APL ======================
I = eye(N); 
X = Output; 
diag_X = diag(X);
Y = X';
diag_Y = diag(Y);
A = Z ./ ( repmat(X', [N, 1]));
A(isnan(A)) = 0; 
L = (I-A)^-1; 
H = L * (L - I); 
% when i≠j
T = H./L; 
% when i=j
for R = 1:NCity

    for C = 1:NCity
        P_MM = L;
        H_MM = H;
        T_MM = T;
    end
    for i = 1:NCity
        for j = 1:NCity
            if i == j
                T_MM(i,j) = H_MM(i,j)./(P_MM(i,j)-1);
            end
        end
    end
    T( R,C) = T_MM(R,C);
end
T(isnan(T)==1) = 0;
% ========================== Calculate APL-backward and APL-forward ======================
APL_backward_vector = mean(T,1);
APL_backward = reshape(APL_backward_vector,[NCity,NSector]);
APL_backward = APL_backward';

APL_forward_vector = mean(T,2);
APL_forward = reshape(APL_forward_vector,[NCity,NSector]);
APL_forward = APL_forward';
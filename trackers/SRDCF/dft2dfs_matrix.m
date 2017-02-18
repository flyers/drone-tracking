% Constructs a sparse matrix that transforms the discrete fourier transform
% (DFT) to the real discrete fourier series (DFS), given the input and
% output index permutations.

function dfs_matrix = dft2dfs_matrix(dft_sym_ind, dft_pos_ind, dft_neg_ind, dfs_sym_ind, dfs_real_ind, dfs_imag_ind)

i_sym = dfs_sym_ind;
j_sym = dft_sym_ind;
v_sym = ones(length(dft_sym_ind) ,1);

i_real_pos = dfs_real_ind;
j_real_pos = dft_pos_ind;
v_real_pos = 1/sqrt(2) * ones(length(dft_pos_ind), 1);

i_real_neg = dfs_real_ind;
j_real_neg = dft_neg_ind;
v_real_neg = 1/sqrt(2) * ones(length(dft_neg_ind), 1);

i_imag_pos = dfs_imag_ind;
j_imag_pos = dft_pos_ind;
v_imag_pos = 1/(1i * sqrt(2)) * ones(length(dft_pos_ind), 1);

i_imag_neg = dfs_imag_ind;
j_imag_neg = dft_neg_ind;
v_imag_neg = -1/(1i * sqrt(2)) * ones(length(dft_neg_ind), 1);

i_tot = [i_sym; i_real_pos; i_real_neg; i_imag_pos; i_imag_neg];
j_tot = [j_sym; j_real_pos; j_real_neg; j_imag_pos; j_imag_neg];
v_tot = [v_sym; v_real_pos; v_real_neg; v_imag_pos; v_imag_neg];

dft_length = length(dft_sym_ind) + length(dft_pos_ind) + length(dft_neg_ind);

dfs_matrix = sparse(i_tot, j_tot, v_tot, dft_length, dft_length);
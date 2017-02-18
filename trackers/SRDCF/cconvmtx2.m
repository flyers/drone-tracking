function H = cconvmtx2(h)

[block_size, num_blocks] = size(h);
num_elem = block_size*num_blocks;

H1 = spalloc(num_elem, block_size, block_size*nnz(h));
H = spalloc(num_elem, num_elem, num_elem*nnz(h));

% create the first n columns
for col = 1:block_size
    H1(:,col) = reshape(circshift(h, [col-1 0]), num_elem, 1);
end

% construct all blocks in H
for block = 1:num_blocks
    H(:,block_size*(block-1)+1:block_size*block) = circshift(H1, [(block-1)*block_size 0]);
end
end

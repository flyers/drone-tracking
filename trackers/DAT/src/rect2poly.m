function P = rect2poly(R)
%RECT2POLY Convert rectangle to polygon
P = zeros(1,8);
% TL
P(1) = R(1);
P(2) = R(2);
% TR
P(3) = R(1)+R(3)-1;
P(4) = R(2);
% BR
P(5) = R(1)+R(3)-1;
P(6) = R(2)+R(4)-1;
% BL
P(7) = R(1);
P(8) = R(2)+R(4)-1;

end


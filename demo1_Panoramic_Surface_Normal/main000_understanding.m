clear; clc;

theta = -40 * pi / 180;
rotR = [cos(theta), -sin(theta); sin(theta), cos(theta)]; % counterclockwise
cur_xy = [0; 1];
new_xy = rotR*cur_xy


theta = 90 * pi / 180;
rotR = [cos(theta), -sin(theta); sin(theta), cos(theta)]; % counterclockwise
cur_xy = [1; 0];
new_xy = rotR*cur_xy



theta = 60 * pi / 180;
rotR = [cos(theta), -sin(theta); sin(theta), cos(theta)]; % counterclockwise
cur_xy = [1; 0];
new_xy = rotR*cur_xy

theta = -30 * pi / 180;
rotR = [cos(theta), -sin(theta); sin(theta), cos(theta)]; % counterclockwise
cur_xy = [0; 1];
new_xy = rotR*cur_xy

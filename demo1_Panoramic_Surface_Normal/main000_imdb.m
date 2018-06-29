load imdb.mat;

imdb.train_localNormal = imdb.train_normal;
imdb.val_localNormal = imdb.val_normal;


for imgIdx = 1:length(imdb.train_image)
    cur_path_to_normal = imdb.train_localNormal{imgIdx};
    new_path_to_normal = strrep(cur_path_to_normal,'/normal/','/localNormal/');
    imdb.train_localNormal{imgIdx} = new_path_to_normal;
end
for imgIdx = 1:length(imdb.val_image)
    cur_path_to_normal = imdb.val_localNormal{imgIdx};
    new_path_to_normal = strrep(cur_path_to_normal,'/normal/','/localNormal/');
    imdb.val_localNormal{imgIdx} = new_path_to_normal;
end
save('imdb.mat','imdb');
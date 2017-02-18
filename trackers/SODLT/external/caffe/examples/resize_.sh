for fold in /data/kxmo/ImageNet/train_extract/*; do
	cp -R $fold /data/nwangab
done
for fold in /data/nwangab/*; do
	for name in $fold/*.JPEG; do
		convert -resize 256x256\! $name $name
	done
done

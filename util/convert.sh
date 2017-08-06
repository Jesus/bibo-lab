# This'll fix the orientation of all *.jpg files in the current dir
for file in `ls *.jpg`; do
	basename=`echo $file | cut -d"." -f 1`
	convert -auto-orient "$file" "$basename-orient.jpg"
done

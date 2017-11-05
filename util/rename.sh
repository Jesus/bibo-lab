i=1
for file in *.jpg; do
  new_name=$(printf "%04d.jpg" "$i") #04 pad to length of 4
  mv -i -- "$file" "$new_name"
  i=$(($i + 1))
done

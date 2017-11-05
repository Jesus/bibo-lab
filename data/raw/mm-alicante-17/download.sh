# for i in `seq 1 25`; do
#   wget "http://mediomaratonalicante.es/index.php/fotos/nggallery/2017/VIII-5k-10k-Medio-Marat%C3%B3n-Alicante-2017/page/$i" -O - >> "mm-alicante-17.html"
# done

for url in `cat mm-alicante-17.html | grep "data-src" | cut -d'"' -f 2`; do
  wget $url
done

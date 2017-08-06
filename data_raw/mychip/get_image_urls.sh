# wget --no-check-certificate https://www.mychip.es/galerias/meta-875-75 -O - > html.txt
# wget --no-check-certificate https://www.mychip.es/galerias/meta-875-75?page=2 -O - >> html.txt
# wget --no-check-certificate https://www.mychip.es/galerias/meta-875-75?page=3 -O - >> html.txt
# wget --no-check-certificate https://www.mychip.es/galerias/meta-875-75?page=4 -O - >> html.txt
# wget --no-check-certificate https://www.mychip.es/galerias/meta-875-75?page=5 -O - >> html.txt
# wget --no-check-certificate https://www.mychip.es/galerias/meta-875-75?page=6 -O - >> html.txt
#
# grep JPG html.txt | grep img | cut -d'"' -f 2 > image_urls.txt

# wget --no-check-certificate https://www.mychip.es/galerias/paso-por-meta-875-01 -O - > html.txt
# wget --no-check-certificate https://www.mychip.es/galerias/paso-por-meta-875-01?page=2 -O - >> html.txt

for url in `cat image_urls.txt`; do
  wget --no-check-certificate $url
done

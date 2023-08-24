dynamod -m 0 -C 8 -d 0.1 --i1 0 -r 2 -o config.start.xml
dynarun config.start.xml -c 3000000
dynamod config.out.xml.bz2 -r 2 
dynarun config.out.xml.bz2 -c 1000000 -o config.restart.xml
dynamod config.restart.xml -T 2 -o config.restart.xml.bz2
dynarun config.restart.xml.bz2 -c 1000000
dynarun config.restart.xml.bz2 -c 100000000
mkdir Production
cp * Production/
cd Production
dynarun config.restart.xml.bz2 -c 500000 -o config0.xml
for i in $(seq 0 1000)
do
dynarun config$i.xml -c 500000 -o config$((i+1)).xml
done
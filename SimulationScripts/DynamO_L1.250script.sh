dynamod -m 1 -C 8 -d 0.05 --i1 0 --f1 1.25 --f2 1 -r 0.7942388901466997 -o config.start.xml
dynarun config.start.xml -c 3000000
dynamod config.out.xml.bz2 -r 0.764 
dynarun config.out.xml.bz2 -c 1000000 -o config.restart.xml
dynamod config.restart.xml -T 0.764 -o config.restart.xml.bz2
dynarun config.restart.xml.bz2 -c 1000000
dynarun config.restart.xml.bz2 -c 100000000

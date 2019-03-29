
#!/bin/sh 
echo Running all 	


for x in  1
	do
		for prob in 5
		do
			python ptBayeslands_sedvec.py -p $prob -s 200 -r 10 -t 10 -swap 0.1 -b 0.25 -pt 0.5
 
		done
	done 
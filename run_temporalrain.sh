
#!/bin/sh 
echo Running all 	 

 
for t in 4
	do
		for x in 200 

		do
 
			python ptBayeslands_extended.py -p 7 -s $x -r 10 -t 10 -swap 0.1 -b 0.25 -pt 0.5 -epsilon 0.5 -rain_intervals $t
 
		done
	done 


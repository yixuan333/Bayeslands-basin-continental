
#!/bin/sh 
echo Running all 	 

 
for t in 4
	do  
		for x in 4000

		do
 
			python ptBayeslands_continental.py -p 1 -s $x -r 10 -t 10 -swap 0.01 -b 0.25 -pt 0.5 -epsilon 0.5 -rain_intervals $t
 
		done
	done 



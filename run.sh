
#!/bin/sh 
echo Running all 	 

 
for t in 4
	do  
		for x in 200

		do
 
			python ptBayeslands_continental.py -p 2 -s $x -r 8 -t 10 -swap 0.9 -b 0.25 -pt 0.5 -epsilon 0.5 -rain_intervals $t #> log.txt
 
		done
	done 



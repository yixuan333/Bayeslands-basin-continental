
#!/bin/bash  
echo Running all 	 

problem=2
replica=4

samples=100
swapint=$samples
maxtemp=2 
burn=0.0
pt_stage=0.35
raintimeint=4
initialtopoep=0.5 # not used anymore
cov=0

echo $problem 


 
for t in 1 #2 3 4
	do  
			#python ptBayeslands.py -p $problem -s $samples -r $replica -t $maxtemp -swap $swapint -b $burn -pt $pt_stage  -epsilon $initialtopoep -rain_intervals $raintimeint
			python visualise.py -p $problem -s $samples -r $replica -t $maxtemp -swap $swapint -b $burn -pt $pt_stage  -epsilon $initialtopoep -rain_intervals $raintimeint
 

	done 


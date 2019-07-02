
#!/bin/bash  
echo Running all 	 

problem=3
replica=10

samples=600
swapint=$samples
maxtemp=2
burn=0.25
pt_stage=0.25
raintimeint=4
initialtopoep=0.5

echo $problem 


 
for t in 1 #4 8 16
	do  
			python ptBayeslands.py -p $problem -s $samples -r $replica -t $maxtemp -swap $swapint -b $burn -pt $pt_stage  -epsilon $initialtopoep -rain_intervals $raintimeint
			python visualise.py -p $problem -s $samples -r $replica -t $maxtemp -swap $swapint -b $burn -pt $pt_stage  -epsilon $initialtopoep -rain_intervals $raintimeint

  
  
	done 


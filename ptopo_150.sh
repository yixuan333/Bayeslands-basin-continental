#!/bin/bash
#Project: Converting paleogeography grid into paleotopography
#Date: 26/07/2017
#Author: Carmen Braz


## Input files ---------

version=P100
# paleotopo=Paleotopo_data/Paleotopo_150_Ma_${version}.shp
paleotopo=Init_topo_polygon/data/Paleotopo_${version}.shp

temperature=$1
# xy files
# resol="_5km_xy"
# resol="_25km"
resol="_50km"

input_latlon=Init_topo_polygon/data/LatLon${resol}.xy
input_utm=Init_topo_polygon/data/UTM${resol}.xy

region=85/179/-55/10

## Output files -----------

# Converted to gmt file
out_gmt=Init_topo_polygon/Paleotopo_${version}.gmt
# Paleotopography converted to grid
ptopo_grid=Init_topo_polygon/ptopo_${version}.nc

# Smoothing factor - increase from 3 for higher resolution models
g_filter=3
# Smoothed grid
smooth_grid=Init_topo_polygon/ptopo_g${g_filter}${resol}.nc

# grdtrack output 
latlon_csv=Init_topo_polygon/topo_150Ma_latlon_g${g_filter}${resol}
final=Init_topo_polygon/Paleotopo

ogr2ogr -f "GMT" ${out_gmt} ${paleotopo}

gmt grdmask ${out_gmt} -R${region} -I0.25 -Nz -aZ=ELEVATION -V -G${ptopo_grid}


gmt grdfilter ${ptopo_grid} -G${smooth_grid} -D0 -I0.25 -V -Fg${g_filter}


gmt grdtrack -G${smooth_grid} ${input_latlon} -V > ${latlon_csv}_${version}.csv


paste ${input_utm} ${latlon_csv}_${version}.csv | awk -F" " '{ print $1 " " $2 " " $5 }' > ${final}_${version}_${temperature}.csv

# Might need to use dos2unix on input utm and latlon

# To change precision of output files for 5km resolution model
awk -F" " '{ printf("%.2f,%.2f,%.2f\n",$1,$2,$3)}' ${final}_${version}_${temperature}.csv > temp.csv
awk -F"," '{ print $1 " " $2 " " $3 }' temp.csv > ${final}_${version}${resol}_prec2_${temperature}.csv

# cp ${final}_${version}${resol}_prec2.csv Parameters_maps${resol}/Paleotopo/${final}_${version}${resol}_prec2.csv
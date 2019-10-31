def run_badlands(self, input_vector):
        #Runs a badlands model with the specified inputs
 
        rain_regiontime = self.rain_region * self.rain_time # number of parameters for rain based on  region and time 

        #Create a badlands model instance
        model = badlandsModel()

        #----------------------------------------------------------------
        # Load the XmL input file
        model.load_xml(str(self.run_nb), self.input, muted=True)

        # prob = [0,1,2]
        # if  problem in prob: #problem==1 or problem==2 : # when you have initial topo (problem is global variable)
        #if problem == 0 or problem ==1 or problem ==2 or problem==3:
         #   init = False
        #else:
         #   init = True # when you need to estimate initial topo

        init = True


        if init == True:

            geoparam  = rain_regiontime+11  # note 10 parameter space is for erod, c-marine etc etc, some extra space ( taking out time dependent rainfall)
            inittopo_vec = input_vector[geoparam:]
            filename=self.input.split("/")
            problem_folder=filename[0]+"/"+filename[1]+"/"

            #Update the initial topography
            #Use the coordinates from the original dem file
            xi=int(np.shape(model.recGrid.rectX)[0]/model.recGrid.nx)
            yi=int(np.shape(model.recGrid.rectY)[0]/model.recGrid.ny)
            #And put the demfile on a grid we can manipulate easily
            elev=np.reshape(model.recGrid.rectZ,(xi,yi)) 

            inittopo_estimate = self.process_inittopo(inittopo_vec)  
            inittopo_estimate = inittopo_estimate[0:  elev.shape[0], 0:  elev.shape[1]]  # bug fix but not good fix - temp @ 

            #Put it back into 'Badlands' format and then re-load the model
            filename=problem_folder+str(self.run_nb)+'/demfile_'+ str(int(self.temperature*10)) +'_demfile.csv' 

            elev_framex = np.vstack((model.recGrid.rectX,model.recGrid.rectY,inittopo_estimate.flatten()))
            np.savetxt(filename, elev_framex.T, fmt='%1.2f' ) 
            model.input.demfile=filename 
            model.build_mesh(model.input.demfile, verbose=False)

        # Adjust precipitation values based on given parameter
        #print(input_vector[0:rain_regiontime] )
        model.force.rainVal  = input_vector[0:rain_regiontime] 

        # Adjust erodibility based on given parameter
        model.input.SPLero = input_vector[rain_regiontime]  
        model.flow.erodibility.fill(input_vector[rain_regiontime ] )

        # Adjust m and n values
        model.input.SPLm = input_vector[rain_regiontime+1]  
        model.input.SPLn = input_vector[rain_regiontime+2] 

        #Check if it is the etopo extended problem
        #if problem == 4 or problem == 3:  # will work for more parameters
        model.input.CDm = input_vector[rain_regiontime+3] # submarine diffusion
        model.input.CDa = input_vector[rain_regiontime+4] # aerial diffusion

        if problem != 1:
            model.slp_cr = input_vector[rain_regiontime+5]
            model.perc_dep = input_vector[rain_regiontime+6]
            model.input.criver = input_vector[rain_regiontime+7]
            model.input.elasticH = input_vector[rain_regiontime+8]
            model.input.diffnb = input_vector[rain_regiontime+9]
            model.input.diffprop = input_vector[rain_regiontime+10] 

        #Check if it is the mountain problem
        '''if problem==10: # needs to be updated
            #Round the input vector 
            k=round(input_vector[rain_regiontime+5],1) #to closest 0.1  @Nathan we need to fix this

            #Load the current tectonic uplift parameters
            tectonicValues=pandas.read_csv(str(model.input.tectFile[0]),sep=r'\s+',header=None,dtype=np.float).values
        
            #Adjust the parameters by our value k, and save them out
            newFile = "Examples/mountain/tect/uplift"+str(self.temperature)+"_"+str(k)+".csv"
            newtect = pandas.DataFrame(tectonicValues*k)
            newtect.to_csv(newFile,index=False,header=False)

            #Update the model uplift tectonic values
            model.input.tectFile[0]=newFile'''

        elev_vec = collections.OrderedDict()
        erodep_vec = collections.OrderedDict()
        erodep_pts_vec = collections.OrderedDict()
        elev_pts_vec = collections.OrderedDict()

        for x in range(len(self.sim_interval)):
            self.simtime = self.sim_interval[x]
            model.run_to_time(self.simtime, muted=True)

            elev, erodep = interpolateArray(model.FVmesh.node_coords[:, :2], model.elevation, model.cumdiff)

            erodep_pts = np.zeros((self.erodep_coords.shape[0]))
            elev_pts = np.zeros((self.erodep_coords.shape[0]))

            for count, val in enumerate(self.erodep_coords):
                erodep_pts[count] = erodep[val[0], val[1]]
                elev_pts[count] = elev[val[0], val[1]]

            print('Sim time: ', self.simtime  , "   Temperature: ", self.temperature)
            elev_vec[self.simtime] = elev
            erodep_vec[self.simtime] = erodep
            erodep_pts_vec[self.simtime] = erodep_pts
            elev_pts_vec[self.simtime] = elev_pts
 
        return elev_vec, erodep_vec, erodep_pts_vec, elev_pts_vec




def main():


if __name__ == '__main__':main()
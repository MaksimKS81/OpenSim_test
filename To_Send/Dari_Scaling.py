
import os.path

modelFolder = os.path.normpath("C:/Users/Maksim/Documents/GitHub/OpenSim_test")

settingsFolder = os.path.normpath("C:/Users/Maksim/Documents/GitHub/OpenSim_test")
  
# Define paths
#modelFolder		=	os.path.join(getResourcesDir(),"Models", "Gait2354_Simbody");
scaleSetup		=	os.path.join(modelFolder, "Dari_scale_settings.xml");
markerSetFile	=	os.path.join(modelFolder, "Dari_Scale_Marker_Set_v0.1.xml");
ikSetupFile		=	os.path.join(modelFolder, "Dari_squat_02_Setup_IK.xml");

#idSetupFile		=	os.path.join(modelFolder, "subject01_Setup_InverseDynamics.xml");

# Models
modelName	    =	os.path.join(modelFolder, "Rajagopal2015_passiveCal_hipAbdMoved_no_markers.osim");
scaleModelName	=	os.path.join(modelFolder, "subject01_simbody.osim");

# Output data files
ikMotionFilePath=	os.path.join(modelFolder,"subject01_walk1_ik.mot");
idResultsFile	=	os.path.join(modelFolder,"subject01_inverse_dynamics.sto");

# Define some of the subject measurements
subjectMass		=	72.6


## load and define model

# Load model 
loadModel(modelName)
# Get a handle to the current model
myModel = getCurrentModel()
#initialize
myModel.initSystem()
myState = myModel.initSystem()

## Scaling Tool

# Create the scale tool object from existing xml
scaleTool = modeling.ScaleTool(scaleSetup)
scaleTool.run()

## load Scaled Model

# Load model 
loadModel(scaleModelName)
# Create a copy of the scaled model for use with the tools.
myModel = getCurrentModel().clone()
#initialize
myModel.initSystem()
myState = myModel.initSystem()




## Inverse Kinematics tool

ikTool = modeling.InverseKinematicsTool(ikSetupFile)
ikTool.setModel(myModel)
ikTool.run()
# Load a motion
loadMotion(ikMotionFilePath)
#initialize
myState = myModel.initSystem()

## Inverse Dynamics 
# Create the ID tool object from existing xml
#idTool = modeling.InverseDynamicsTool(idSetupFile)
# Set the model to scaled model from above
#idTool.setModel(myModel)
# Set the full path to the External Loads File
#idTool.setExternalLoadsFileName(extLoadsFile)
# Run the tool
#idTool.run()
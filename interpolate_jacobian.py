# example of Python script using cisstRobotPython

import cisstRobotPython
import numpy as np

# create a manipulator
r = cisstRobotPython.robManipulator()

# load DH parameters from a .rob file, result is 0 if loaded properly 
# r.LoadRobot('/home/ndrakes1/catkin_ws/src/cisst-saw/sawIntuitiveResearchKit/share/deprecated/dvpsm.rob')
r.LoadRobot("dvpsm.rob")
# check the length of the kinematic chain
print('length of kinematic chain: %d' % len(r.links))

# position when all joints are at 0
jp = np.zeros(6)
cp = np.zeros(shape = (4, 4)) # 4x4 matrix
cp = r.ForwardKinematics(jp)
print('joint position:')
print(jp)
print('cartesian position:')
print(cp)

# for a PSM, joint 3 is along instrument shaft
print('cartesian position for joint position [0,0,0.08,0,0,0]')
jp[2] = 0.08 # 8cm
cp = r.ForwardKinematics(jp)
print('joint position:')
print(jp)
print('cartesian position:')
print(cp)

# for a PSM, calculate Jacobian
jp = np.zeros(6) # joint angles
J_body = np.zeros(shape=(6,6), dtype=np.double) # placeholder
r.JacobianBody(jp, J_body)
print('body Jacobian:')
print(J_body)



np.set_printoptions(suppress=True)

# jp = np.array([0	,9.02341639931703E-07	,0.120000297771919	,0	,0, 0.00348943471394])
jp = np.array([-4.930237700113145882e-01,-1.032584729897805476e+00,1.357677554614626358e-01,-7.419913738162179362e-01,-2.500853084749094513e-01,5.853892118192192129e-01])

J_spatial = np.zeros(shape=(6,6), dtype=np.double) # placeholder
r.JacobianSpatial(jp, J_spatial)
print('Spatial Jacobian:')
print(J_spatial)

############### import csv data
import csv

# Open the CSV file
with open('/home/ndrakes1/Downloads/ToSMART/dvrk-si-3-15/train/free_space/jacobian/interpolated_all_jacobian.csv', 'r') as file:
    reader = csv.reader(file)

    # Convert the reader object to a list of lists
    data = list(reader)
    
    # Convert the list to a NumPy array
    data_array = np.array(data)

data = data_array[:,1:]  # read the 1-37 cols. remember the first col is time
shape = np.array(data.shape)  # shape is a tuple. convert to array

J_spatial_collect = data.reshape(shape[0],6,6)[0].astype(float)  # reader reads out strings. convert to float

np.set_printoptions(suppress=True)  # print in a pretty way
print('first row is:')
print(J_spatial_collect)

jacobian_error_count = 0

for i in range (J_spatial.shape[0]):
   for j in range(J_spatial.shape[1]):
      if (J_spatial[i][j] - J_spatial_collect[i][j] >= 1e-5):
          print("ERROR: ", J_spatial[i][j], ", ", J_spatial_collect[i][j] )
	  jacobian_error_count+=1


if (jacobian_error_count >= 1):
   print("ERROR: calculated jacobian does not match")

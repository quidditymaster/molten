# molten
A python application that uses TensorFlow to learn representations of 3D objects with desired properties. 

The molten.py script can be used to apply gradient descent to point clouds attempting to make the silhouette from various directions match some given image. We can attempt to learn a representation of the classic Godel Escher Bach cover block with a command something like.

python path/to/molten/molten.py --images G.jpg E.jpg B.jpg --output cloud.csv --image-dir path/to/molten/letters/Courier

This will generate a csv file with columns corresponding to the x, y, and z coordinates of points in our cloud. We can make them into a stl file via the helper script cloud2mesh.

python path/to/molten/cloud2mesh.py cloud.csv --output cloud.stl

You can load the stl file in your favorite mesh viewer. You should see something similar to the following.

![GEB mesh](https://raw.githubusercontent.com/quidditymaster/molten/master/geb_mesh.png)

IMPORTANT! these mesh files may be fed into 3D printing programs and edited but they are not ready to be printed as is. In particular the normals of the mesh will need to be made consistent to keep the inside and outside distinct.

Both molten.py and cloud2mesh have many more options than are apparent here, proper documentation coming some day after I finish my thesis.
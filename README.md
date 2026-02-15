# LapTimeSimulation
Planar and non-planar lap time simulation including GG diagram generation and track preprocessing.

"3d laptime simulation" computes optimal laptime using coordinates and elevation. "2d laptime simulation" only computes optimal laptime using coordinates (everything is planar). Optimal laptime is currently only computed along a centerline of the track. There is functionality for finding the minimum curvature line through an optimization, but this generates unrealistic racing lines, so optimal line calculation is still in progress.

GGV diagram example:
![GGV Diagram](assets/images/ggv.png)

Here is an example of the final optimized lap time visual for an FSAE car around Circuit of Spa-Francorchamps:
![Example 3D laptime sim](assets/images/spa_test.png)

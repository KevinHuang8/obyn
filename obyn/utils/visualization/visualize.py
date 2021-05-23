import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def plot_point_cloud(points):
    '''
    Display a 3D scatterplot of a point cloud

    points - a shape (N, 5) numpy array, where N is the number of points.
    Columns 1, 2, and 3 are X, Y, and Z coords. Column 5 is the label.
    '''

    color = points[:, 4]

    ax = plt.axes(projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2], c=color, s=5)
    plt.show()
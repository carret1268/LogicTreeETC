import matplotlib.pyplot as plt
import numpy as np

class ArrowETC:
    '''
    ArrowETC objects to hold lists of points for plotting arrows
    Differentiated from matplotlib's FancyArrowPatch by having more information stored
    about the coordinates of every single vertex, and easily being able to produce a
    multi-segmented arrow with multiple 'joints'.
    
    Required parameters:
        path -> list of tuples of float/int coords:
            Each tuple is the midpoint of the end of a line segment/arrow head in order
            with the first point being the 'butt' of the arrow and the final point being 
            the 'head' of the arrow.
        arrow_width -> float/int:
            Declares the width of your arrow in graph coordinate units
            
    Optional parameters:
        arrow_head -> Bool:
            If False, no arrow head will be made, the last point in path will be
            located at the end of jointed rectangle
            If True, an arrow head will be made with the last point in path located
            at its tip
    '''
    def __init__(self, path, arrow_width, arrow_head=False):
        self.path = path
        self.x_path = [coord[0] for coord in path]
        self.y_path = [coord[1] for coord in path]
        self.n_path = len(path)
        self.n_segments = self.n_path - 1 # actual number of line segments
        self.n_segment_vertices = 2*(1 + self.n_segments) # vertex count w/o arrow head
        self.segment_lengths = self._get_segment_length()
        if arrow_head == True:
            self.n_vertices = self.n_segment_vertices + 3 # vertex count w/ arrow head
        else:
            self.n_vertices = self.n_segment_vertices
        # find the angles each segment makes with the (+) horizontal (CCW)
        self.path_angles = self._get_angles(path=path)
        # getting angles in reverse is essential for the way vertices are calculated
        self.reverse_path_angles = self._get_angles(path=path[::-1])
        self.arrow_width = arrow_width
        self.arrow_head = arrow_head
        # verts need to wrap back around to first vertex for plotting
        verts = self._get_vertices()
        self.vertices = np.vstack((verts, verts[0]))
        self.x_vertices = self.vertices[:,0]
        self.y_vertices = self.vertices[:,1]
    
    def _get_vertices(self):
        '''
        method for determining each vertex of our segmented arrow
        moving CW around the arrow starting from (but excluding) the
        first coordinate found in the path.
        '''
        width = self.arrow_width
        path = self.path
        vertices = []
        # iterate through the path normally first, get first half of vertices
        for i in range(self.n_path-1):
            # get the next two neighboring points starting at 'butt'
            A, B = path[i], path[i+1]
            Ax, Ay = A[0], A[1]
            Bx, By = B[0], B[1]
            theta_1 = self.path_angles[i] # angle of this line segment
            # at the end of this half of vertices, there wont be an angle for next segment
            try:
                theta_2 = self.path_angles[i+1]
            except:
                theta_2 = None
            
            # first vertex is special and needs to be calculated separately
            if i == 0:
                vert = self._get_first_vertex(Ax, Ay, theta_1)
                vertices.append(vert)
            # Get the vertex
            vert = self._vertex_from_angle(Bx, By, theta_1, theta_2)
            vertices.append(vert)
        
        # generate an arrow head if desired
        if self.arrow_head:
            B = vertices[-1]
            Bx, By = B[0], B[1]
            verts = self._get_arrow_head_vertices(Bx, By, path[-1][0], path[-1][1], theta_1)
            # replace last vertex with new one to make room for arrow head
            vertices[-1] = verts[0]
            # fill in the 3 vertices of arrow head
            for point in verts[1:]:
                vertices.append(point)
    
            
        # now iterate through path backwards to get the last half of vertices
        path = path[::-1]
        for i in range(self.n_path-1):
            # get the next two neighboring points starting at 'butt'
            A, B = path[i], path[i+1]
            Ax, Ay = A[0], A[1]
            Bx, By = B[0], B[1]
            theta_1 = self.reverse_path_angles[i] # angle of this line segment
            # at the end of this half of vertices, there wont be an angle for next segment
            try:
                theta_2 = self.reverse_path_angles[i+1]
            except:
                theta_2 = None
                
            # first vertex is special and needs to be calculated separately, If we have no arrow head
            if i == 0 and not self.arrow_head:
                vert = self._get_first_vertex(Ax, Ay, theta_1)
                vertices.append(vert)
            # Get the vertex
            vert = self._vertex_from_angle(Bx, By, theta_1, theta_2)
            vertices.append(vert)
        return np.array(vertices, dtype=float)

    def _get_arrow_head_vertices(self, Bx, By, tipx, tipy, theta_1):
        '''
        Method to find the three vertices of arrow head
        Also to move the last vertex back to make room for the arrow head
        '''
        width = self.arrow_width
        arrow_head_width = width*0.55
        arrow_head_angle = 50*np.pi/180 # angle vertex0 and vert2 make with tip (vert1)
        tip_to_tip_dist = np.tan(arrow_head_angle)*(arrow_head_width + width/2)
        # 4 cases
        verts = []
        if theta_1 == 0:
            # move last vertex back
            vertx = Bx - tip_to_tip_dist
            verty = By
            vert = np.array([vertx, verty], dtype=float)
            verts.append(vert)
            # get first head vertex
            vertx = vertx
            verty = verty + arrow_head_width
            vert = np.array([vertx, verty], dtype=float)
            verts.append(vert)
            # get tip of arrow
            vertx = tipx
            verty = tipy
            vert = np.array([vertx, verty], dtype=float)
            verts.append(vert)
            # get last vertex of arrow
            vertx = verts[0][0]
            verty = verts[0][1] - width - arrow_head_width
            vert = np.array([vertx, verty], dtype=float)
            verts.append(vert)
            # complete the cycle putting us back on the base arrow body
            vertx = vertx
            verty = verty + arrow_head_width
            vert = np.array([vertx, verty], dtype=float)
            verts.append(vert)
        elif theta_1 == np.pi/2:
            # move last vertex down
            vertx = Bx
            verty = By - tip_to_tip_dist
            vert = np.array([vertx, verty], dtype=float)
            verts.append(vert)
            # get first head vertex
            vertx = vertx - arrow_head_width
            verty = verty
            vert = np.array([vertx, verty], dtype=float)
            verts.append(vert)
            # get tip of arrow
            vertx = tipx
            verty = tipy
            vert = np.array([vertx, verty], dtype=float)
            verts.append(vert)
            # get last vertex of arrow
            vertx = verts[0][0] + width + arrow_head_width
            verty = verts[0][1]
            vert = np.array([vertx, verty], dtype=float)
            verts.append(vert)
            # complete the cycle putting us back on the base arrow body
            vertx = vertx - arrow_head_width
            verty = verty
            vert = np.array([vertx, verty], dtype=float)
            verts.append(vert)
        elif theta_1 == np.pi:
            # move last vertex back
            vertx = Bx + tip_to_tip_dist
            verty = By
            vert = np.array([vertx, verty], dtype=float)
            verts.append(vert)
            # get first head vertex
            vertx = vertx
            verty = verty - arrow_head_width
            vert = np.array([vertx, verty], dtype=float)
            verts.append(vert)
            # get tip of arrow
            vertx = tipx
            verty = tipy
            vert = np.array([vertx, verty], dtype=float)
            verts.append(vert)
            # get last vertex of arrow
            vertx = verts[0][0]
            verty = verts[0][1] + width + arrow_head_width
            vert = np.array([vertx, verty], dtype=float)
            verts.append(vert)
            # complete the cycle putting us back on the base arrow body
            vertx = vertx
            verty = verty - arrow_head_width
            vert = np.array([vertx, verty], dtype=float)
            verts.append(vert)
        elif theta_1 == 3*np.pi/2:
            # move last vertex back
            vertx = Bx
            verty = By + tip_to_tip_dist
            vert = np.array([vertx, verty], dtype=float)
            verts.append(vert)
            # get first head vertex
            vertx = vertx + arrow_head_width
            verty = verty
            vert = np.array([vertx, verty], dtype=float)
            verts.append(vert)
            # get tip of arrow
            vertx = tipx
            verty = tipy
            vert = np.array([vertx, verty], dtype=float)
            verts.append(vert)
            # get last vertex of arrow
            vertx = verts[0][0] - width - arrow_head_width
            verty = verts[0][1]
            vert = np.array([vertx, verty], dtype=float)
            verts.append(vert)
            # complete the cycle putting us back on the base arrow body
            vertx = vertx + arrow_head_width
            verty = verty
            vert = np.array([vertx, verty], dtype=float)
            verts.append(vert)
        return verts
    
    def _get_first_vertex(self, Ax, Ay, theta_1):
        '''
        Method to find the first vertex in a pass through segments
        to be used in _get_vertices method
        '''
        width = self.arrow_width
        if theta_1 == 0:
            vertx = Ax
            verty = Ay + width/2
            vert = np.array([vertx, verty], dtype=float)
        elif theta_1 == np.pi/2:
            vertx = Ax - width/2
            verty = Ay
            vert = np.array([vertx, verty], dtype=float)
        elif theta_1 == np.pi:
            vertx = Ax
            verty = Ay - width/2
            vert = np.array([vertx, verty], dtype=float)
        elif theta_1 == 3*np.pi/2:
            vertx = Ax + width/2
            verty = Ay
            vert = np.array([vertx, verty], dtype=float)    
        return vert
    
            
    def _vertex_from_angle(self, Bx, By, theta_1, theta_2):
        '''
        Method to find vertex from segment end points and two angles
        to be used in _get_vertices method
        '''
        width = self.arrow_width
        # first when there is a segment ahead
        if ((theta_1 == 0) and (theta_2 == np.pi/2)) or ((theta_1 == np.pi/2) and (theta_2 == 0)):
            vertx = Bx - width/2
            verty = By + width/2
            vert = np.array([vertx, verty], dtype=float)
        elif ((theta_1 == 0) and (theta_2 == 3*np.pi/2)) or ((theta_1 == 3*np.pi/2) and (theta_2 == 0)):
            vertx = Bx + width/2
            verty = By + width/2
            vert = np.array([vertx, verty], dtype=float)
        elif ((theta_1 == np.pi) and (theta_2 == np.pi/2)) or ((theta_1 == np.pi/2) and (theta_2 == np.pi)):
            vertx = Bx - width/2
            verty = By - width/2
            vert = np.array([vertx, verty], dtype=float)
        elif ((theta_1 == np.pi) and (theta_2 == 3*np.pi/2)) or ((theta_1 == 3*np.pi/2) and (theta_2 == np.pi)):
            vertx = Bx + width/2
            verty = By - width/2
            vert = np.array([vertx, verty], dtype=float)
        # now if there is no segment ahead
        elif (theta_1 == 0) and (theta_2 is None):
            vertx = Bx
            verty = By + width/2
            vert = np.array([vertx, verty], dtype=float)
        elif (theta_1 == np.pi/2) and (theta_2 is None):
            vertx = Bx - width/2
            verty = By
            vert = np.array([vertx, verty], dtype=float)
        elif (theta_1 == np.pi) and (theta_2 is None):
            vertx = Bx
            verty = By - width/2
            vert = np.array([vertx, verty], dtype=float)
        elif (theta_1 == 3*np.pi/2) and (theta_2 is None):
            vertx = Bx + width/2
            verty = By
            vert = np.array([vertx, verty], dtype=float)
        return vert
    
    def _get_angles(self, path):
        '''
        method for determining the angles from positive x-axis for each line
        segment declared with the path
        '''
        angles = []
        for i in range(self.n_segments):
            # get the next two neighboring points starting at 'butt'
            p1, p2 = path[i], path[i+1]
            x1, y1 = p1[0], p1[1]
            x2, y2 = p2[0], p2[1]
            
            # for (+) horizontal line
            if (x2 > x1) and (y2 == y1):
                theta = 0
            # for (+) vertical line
            elif (x2 == x1) and (y2 > y1):
                theta = np.pi/2
            # for (-) horizontal line
            elif (x2 < x1) and (y2 == y1):
                theta = np.pi
            # for (-) vertical line
            elif (x2 == x1) and (y2 < y1):
                theta = 3*np.pi/2
            # Get the angles starting with Quadrant 1
            elif (x2 > x1) and (y2 > y1):
                theta = np.arctan((y2-y1)/(x2-x1))
            # Quadrant 2
            elif (x2 < x1) and (y2 > y1):
                # start with angle CCW from (+) vertical
                phi = np.arctan((x2-x1)/(y2-y1))
                theta = np.pi/2 + abs(phi)
            # Quadrant 3
            elif (x2 < x1) and (y2 < y1):
                # start with angle CCW from (-) horizontal
                phi = np.arctan((y2-y1)/(x2-x1))
                theta = np.pi + abs(phi)
            # Quadrant 4
            else:
                # start with angle CW from (+) horizontal
                phi = np.arctan((y2-y1)/(x2-x1))
                theta = 2*np.pi - abs(phi)
            # throw an error for non-Right angles
            if (theta != 0) and (theta != np.pi) and (theta != np.pi/2) and (theta != 3*np.pi/2):
                raise Exception(f'The arrow path must be limited to straight line segments falling on +/- x and y axes\nThe segment between point {p1} and point {p2} do not fall on an axis')
            angles.append(theta)
        return angles
    
    def _get_segment_length(self):
        '''
        method for determining the length of each segment
        '''
        distances = []
        for i in range(self.n_segments):
            p1, p2 = self.path[i], self.path[i+1]
            x1, y1 = p1[0], p1[1]
            x2, y2 = p2[0], p2[1]
            d = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            distances.append(d)
        return distances
    
    def show_arrow(self, ec='white', fc='cyan', lw=0.6):
        '''
        method for producing a matplotlib.pyplot plot of arrow
        '''
        x = self.x_vertices
        y = self.y_vertices
        # generate figure and axis to put boxes in
        fig = plt.figure(figsize=(8,8), frameon=True, facecolor='black')
        ax = plt.axes()
        ax.axis('off')
        plt.gca().set_aspect('equal')
        # set axis bounds
        xdiff = (max(x) - min(x))*0.2
        ydiff = (max(y) - min(y))*0.2
        xmin = min(x) - xdiff
        xmax = max(x) + xdiff
        ymin = min(y) - ydiff
        ymax = max(y) + ydiff
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)
        # plot lines and vertices
        ax.plot(x, y, color=ec, lw=lw, zorder=100)
        ax.fill(x, y, color=fc)
        
            

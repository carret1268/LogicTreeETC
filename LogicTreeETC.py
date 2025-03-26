from matplotlib.patches import BoxStyle
import matplotlib.pyplot as plt

from ArrowETC import ArrowETC
from LogicBoxETC import LogicBox

class LogicTree:
    '''
    Class for building logic tree graphs by placing LogicBox objects and connecting them
    with lines/arrows. Uses ArrowETC objects for connecting boxes which are particularly 
    nice as they (1) use data dimensions for their width, (2) can just be given a list of 
    vertices to pass arrows/lines through (limited to 90degree bends), and (3) have robust
    attributes about dimensions and vertex positions. 
    Built on top of matplotlib. 
    Latex can be used to render text and if turned on (when calling add_box() method) one must 
    have the {bm} package for allowing math symbols to be bolded, {amsmath} package for a 
    variety of math options including embedding text within inline math, {soul} for underlining
    with a rich set of features, and {relsize} to make math symbols larger since smaller symbols
    such as \mu become much smaller than the surrounding text.
    
    Start by initializing a LogicTree object, add boxes using the add_box() method, then 
    connect your boxes with the add_connection() and add_connection_biSplit() methods
    
    Optional Parmeters:
        fig_size -> tuple of floats/ints:
            Determines the figure size of matplotlib figure (x, y)
        xlims -> tuple of floats/ints:
            Determines the minimum and maximum x values used for plotting on the 
            matplotlib axis (x_min, x_max)
            Can cause weird stretching if xlims != ylims
        ylims -> tuple of floats/ints:
            Determines the minimum and maximum y values used for plotting on the 
            matplotlib axis (y_min, y_max)
            Can cause weird stretching if xlims != ylims
        fig_fc -> str:
            Determines the background color of figure
        title -> str:
            Variable stored for placing a title on the figure using the make_title()
            method. Can be changed when calling make_title()
        font_dict -> dictionary:
            A dictionary of parameters to determine the fonts used for general text,
            this may be changed anytime one calls the add_box() method
            See matplotlib documentation for more options
            https://matplotlib.org/stable/api/text_api.html#matplotlib.text.Text
        font_dict_title -> dicitonary:
            A dicionary of parameters to determine the font used for the figure title
            See matplotlib documentation for more options
            https://matplotlib.org/stable/api/text_api.html#matplotlib.text.Text
        text_color -> str:
            A string that determines the color of text. If not set as text_color=None,
            it will override any color key set in font_dict
        title_color -> str:
            A string that determines the color of your title. If not set as title_color=None,
            it will override any color key set in font_dict_title
    '''
    def __init__(self, fig_size=(9,9), xlims=(0,100), ylims=(0,100), fig_fc='black', \
                title=None, font_dict=None, font_dict_title=None, text_color=None, title_color=None):
        # dictionary to store boxes by their identifier to get there dims/coords
        self.boxes = {}  
        self.title = title
        self.xlims = xlims
        self.ylims = ylims
        
        # font dictionary for title
        if font_dict_title is None:
            font_dict_title = dict(fontname='Sitka', fontsize=34, color='white')
        if title_color is not None:
            font_dict_title['color'] = title_color
        self.title_font = font_dict_title
        
        # default fontdict
        if font_dict is None:
            font_dict = {
                'fontname': 'Leelawadee UI',
                'fontsize': 14,
                'color': 'white'
            }
        if text_color is not None:
            font_dict['color'] = text_color
        self.font_dict = font_dict
        
        
        # underlining options for latex rendering
        self.latex_ul_depth = '1pt'
        self.latex_ul_width = '1pt'
        
        # generate figure and axis to put boxes in
        fig = plt.figure(figsize=fig_size, frameon=True, facecolor=fig_fc)
        ax = plt.axes()

        # set axes limits remove ticks and spines from axis (blank figure)
        ax.set_xlim(xlims[0], xlims[1])
        ax.set_ylim(ylims[0], ylims[1])
        ax.axis('off')
        fig.canvas.draw_idle() # this is needed to use bbox.get_window_extent() method for txtbox sizes
        self.fig = fig
        self.ax = ax
        
    def _get_pathsForBi_left_then_right(self, Ax2, Ay2, left_box, right_box, tip_offset):
        '''
        "private" method to be used for generating the paths of a 4vertex connection.
        For using in the add_connection_biSplit() method. The center of left_box must
        be to the left of the center of right_box
        
        Required parameters:
            Ax2 -> float/int:
                x position of the tip of previous line (starting position of butt of current)
            Ay2 -> float/int:
                y position of the tip of previous line (starting position of butt of current)
            left_box -> LogicBox object:
                the LogicBox object whose center is to the left of right_box
            right_box -> LogicBox objectL
                the LogicBox object whoes center is to the right of left_box
            tip_offset -> float/int:
                Arrow tips sometimes overlap with text bbox's depending on figuresize,
                x and y lims, fontsize, etc..., so tip_offset tells how much
                to offset the arrow tip
        '''
        # create the leftward arrow
        Lx1 = Ax2
        Ly1 = Ay2
        Lx2 = left_box.x_center
        Ly2 = Ly1
        Lx3 = Lx2
        # if left_box is below boxA
        if Ay2 > left_box.y_center:
            Ly3 = left_box.yTop + tip_offset
        # otherwise it must be above boxA
        else:
            Ly3 = left_box.yBottom - tip_offset
        # creat the rightward arrow
        Rx1 = Ax2
        Ry1 = Ay2
        Rx2 = right_box.x_center
        Ry2 = Ry1
        Rx3 = Rx2
        # if right_box is below boxA
        if Ay2 > right_box.y_center:
            Ry3 = right_box.yTop + tip_offset
        # otherwise it must be above boxA
        else:
            Ry3 = right_box.yBottom - tip_offset
        # set paths
        path_left = [(Lx1, Ly1), (Lx2, Ly2), (Lx3, Ly3)]
        path_right = [(Rx1, Ry1), (Rx2, Ry2), (Rx3, Ry3)]

        return path_left, path_right    

    def add_box(self, xpos, ypos, text, box_name, bbox_fc, bbox_ec, font_dict=None, \
                text_color=None, fs=None, font_weight=None, lw=1.6, \
                bbox_style=BoxStyle('Round', pad=0.6), va='center', ha='right', \
                use_tex_rendering=False, ul=False, ul_depth_width=None):
        
        '''
        method for adding a box to the LogicTree using matplotlib.text
        
        Required parameters:
            xpos -> float/int:
                determines the x position of the box
            ypos -> float/int:
                determines the y position of the box
            text -> str:
                the string of text that will be shown inside of the box. Latex rendering
                may be used if use_tex_rendering is set to True
            box_name -> str:
                a unique identifier for your LogicBox to be stored in the boxes attribute
                of your LogicTree object. This identifier must be used to make connections
                with the provided methods
            bbox_fc -> str:
                the face color of your text box. Use RGBA values if you want a transparent
                box. E.g., bbox_fc=(0,0,0,0)
            bbox_ec -> str:
                the edge color of your text box. Use RGBA values if you want a transparent
                box. E.g., bbox_fc=(0,0,0,0)
            
        Optional Parameters:
            font_dict -> dictionary:
                A dictionary of parameters to determine the fonts used for your text
                See matplotlib documentation for more options
                https://matplotlib.org/stable/api/text_api.html#matplotlib.text.Text
            text_color -> str:
                A string that determines the color of text. If not set as text_color=None,
                it will override any color key set in font_dict
            fs -> int:
                Determine the fontsize of your text. Overrides fontsize provided in font_dict
            font_weight -> str:
                Determines the font weight of your text. Overrides the weight provided in font_dict
                Options: ['normal', 'bold', 'heavy', 'light', 'ultrabold', 'ultralight']
            lw -> float/int:
                determines the linewidth of the bbox surrounding your text
                bbox_style -> BoxStyle object:
                matplotlib.patches.BoxStyle object for box shape and padding
                https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.BoxStyle.html
            va -> str:
                Determines the vertical alignment of your text box.
                Options: ['left', 'center', 'right']
            ha -> str:
                Determines the horizontal alignment of your text box
                Options: ['top', 'center', 'bottom']
            use_tex_rendering -> bool:
                Determines whether or not to include a latex preamble in generating your text.
                If set to True, one must have the {bm} package for allowing math symbols to be 
                bolded, {amsmath} package for a variety of math options including embedding text 
                within inline math, {soul} for underlining with a rich set of features, 
                and {relsize} to make math symbols larger since smaller symbols such as \mu 
                become much smaller than the surrounding text.
            ul -> bool:
                Option used to underline text if use_tex_rendering is True
            ul_depth_width -> tuple of floats/ints:
                A tuple that determines the depth and width of your underlining if
                use_tex_rendering is True and ul is True. (depth, width)                
        '''
        # option to use latex rendering (minimal font options with latex, so not default)
        if use_tex_rendering:
            # our latex preamble for importing latex packages and making a command
            # \bigsymbol{} for enlarging latex math symbols
            latex_preamble = f'''\\usepackage{{bm}}
                                 \\usepackage{{amsmath}}
                                 \\usepackage{{soul}}
                                 \\setul{{2pt}}{{1pt}}
                                 \\usepackage{{relsize}}
                                 \\newcommand{{\\bigsymbol}}[1]{{\\mathlarger{{\\mathlarger{{\\mathlarger{{#1}}}}}}}}
                                 '''
            # update rcParams to use latex
            plt.rcParams.update({
                "text.usetex": True,
                "font.family": "cm",
                'text.latex.preamble': latex_preamble
            })
        else:
            plt.rcParams.update({"text.usetex": False})
            
        # set fontidct of not provided
        if font_dict is None:
            font_dict = self.font_dict
        # if specific text color is specified, change it in font_dict
        if text_color is not None:
            font_dict['color'] = text_color
        # if specific fontsize is specified, change it in font_dict
        if fs is not None:
            font_dict['fontsize'] = fs
        # if weight is specified, change it in font_dict
        if font_weight is not None:
            font_dict['weight'] = font_weight
            
        # create a logicBox object which stores all of this information
        myBox = LogicBox(xpos=xpos, ypos=ypos, text=text, box_name=box_name, bbox_fc=bbox_fc, \
                        bbox_ec=bbox_ec, bbox_style=bbox_style, font_dict=font_dict, \
                        va=va, ha=ha, lw=lw)
        
        # add latex commands to text for underlining 
        if use_tex_rendering and (ul or ul_depth_width is not None):
            text_str = r'\ul{' + myBox.text + r'}'
            # if underlining parameters are set, add the command to change them
            if ul_depth_width is not None:
                text_str = f'\\setul{{{ul_depth_width[0]}}}{{{ul_depth_width[1]}}}' + text_str
        else:
            text_str = myBox.text
        # make the text
        txt = self.ax.text(x=myBox.x, y=myBox.y, s=text_str, fontdict=myBox.font_dict, \
                      bbox=myBox.style, va=myBox.va, ha=myBox.ha)
        
        # get our box's dims and edge positions to store in myBox object
        bbox = plt.gca().transData.inverted().transform_bbox(txt.get_window_extent(renderer=self.fig.canvas.get_renderer())) # coords of text
        wpad = txt.get_bbox_patch().get_extents().width # pad size for width
        hpad = txt.get_bbox_patch().get_extents().height # pad size for height
        myBox.xLeft, myBox.xRight = bbox.x0 - wpad, bbox.x1 + wpad
        myBox.yBottom, myBox.yTop = bbox.y0 - hpad, bbox.y1 + wpad
        myBox.width = myBox.xRight - myBox.xLeft
        myBox.height = myBox.yTop - myBox.yBottom
        myBox.x_center = myBox.xRight - myBox.width/2
        myBox.y_center = myBox.yTop - myBox.height/2
        
        # store box in our LogicTree object's box dictionary to grab dimensions when needed
        self.boxes[myBox.name] = myBox
        
        
    def add_connection_biSplit(self, boxA, boxB, boxC, arrow_head=True, arrow_width=0.5, \
                          fill_connection=True, fc_A=None, ec_A=None, fc_B=None, ec_B=None, \
                          fc_C=None, ec_C=None, lw=0.5, butt_offset=0, tip_offset=0):
        '''
        Method for adding a connection between three boxes with a butt at boxA
        and tips at both boxB and boxC. Only works for top and bottom connections
        at the horizontal center of the boxes. The connection is made with and
        ArrowETC object.
        
        Required parameters:
            boxA -> LogicBox object:
                The parent LogicBox object that you want to connect to boxB and boxC,
                with the butt of the connector (ArrowETC object) on the top or bottom 
                of boxA -- boxA must be located either above both boxB and boxC OR
                below both boxB and boxC
            boxB -> LogicBox object:
                The first child connected to boxA, with a tip at the top or bottom of boxB.
                Must be above or below boxA
            boxC -> LogicBox object:
                The second child connected to boxA, with a tip at the top or bottom of boxC.
                Must be above or below boxA
            arrow_head -> bool:
                Determines if the connection with have arrow heads pointing to boxB and boxC
            arrow_width -> float/int:
                Determines the width of the connector in data dimension units
            fill_connection -> bool:
                Determines if the arrows will be filled with color or not
            fc_A -> str:
                Determines the fill color of the first part of the connection stemming 
                from boxA. If fc_A=None, fc_A will equal the face color of boxA.
                If fc_='ec', fc_A will equal the edge color of boxA
            ec_A -> str:
                Determines the edge color of the first part of the connection stemming
                from boxA. If ec_A=None, ec_A will equal the edge color of boxA.
                If ec_A='fc', ec_A will equal the face color of boxA
            fc_B -> str:
                Determines the fill color of the part of the connection leading to boxB. 
                If fc_B=None, fc_B will equal the face color of boxB.
                If fc_B='ec', fc_B will equal the edge color of boxB
            ec_B -> str:
                Determines the edge color of the part of the connection leading to boxB. 
                If ec_B=None, ec_B will equal the edge color of boxB.
                If ec_B='fc', ec_B will equal the face color of boxB
            fc_C -> str:
                Determines the fill color of the part of the connection leading to boxC. 
                If fc_C=None, fc_C will equal the face color of boxC.
                If fc_C='ec', fc_C will equal the edge color of boxC
            ec_C -> str:
                Determines the edge color of the part of the connection leading to boxC. 
                If ec_C=None, ec_C will equal the edge color of boxC.
                If ec_C='fc', ec_C will equal the face color of boxC
            lw -> float/int:
                The linewidth of the connection edges
            butt_offset -> float/int:
                Arrow tips sometimes overlap with text bbox's depending on figuresize,
                x and y lims, fontsize, etc..., so butt_offset tells how much
                to offset the arrow butt
            tip_offset -> float/int:
                Arrow tips sometimes overlap with text bbox's depending on figuresize,
                x and y lims, fontsize, etc..., so tip_offset tells how much
                to offset the arrow tip
        '''
        # do stylizing stuff
        if fill_connection:
            # option for face color to equal edgecolor
            if fc_A == 'ec':
                fc_A = boxA.edge_color
            # if no option specified, face color of arrow is same as face color of box
            elif fc_A is None:
                fc_A = boxA.face_color
            # option for face color to equal edgecolor
            if fc_B == 'ec':
                fc_B = boxB.edge_color
            # if no option specified, face color of arrow is same as face color of box
            elif fc_B is None:
                fc_B = boxB.face_color
            # option for face color to equal edgecolor
            if fc_C == 'ec':
                fc_C = boxC.edge_color
            # if no option specified, face color of arrow is same as face color of box
            elif fc_C is None:
                fc_C = boxC.face_color
        
        if ec_A =='fc':
            ec_A = boxA.face_color
        elif ec_A is None:
            ec_A = boxA.edge_color
        if ec_B =='fc':
            ec_B = boxB.face_color
        elif ec_B is None:
            ec_B = boxB.edge_color
        if ec_C =='fc':
            ec_C = boxC.face_color
        elif ec_C is None:
            ec_C = boxC.edge_color
        
        # first take the case of boxA being above boxes B and C
        if (boxA.y_center > boxB.y_center) and (boxA.y_center > boxC.y_center):
            # create the downward line from BoxA to center
            Ax1 = boxA.x_center
            Ay1 = boxA.yBottom - butt_offset
            Ax2 = Ax1
            # take it down to the midpoint of boxA and the highest of boxes B and C
            if boxB.yTop >= boxC.yTop:
                Ay2 = (Ay1 + boxB.yTop)/2
            else:
                Ay2 = (Ay1 + boxC.yTop)/2
            # set path for downward segment
            path = [(Ax1, Ay1), (Ax2, Ay2)]
            arrow = ArrowETC(path=path, arrow_head=False, arrow_width=arrow_width)

            # get vertices
            x = arrow.x_vertices[:-1]
            y = arrow.y_vertices[:-1]
            self.ax.plot(x, y, color=ec_A, lw=0.01)
            # fill arrow if desired
            if fill_connection:
                self.ax.fill(x, y, color=fc_A)
                
            # take the case that boxB is to the left of boxC 
            if boxB.x_center < boxC.x_center:
                # get paths
                path_left, path_right = self._get_pathsForBi_left_then_right(Ax2, Ay2, left_box=boxB, \
                                                                             right_box=boxC, tip_offset=tip_offset)
                # make left arrow
                arrow = ArrowETC(path=path_left, arrow_head=True, arrow_width=arrow_width)
                # get vertices
                x = arrow.x_vertices[:-1]
                y = arrow.y_vertices[:-1]
                self.ax.plot(x, y, color=ec_B, lw=lw)
                # fill arrow if desired
                if fill_connection:
                    self.ax.fill(x, y, color=fc_B)
                
                # make right arrow
                arrow = ArrowETC(path=path_right, arrow_head=True, arrow_width=arrow_width)
                # get vertices
                x = arrow.x_vertices[:-1]
                y = arrow.y_vertices[:-1]
                self.ax.plot(x, y, color=ec_C, lw=lw)
                # fill arrow if desired
                if fill_connection:
                    self.ax.fill(x, y, color=fc_C)
                    
            # take the case that boxB is to the right of boxC 
            elif boxB.x_center > boxC.x_center:
                # get paths
                path_left, path_right = self._get_pathsForBi_left_then_right(Ax2, Ay2, left_box=boxC, right_box=boxB, tip_offset=tip_offset)
                # make left arrow
                arrow = ArrowETC(path=path_left, arrow_head=True, arrow_width=arrow_width)
                # get vertices
                x = arrow.x_vertices[:-1]
                y = arrow.y_vertices[:-1]
                self.ax.plot(x, y, color=ec_C, lw=lw)
                # fill arrow if desired
                if fill_connection:
                    self.ax.fill(x, y, color=fc_C)
                
                # make right arrow
                arrow = ArrowETC(path=path_right, arrow_head=True, arrow_width=arrow_width)
                # get vertices
                x = arrow.x_vertices[:-1]
                y = arrow.y_vertices[:-1]
                self.ax.plot(x, y, color=ec_B, lw=lw)
                # fill arrow if desired
                if fill_connection:
                    self.ax.fill(x, y, color=fc_B)
                
        # now take the case of boxA being below boxes B and C
        elif (boxA.y_center < boxB.y_center) and (boxA.y_center < boxC.y_center):
            # create the upward line from BoxA to center
            Ax1 = boxA.x_center
            Ay1 = boxA.yTop + butt_offset
            Ax2 = Ax1
            # take it down to the midpoint of boxA and the highest of boxes B and C
            if boxB.yBottom <= boxC.yBottom:
                Ay2 = (Ay1 + boxB.yBottom)/2
            else:
                Ay2 = (Ay1 + boxC.yBottom)/2
            # set path for downward segment
            path = [(Ax1, Ay1), (Ax2, Ay2)]
            arrow = ArrowETC(path=path, arrow_head=False, arrow_width=arrow_width)

            # get vertices
            x = arrow.x_vertices[:-1]
            y = arrow.y_vertices[:-1]
            self.ax.plot(x, y, color=ec_A, lw=0.01)
            # fill arrow if desired
            if fill_connection:
                self.ax.fill(x, y, color=fc_A)

            # take the case that boxB is to the left of boxC 
            if boxB.x_center < boxC.x_center:
                # get paths
                path_left, path_right = self._get_pathsForBi_left_then_right(Ax2, Ay2, left_box=boxB, \
                                                                             right_box=boxC, tip_offset=tip_offset)
                # make left arrow
                arrow = ArrowETC(path=path_left, arrow_head=True, arrow_width=arrow_width)
                # get vertices
                x = arrow.x_vertices[:-1]
                y = arrow.y_vertices[:-1]
                self.ax.plot(x, y, color=ec_B, lw=lw)
                # fill arrow if desired
                if fill_connection:
                    self.ax.fill(x, y, color=fc_B)
                
                # make right arrow
                arrow = ArrowETC(path=path_right, arrow_head=True, arrow_width=arrow_width)
                # get vertices
                x = arrow.x_vertices[:-1]
                y = arrow.y_vertices[:-1]
                self.ax.plot(x, y, color=ec_C, lw=lw)
                # fill arrow if desired
                if fill_connection:
                    self.ax.fill(x, y, color=fc_C)
                    
            # take the case that boxB is to the right of boxC 
            elif boxB.x_center > boxC.x_center:
                # get paths
                path_left, path_right = self._get_pathsForBi_left_then_right(Ax2, Ay2, left_box=boxC, right_box=boxB, tip_offset=tip_offset)
                # make left arrow
                arrow = ArrowETC(path=path_left, arrow_head=True, arrow_width=arrow_width)
                # get vertices
                x = arrow.x_vertices[:-1]
                y = arrow.y_vertices[:-1]
                self.ax.plot(x, y, color=ec_C, lw=lw)
                # fill arrow if desired
                if fill_connection:
                    self.ax.fill(x, y, color=fc_C)
                
                # make right arrow
                arrow = ArrowETC(path=path_right, arrow_head=True, arrow_width=arrow_width)
                # get vertices
                x = arrow.x_vertices[:-1]
                y = arrow.y_vertices[:-1]
                self.ax.plot(x, y, color=ec_B, lw=lw)
                # fill arrow if desired
                if fill_connection:
                    self.ax.fill(x, y, color=fc_B)
            
    def add_connection(self, boxA, boxB, arrow_head=True, arrow_width=0.5, fill_connection=True, \
                             butt_offset=0, tip_offset=0, fc=None, ec=None, lw=0.7):
        '''
        Makes a segmented arrow from boxA to boxB. Only works for straight arrows when
        boxA and boxB lie on the same axis.
        
        Required Parameters:
            boxA -> LogicBox object:
                The LogicBox object where the butt of your arrow will stem from
            boxB -> LogicBox objectL
                The LogicBoc object where the tip of your arrow will point to

        Optional Parameters:
            arrow_head -> bool:
                Determines if the connection with have arrow heads pointing to boxB and boxC
            arrow_width -> float/int:
                Determines the width of the connector in data dimension units
            fill_connection -> bool:
                Determines if the arrows will be filled with color or not
            butt_offset -> float/int:
                Arrow tips sometimes overlap with text bbox's depending on figuresize,
                x and y lims, fontsize, etc..., so butt_offset tells how much
                to offset the arrow butt
            tip_offset -> float/int:
                Arrow tips sometimes overlap with text bbox's depending on figuresize,
                x and y lims, fontsize, etc..., so tip_offset tells how much
                to offset the arrow tip
            fc -> str:
                Determines the fill color of the connection. If fc=None, fc will equal 
                the face color of boxB. If fc='ec', fc will equal the edge color of boxB
            ec -> str:
                Determines the edge color of the connection. If ec=None, ec will equal 
                the face color of boxB. If ec='fc', ec will equal the face color of boxB
        '''
        # handle colors
        if fill_connection:
            # if no fc is chosen, take the fc of connection to be fc of boxB
            if fc is None or fc == 'fc':
                fc = boxB.face_color
            elif fc == 'ec':
                fc = boxB.edge_color
        # if no ec is chosen, take ec of connection to be ec of boxB
        if ec is None or ec == 'ec':
            ec = boxB.edge_color
        elif ec == 'fc':
            ec = boxB.face_color
            
        # first case, boxA and boxB are on the same row
        if boxA.y_center == boxB.y_center:
            # boxA is to the left of boxB
            if boxA.x_center < boxB.x_center:
                Ax, Ay = boxA.xRight + butt_offset, boxA.y_center
                Bx, By = boxB.xLeft - tip_offset, boxB.y_center
            # boxA is to the right of boxB
            elif boxA.x_center > boxB.x_center:
                Ax, Ay = boxA.xLeft - butt_offset, boxA.y_center
                Bx, By = boxB.xRight + tip_offset, boxB.y_center
            path = [(Ax, Ay), (Bx, By)]
        # second case, boxA is below boxB
        elif boxA.y_center < boxB.y_center:
            # same column
            if boxA.x_center == boxB.x_center:
                Ax, Ay = boxA.x_center, boxA.yTop + butt_offset
                Bx, By = boxB.x_center, boxB.yBottom - tip_offset
                path = [(Ax, Ay), (Bx, By)]
            # boxes are offset in the x-axis
            else:
                Ax, Ay =  boxA.x_center, boxA.yTop + butt_offset
                Bx = boxB.x_center
                By = (boxB.yBottom + boxA.yTop)/2
                Cx, Cy = Bx, boxB.yBottom - tip_offset
                path = [(Ax, Ay), (Bx, By), (Cx, Cy)]
        # third case, boxA is above boxB
        elif boxA.y_center > boxB.y_center:
            # same column
            if boxA.x_center == boxB.x_center:
                Ax, Ay = boxA.x_center, boxA.yBottom - butt_offset
                Bx, By = boxB.x_center, boxB.yTop + tip_offset
                path = [(Ax, Ay), (Bx, By)]
            # boxes are offset in the x-axis
            else:
                Ax, Ay =  boxA.x_center, boxA.yBottom - butt_offset
                Bx = boxA.x_center
                By = (boxB.yTop + boxA.yBottom)/2
                Cx, Cy = boxB.x_center, By
                Dx, Dy = Cx, boxB.yTop + tip_offset
                path = [(Ax, Ay), (Bx, By), (Cx, Cy), (Dx, Dy)]
                
        # create arrow object and 
        arrow = ArrowETC(path=path, arrow_head=arrow_head, arrow_width=arrow_width)
        x = arrow.x_vertices
        y = arrow.y_vertices
        self.ax.plot(x, y, color=ec, lw=lw)
        # fill arrow if desired
        if fill_connection:
            self.ax.fill(x, y, color=fc)
            
    def make_title(self, pos='left', consider_box_x=True, new_title=None):
        '''
        Method for placing a title on your LogicTree in the top left, top right, or top center.
        It is important to place your box objects below the ymax value in ylims, otherwise
        the title might overlap with your content.
        
        Optional Parameters:
            pos -> str:
                Decides the horizontal alignment of the title.
                Options: ['left', 'right', 'center']
            consider_box_x -> bool:
                Determines whether the horizontal position of the title is determined
                by the values in xlims, or if it determined by the positions of your
                LogicBox objects.
            new_title -> str:
                Lets you set a new title for your LogicTree
        '''
        if new_title is not None:
            self.title = new_title
        
        # if we are to ignore consider_box_x, use xlims to find the horizontal placement of title
        if not consider_box_x:
            if pos == 'left':
                ha = 'left'
                x = self.xlims[0]
            elif pos == 'center':
                ha = 'center'
                x = (self.xlims[1] + self.xlims[0])/2
            elif pos == 'right':
                ha = 'right'
                x = self.xlims[1]
            else:
                raise Exception("pos paramter should have value of ['left', 'right', 'center']")
        
        # if we are to consider_box_x
        else:
            xFarLeft = 0
            xFarRight = 0
            for box in self.boxes:
                if self.boxes[box].xLeft < xFarLeft:
                    xFarLeft = self.boxes[box].xLeft
                if self.boxes[box].xRight > xFarRight:
                    xFarRight = self.boxes[box].xRight
            if pos == 'left':
                ha = 'left'
                x = xFarLeft
            elif pos == 'right':
                ha = 'right'
                x = xFarRight
            elif pos == 'center':
                ha = 'center'
                x = (xFarRight + xFarLeft)/2
            else:
                raise Exception("pos paramter should have value of ['left', 'right', 'center']")
        
        # finally make the title
        self.ax.text(x=x, y=self.ylims[1], s=self.title, va='top', ha=ha, fontdict=self.title_font)
                
    def save_as_png(self, file_name, dpi=800):
        '''
        Lets your save your LogicTree as a PNG file
        
        Required Parameters:
            file_name -> str:
                The path/name of your output png file
        
        Optional Parameters:
            dpi -> int:
                The dpi (resolution) of your output png file. A higher number leads to a
                clearer image, but a larger file size
        '''
        self.fig.savefig(file_name, dpi=dpi, bbox_inches='tight')
       




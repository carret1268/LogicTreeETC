from matplotlib.patches import BoxStyle

class LogicBox:
    '''
    class for building a box for LogicTree
    
    Required parameters:
        xpos -> float/int:
            The x position of left, right, or center of box determined by ha
        ypos -> float/int:
            The y position of top, bottom, or center of box determined by va
        text -> str:
            The text that will be displayed within the box
        box_name -> str:
            An identifying name for the box for referencing the LogicBox object after 
            initialization
        bbox_fc -> str:
            Face color of box
        bbox_ec -> str:
            Edge color of box
        font_dict -> dictionary:
            Text dictionary for styling text parameter. See matplotlip text parameters:
            https://matplotlib.org/stable/api/text_api.html#matplotlib.text.Text
            
    Optional Parameters:
        bbox_style -> BoxStyle object:
            matplotlib.patches.BoxStyle object for box shape and padding
            https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.BoxStyle.html
        lw -> float/int:
            Line width of outline/edge of box
        va -> str:
            Vertical alignment with values ['left', 'center', 'right']
        ha -> str:
            Horizontal alignment with values ['top', 'center', 'bottom']
    '''
    def __init__(self, xpos, ypos, text, box_name, bbox_fc, bbox_ec, font_dict, \
                 bbox_style=BoxStyle('Square', pad=0.5), lw=1.6, va='center', ha='left'):
        # create a bbox style object for styling text box
        my_style = self._my_bbox_style(facecolor=bbox_fc, edgecolor=bbox_ec, linewidth=lw, \
                                     boxstyle=bbox_style)
        
        self.x = xpos
        self.y = ypos
        self.text = text
        self.name = box_name
        self.face_color = bbox_fc
        self.edge_color = bbox_ec
        self.style = my_style
        self.font_dict = font_dict
        self.va = va
        self.ha = ha
        self.lw = 1.6
        self.xLeft = None
        self.xRight = None
        self.yBottom = None
        self.yTop = None
        self.width = None
        self.height = None
        self.x_center = None
        self.y_center = None
        
    def _my_bbox_style(self, facecolor, edgecolor, linewidth, boxstyle):
        '''
        method used to create the dictionary for the BoxStyle of text bbox
        
        Required Parameters:
            facecolor -> str:
                Determines the facecolor of your LogicBox
            edgecolor -> str:
                Determines the edgecolor of your LogicBox
            linewidth -> float/int:
                Determines the width of the LogicBox edges
            boxstyle -> BoxStyle object:
                matplotlib.patches.BoxStyle object for box shape and padding
                https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.BoxStyle.html
        '''
        my_style = {
            'boxstyle': boxstyle,
            'facecolor': facecolor,
            'edgecolor': edgecolor,
            'linewidth': linewidth
        }
        return my_style
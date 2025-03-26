'''
E. Tyler Carr
April 27, 2023
Last Updated: May 5, 2023

This code is used to take in filtered data for the WebApp and produce
a rudimentary mock up for an occurence decision tree that show how many sample are 
filtered out at certain criteria (missing samples, replicate threshold,
CV threshold, and MDL)
'''
import matplotlib.font_manager as fm
from matplotlib.patches import BoxStyle
import pandas as pd

from LogicTreeETC import LogicTree

def make_tree():
    # read in csv as pandas dataframe, we just want the sheet with the summation data
    df = pd.read_csv('.\\logic_tree_data.csv')

    # set variables for filling the logic tree
    # first row of boxes for filling
    n_total_sample_occurence = df['n_total_sample_occurence'].iloc[0]
    str_total_sample_occurence = f'Total Sample Occurence (N = {n_total_sample_occurence:,})'
    n_missing_occurence = df['n_missing_occurence'].iloc[0]
    str_missing_occurence = f'Missing (N = {n_missing_occurence:,})'

    # second row of boxes for filling
    n_over_replicate = df['n_over_replicate'].iloc[0]
    str_over_replicate = f'$\\geq$ Replicate Threshold (N = {n_over_replicate:,})'
    n_under_replicate = df['n_under_replicate'].iloc[0]
    str_under_replicate = f'$<$ Replicate Threshold (N = {n_under_replicate:,})'

    # third row of boxes for filling
    n_under_CV = df['n_under_CV'].iloc[0]
    str_under_CV = f'$\\leq$ CV Threshold (N = {n_under_CV:,})'
    n_over_CV = df['n_over_CV'].iloc[0]
    str_over_CV = f'$>$ CV Threshold (N = {n_over_CV:,})'

    # fourth row of boxes for filling
    n_under_CV_over_MDL = df['n_under_CV_over_MDL'].iloc[0]
    str_under_CV_over_MDL = f'$\\geq$ MDL (N = {n_under_CV_over_MDL:,})'
    n_under_CV_under_MDL = df['n_under_CV_under_MDL'].iloc[0]
    str_under_CV_under_MDL = f'$<$ MDL (N = {n_under_CV_under_MDL:,})'
    n_over_CV_over_MDL = df['n_over_CV_over_MDL'].iloc[0]
    str_over_CV_over_MDL = f'$\\geq$ MDL (N = {n_over_CV_over_MDL:,})'
    n_over_CV_under_MDL = df['n_over_CV_under_MDL'].iloc[0]
    str_over_CV_under_MDL = f'$<$ MDL (N = {n_over_CV_under_MDL:,})'

    # set user defined variables
    replicate_threshold = df['replicate_threshold'].iloc[0]
    replicate_threshold_str = f'\\textbf{{Replicate Threshold = {replicate_threshold}}}'
    CV_threshold = df['CV_threshold'].iloc[0]
    CV_threshold_str = f'\\textbf{{CV Threshold = {CV_threshold}}}'
    MDL = r'$\bigsymbol{\mu}_{\text{MB}} \text{ + } \bigsymbol{3\sigma}_{\text{MB}}$'
    MDL_str = f'\\textbf{{MDL = {MDL}}}'
            
    # set variables for row position and colors
    y_row1 = 110
    y_row2 = 60
    y_row3 = 10
    y_row4 = -30
    general_fc = 'black'
    general_ec = 'white'
    missing_fc = 'dimgrey'
    missing_ec = 'xkcd:light blue grey'
    under_CV_over_MDL_ec = 'xkcd:ocean'
    over_replicate_ec = 'xkcd:bright sky blue'
    over_CV_fc = 'xkcd:cherry'
    over_CV_ec = 'xkcd:rosa'
    under_CV_ec = 'xkcd:water blue'
    over_CV_over_MDL_ec = 'xkcd:light salmon'
    over_CV_over_MDL_fc = 'xkcd:rust orange'
    arr_width = 3.8
    tip_offset = 0.9

    xlims = (-50, 135)
    ylims = (-50, 135)
    logic_tree = LogicTree(xlims=xlims, ylims=ylims, title='Logic Tree - Sample Occurence')

    # total sample occurence box
    logic_tree.add_box(xpos=75, ypos=y_row1, text=str_total_sample_occurence, \
                       box_name="Total Sample Occurence", bbox_fc=general_fc, bbox_ec=general_ec)
    # missing box
    logic_tree.add_box(xpos=99, ypos=y_row1, text=str_missing_occurence, ha='left', \
                       box_name="Missing", bbox_fc=missing_fc, bbox_ec=missing_ec)
    # over replicate threshold box
    logic_tree.add_box(xpos=55, ypos=y_row2, text=str_over_replicate, ha='right', \
                       box_name="Over Replicate", bbox_fc=general_fc, \
                       bbox_ec=over_replicate_ec)
    # under replicate threshold box
    logic_tree.add_box(xpos=65, ypos=y_row2, text=str_under_replicate, ha='left', \
                       box_name="Under Replicate", bbox_fc=missing_fc, bbox_ec=missing_ec)
    # under CV threshold box
    logic_tree.add_box(xpos=20, ypos=y_row3, text=str_under_CV, ha='right', \
                       box_name="Under CV", bbox_fc=general_fc, bbox_ec=under_CV_ec)
    # over CV threshold box
    logic_tree.add_box(xpos=71, ypos=y_row3, text=str_over_CV, ha='left', \
                       box_name="Over CV", bbox_fc=over_CV_fc, bbox_ec=over_CV_ec)
    # under CV, over MDL threshold box
    logic_tree.add_box(xpos=-15, ypos=y_row4, text=str_under_CV_over_MDL, ha='right', \
                       box_name="Under CV, Over MDL", bbox_fc=general_fc, \
                       bbox_ec=under_CV_over_MDL_ec)
    # under CV, under MDL threshold box
    logic_tree.add_box(xpos=-8, ypos=y_row4, text=str_under_CV_under_MDL, ha='left', \
                       box_name="Under CV, Under MDL", bbox_fc=missing_fc, bbox_ec=missing_ec)
    # over CV, over MDL threshold box
    logic_tree.add_box(xpos=98, ypos=y_row4, text=str_over_CV_over_MDL, ha='right', \
                       box_name="Over CV, Over MDL", bbox_fc=over_CV_over_MDL_fc, \
                       bbox_ec=over_CV_over_MDL_ec)
    # over CV, under MDL threshold box
    logic_tree.add_box(xpos=105, ypos=y_row4, text=str_over_CV_under_MDL, ha='left', \
                       box_name="Over CV, Under MDL", bbox_fc=missing_fc, \
                       bbox_ec=missing_ec)

    # add arrow between Total Sample Occurence box and Missing box
    logic_tree.add_connection(boxA=logic_tree.boxes['Total Sample Occurence'], \
                                      boxB=logic_tree.boxes['Missing'], \
                                      arrow_head=True, arrow_width=arr_width, fill_connection=True, \
                                      tip_offset=0.8, lw=1.2)

    # add bifurcation arrows between Total Sample Occurence, Over Replicate, and Under Replicate
    logic_tree.add_connection_biSplit(boxA=logic_tree.boxes['Total Sample Occurence'],
                                 boxB=logic_tree.boxes['Over Replicate'],
                                 boxC=logic_tree.boxes['Under Replicate'], \
                                 arrow_head=True, arrow_width=arr_width, fill_connection=True, \
                                 fc_A='ec', ec_B='xkcd:off white', fc_B='ec', lw=1.3, tip_offset=tip_offset)

    # add bifurcation arrows between Over Replicate, under CV and over CV
    logic_tree.add_connection_biSplit(boxA=logic_tree.boxes['Over Replicate'],
                                 boxB=logic_tree.boxes['Under CV'],
                                 boxC=logic_tree.boxes['Over CV'], \
                                 arrow_head=True, arrow_width=arr_width, fill_connection=True, \
                                 fc_A='ec', ec_B='xkcd:off white', fc_B='ec', lw=1.3, tip_offset=tip_offset)
    # add bifurcation arrows between under CV, over MDL and under MDL
    logic_tree.add_connection_biSplit(boxA=logic_tree.boxes['Under CV'],
                                 boxB=logic_tree.boxes['Under CV, Over MDL'],
                                 boxC=logic_tree.boxes['Under CV, Under MDL'], \
                                 arrow_head=True, arrow_width=arr_width, fill_connection=True, \
                                 fc_A='ec', ec_B='xkcd:off white', fc_B='ec', lw=1.3, tip_offset=tip_offset)
    # add bifurcation arrows between over CV, over MDL and under MDL
    logic_tree.add_connection_biSplit(boxA=logic_tree.boxes['Over CV'],
                                 boxB=logic_tree.boxes['Over CV, Over MDL'],
                                 boxC=logic_tree.boxes['Over CV, Under MDL'], \
                                 arrow_head=True, arrow_width=arr_width, fill_connection=True, lw=1.3, tip_offset=tip_offset)


    # add annotations for threshold parameters
    font_dict_annotation = {
                    'fontsize': 16,
                    'color': 'white'
                }
    y_row1_5 = (y_row1+y_row2)/2
    # add a dummy text to put on the right side to break crouding on the edge
    logic_tree.add_box(xpos=135, ypos=y_row1_5, text='            ', \
                       box_name="Dummy - Annotation", bbox_fc=(1,1,1,0), \
                       bbox_ec=(1,1,1,0), ha='left', va='center', \
                       bbox_style=BoxStyle('Square', pad=0.3), font_dict=font_dict_annotation, lw=1)
    # replicate threshold
    logic_tree.add_box(xpos=0, ypos=y_row1_5, text=replicate_threshold_str, \
                       box_name="Replicate Threshold - Annotation", bbox_fc=(1,1,1,0), \
                       bbox_ec=(1,1,1,0), ha='right', va='center', \
                       bbox_style=BoxStyle('Square', pad=0.3), font_dict=font_dict_annotation, lw=1, \
                       use_tex_rendering=True, ul=True)

    # CV Threshold
    y_row2_5 = (y_row2 + y_row3)/2
    logic_tree.add_box(xpos=-25, ypos=y_row2_5, text=CV_threshold_str, \
                       box_name="CV Threshold - Annotation", bbox_fc=(1,1,1,0), \
                       bbox_ec=(1,1,1,0), ha='right', va='center', \
                       bbox_style=BoxStyle('Square', pad=0.3), font_dict=font_dict_annotation, lw=1, \
                       use_tex_rendering=True, ul=True)

    # MDL
    y_row3_5 = (y_row4 + y_row3)/2
    logic_tree.add_box(xpos=-44, ypos=y_row3_5, text=MDL_str, \
                       box_name="MDL - Annotation", bbox_fc=(1,1,1,0), \
                       bbox_ec=(1,1,1,0), ha='right', va='center', \
                       bbox_style=BoxStyle('Square', pad=0.3), font_dict=font_dict_annotation, lw=1, \
                       use_tex_rendering=True, ul=True, ul_depth_width=('8pt', '1pt'))

    # add flag texts
    logic_tree.add_box(xpos=27, ypos=y_row1_5+arr_width*0.85, text=r'\textit{\textbf{Kept}}', \
                       box_name="Kept - Annotation", bbox_fc=(1,1,1,0), \
                       bbox_ec=(1,1,1,0), ha='right', va='bottom', \
                       bbox_style=BoxStyle('Square', pad=0.1), font_dict=font_dict_annotation, \
                       use_tex_rendering=True, fs=12)
    logic_tree.add_box(xpos=65, ypos=y_row1_5+arr_width*0.85, text=r'\textit{\textbf{Removed}}', \
                       box_name="Removed - Annotation", bbox_fc=(1,1,1,0), \
                       bbox_ec=(1,1,1,0), ha='center', va='bottom', \
                       bbox_style=BoxStyle('Square', pad=0.1), font_dict=font_dict_annotation, \
                       use_tex_rendering=True, fs=12)
    logic_tree.add_box(xpos=56, ypos=y_row2_5+arr_width*0.85, text=r'\textit{\textbf{CV Flag}}', \
                       box_name="Removed - Annotation", bbox_fc=(1,1,1,0), \
                       bbox_ec=(1,1,1,0), ha='center', va='bottom', \
                       bbox_style=BoxStyle('Square', pad=0.1), font_dict=font_dict_annotation, \
                       use_tex_rendering=True, fs=12)
    logic_tree.add_box(xpos=-9, ypos=y_row3_5+arr_width*0.85, text=r'\textit{\textbf{MDL Flag}}', \
                       box_name="MDL01 - Annotation", bbox_fc=(1,1,1,0), \
                       bbox_ec=(1,1,1,0), ha='left', va='bottom', \
                       bbox_style=BoxStyle('Square', pad=0.1), font_dict=font_dict_annotation, \
                       use_tex_rendering=True, fs=12)
    logic_tree.add_box(xpos=105, ypos=y_row3_5+arr_width*0.85, text=r'\textit{\textbf{MDL Flag}}', \
                       box_name="MDL02 - Annotation", bbox_fc=(1,1,1,0), \
                       bbox_ec=(1,1,1,0), ha='left', va='bottom', \
                       bbox_style=BoxStyle('Square', pad=0.1), font_dict=font_dict_annotation, \
                       use_tex_rendering=True, fs=12)

    # add title
    logic_tree.make_title(pos='left')
    logic_tree.save_as_png(file_name='.\\logic_tree-sample_occurence03.png', dpi=900)

def check_for_font(font):
    font = font.lower()
    
    # List all available fonts
    for f in fm.findSystemFonts(fontpaths=None, fontext='ttf'):
        if font in f.lower():
            print(f)
            return
    print(f"{font} not found")

if __name__ == '__main__':
    # check_for_font("Leelawadee")
    make_tree()
    



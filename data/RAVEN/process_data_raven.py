
import os
import argparse
import json
import string
import ast
import tqdm
import numpy as np
import xml.etree.ElementTree as ET

# parse arguments
parser = argparse.ArgumentParser(description='Convert RAVEN data split to text and symbolic jsonl inputs compatible with models. Folders are supported. If folder input, all files in folder are processed.')
parser.add_argument('input_path')
parser.add_argument('output_path')
parser.add_argument('--input_path_aux', action='store', help='If input is a xml file, auxiliary npz input file.')
parser.add_argument('--type', action='store', default='text', help='Type of output: text, symbolic or pixel.')

args = parser.parse_args()
input_file_path = args.input_path
output_file_path = args.output_path
input_file_path_aux = args.input_path_aux

output_type = args.type
if output_type not in ['text', 'symbolic', 'count', 'pixel']:
    raise ValueError(f"Output type {output_type} not supported.")

is_folder = os.path.isdir(input_file_path)

if is_folder:
    input_list = [input_file_path + f for f in os.listdir(input_file_path) if os.path.isfile(input_file_path + f)]
elif input_file_path_aux is not None:
    input_list = [input_file_path, input_file_path_aux]
else:
    raise ValueError(f"Input path {input_file_path} must be a folder or an auxiliary input file must be provided.")
    
if os.path.isdir(output_file_path):
    raise ValueError(f"Output path {output_file_path} must not be a folder.")

print(f"Writing {('all elements of ' + input_file_path) if is_folder else ('from ' + input_file_path + ' with auxiliary ' + input_file_path_aux)} to {output_file_path} in {output_type} format.")



# Constants
PROBLEM_MATRIX = 8
ANSWER_SET = 8



# Helper functions
def get_xml_npz_files(input_file_list):
    nb_matches = 0
    res = {}
    for input_file in input_file_list:
        if input_file.endswith('.xml'):
            name = input_file[:-4]
            if name in res:
                res[name]['xml'] = True
                nb_matches += 1
            else:
                res[name] = {'xml': True, 'npz': False}
        elif input_file.endswith('.npz'):
            name = input_file[:-4]
            if name in res:
                res[name]['npz'] = True
                nb_matches += 1
            else:
                res[name] = {'xml': False, 'npz': True}
        else:
            raise ValueError(f"Input file {input_file} has unsupported file extension.")
    
    if nb_matches != len(res):
        raise ValueError(f"Input files do not match. {nb_matches} matches found, but {len(res)} files.")
    
    return list(res.keys())


def file_to_tree(file_path):
    tree = ET.parse(file_path)
    return tree.getroot()

def print_tree(root, level=0):
    print('  ' * level + root.tag + ' ' + str(root.attrib))
    for child in root:
        print_tree(child, level + 1)

def get_in_tree(root, label, level=0):
    values = []
    levels = []
    if root.tag == label:
        return [root], level
    for child in root:
        child_values, child_level = get_in_tree(child, label, level + 1)
        values += child_values
        levels += [child_level] if len(child_values) > 0 else []
    level = max(levels + [level])
    return values, level

def format_sample(problem_panels, answer_panels):
    sample = {
        "input": [],
        "ideal": ""
    }
    sample["input"].append({
        "role": "system",
        "content": f"Find the pattern number {PROBLEM_MATRIX+1} that completes the sequence. Pick the letter in front of the correct pattern that logically follows in the sequence from the answer set. "\
                    + f"Patterns in the sequence are preceded by a number from 1 to {PROBLEM_MATRIX}. "\
                    + f"Patterns in the answer set are preceded by a letter from A to {string.ascii_uppercase[ANSWER_SET-1]}. Only return the letter in front of the correct pattern."
    })
    
    for i, panel in enumerate(problem_panels):
        sample["input"].append({
            "role": "system",
            "content": f"{i+1}. {panel}"
        })
    
    for i, panel in enumerate(answer_panels):
        sample["input"].append({
            "role": "system",
            "content": f"{string.ascii_uppercase[i]}. {panel}"
        })
    
    return sample

def extract_ideal(file_path):
    with np.load(file_path) as data:
        target = data['target']
    return string.ascii_uppercase[target]

def extract_raw(file_path):
    with np.load(file_path) as data:
        image = data['image']

        # inverse colors
        image = 255 - image

        # pool image with 10x10 kernel max pooling
        batch, M, N = image.shape
        K = 10
        L = 10
        MK = M // K
        NL = N // L
        image = image.reshape(-1, MK, K, NL, L).max(axis=(2, 4))

    return image


# Tree parser class
class PanelParser():

    def __init__(self, 
                 panel_sentence : str,
                 struct_sentences : dict,
                 component_sentences : dict,
                 layout_mapping : dict,
                 layout_start_sentence : str,
                 layout_mid_sentence : str,
                 layout_end_sentence : str,
                 entity_sentence : str,
                 entity_attributes : dict):
        self.panel_sentence = panel_sentence
        self.struct_sentences = struct_sentences
        self.component_sentences = component_sentences
        self.layout_mapping = layout_mapping
        self.layout_start_sentence = layout_start_sentence
        self.layout_mid_sentence = layout_mid_sentence
        self.layout_end_sentence = layout_end_sentence
        self.entity_sentence = entity_sentence
        self.entity_attributes = entity_attributes


    def __call__(self, panel):
        desc = [self.gen_struct_description(struct) for struct in panel]
        if len(desc) == 1:
            return self.panel_sentence.format(desc[0])
        else:
            raise ValueError(f"Panel should contain a single struct, but has {len(desc)}")
        
    
    def gen_struct_description(self, struct):
        try:
            description = self.struct_sentences[struct.attrib['name']]
        except KeyError:
            raise ValueError(f"Unknown struct name {struct.attrib['name']}")
        
        return description.format("".join([self.gen_component_description(component) for component in struct]))
    
    def gen_component_description(self, component):
        try:
            description = self.component_sentences[component.attrib['name']]
        except KeyError:
            raise ValueError(f"Unknown component name {component.attrib['name']}")
        
        return description.format("".join([self.gen_layout_description(layout) for layout in component]))

    def gen_layout_description(self, layout):
        try:
            pos = self.layout_mapping[layout.attrib['name']]
            coords = [str(c) for c in ast.literal_eval(layout.attrib['Position'])]
            pos_voc = dict(zip(coords, pos))
        except KeyError:
            pos_voc = None
        
        description = self.layout_start_sentence
        for i, entity in enumerate(layout):
            entity_desc = self.gen_entity_description(entity, pos_voc)
            description += (self.layout_mid_sentence.format(entity_desc)) if i < len(layout) - 1 else (self.layout_end_sentence.format(entity_desc))

        return description

    def gen_entity_description(self, entity, pos_voc : list = None):
        attr_list = []
        for attr in self.entity_attributes:
            if attr not in entity.attrib:
                raise ValueError(f"Entity is missing attribute {attr}")
            if attr == 'bbox':
                pos = ""
                if pos_voc is not None:
                    pos = pos_voc[entity.attrib['bbox']]
                attr_list.append(pos)
            else:
                attr_list.append(self.entity_attributes[attr][int(entity.attrib[attr])])

        return self.entity_sentence.format(*attr_list)
    


# Generate parsers
if output_type == 'text':
    panel_parser = PanelParser(
                    panel_sentence = "On an image, {}",
                    struct_sentences = {
                        'Singleton': "{}",
                        'Left_Right': "a first figure is displayed on the left and a second is displayed on the right. {}",
                        'Up_Down': "a first figure is displayed above and a second is displayed below. {}",
                        'Out_In': "a second figure is displayed inside a first one. {}"
                    },
                    component_sentences = {
                        'Grid': "{}",
                        'Left': "On the left: {}",
                        'Right': "On the right: {}",
                        'Up': "Above: {}",
                        'Down': "Below: {}",
                        'In': "Inside: {}",
                        'Out': "Outside: {}"
                    },
                    layout_mapping = {
                        'Distribute_Four': [' in the top left', ' in the top right', ' in the bottom left', ' in the bottom right'],
                        'In_Distribute_Four': [' in the top left', ' in the top right', ' in the bottom left', ' in the bottom right'],
                        'Distribute_Nine': [' in the top left', ' in the top center',  ' in the top right', ' in the center left', ' in the center', ' in the center right', ' in the bottom left', ' in the bottom center', ' in the bottom right'],
                    },
                    layout_start_sentence = "",
                    layout_mid_sentence = "{}, ",
                    layout_end_sentence = "{}. ",
                    entity_sentence = "a {} {} {} rotated at {} degrees{}", # the order of appearance in the sentence corresponds to the order of appearance in the entity_attributes dictionary
                    entity_attributes = {'Size': ['tiny', 'small', 'medium', 'large', 'huge', 'giant'],
                                        'Color': ['red', 'orange', 'yellow', 'lime', 'green', 'cyan', 'blue', 'purple', 'pink', 'white'], # true colors are shades of gray: [255, 224, 196, 168, 140, 112, 84, 56, 28, 0]
                                        'Type': ['none', 'triangle', 'square', 'pentagon', 'hexagon', 'circle'],
                                        'Angle' : ['-135', '-90', '-45', '0', '45', '90', '135', '180'],  # constant values taken from https://github.com/WellyZhang/RAVEN/blob/master/src/dataset/const.py
                                        'bbox' : None, # special attribute determined by the layout mapping
                                        }
                )
elif output_type == 'symbolic':
    panel_parser = PanelParser(
                    panel_sentence = "{}",
                    struct_sentences = {
                        'Singleton': "{}",
                        'Left_Right': "{}",
                        'Up_Down': "{}",
                        'Out_In': "{}"
                    },
                    component_sentences = {
                        'Grid': "{}",
                        'Left': "{}| ",
                        'Right': "{}",
                        'Up': "{}\n---------------\n",
                        'Down': "{}",
                        'In': "---------------\n{}\n---------------",
                        'Out': "{}\n"
                    },
                    layout_mapping = {
                        'Distribute_Four': [' TL', ' TR', ' BL', ' BR'],
                        'In_Distribute_Four': [' TL', ' TR', ' BL', ' BR'],
                        'Distribute_Nine': [' TL', ' TC',  ' TR', ' CL', ' C', ' CR', ' BL', ' BC', ' BR'],
                    },
                    layout_start_sentence = "[",
                    layout_mid_sentence = "{}, ",
                    layout_end_sentence = "{}] ",
                    entity_sentence = "({}, {}, {}, {},{})", # the order of appearance in the sentence corresponds to the order of appearance in the entity_attributes dictionary
                    entity_attributes = {'Size': ['A', 'B', 'C', 'D', 'E', 'F'],
                                        'Color': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
                                        'Type': ['A', 'B', 'C', 'D', 'E', 'F'],
                                        'Angle' : ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
                                        'bbox' : None, # special attribute determined by the layout mapping
                                        }
                )
elif output_type == 'count':
    panel_parser = PanelParser(
                    panel_sentence = "{}",
                    struct_sentences = {
                        'Singleton': "{}",
                        'Left_Right': "{}",
                        'Up_Down': "{}",
                        'Out_In': "{}"
                    },
                    component_sentences = {
                        'Grid': "{}",
                        'Left': "{}| ",
                        'Right': "{}",
                        'Up': "{}\n---------------\n",
                        'Down': "{}",
                        'In': "---------------\n{}\n---------------",
                        'Out': "{}\n"
                    },
                    layout_mapping = {
                        'Distribute_Four': [' TL', ' TR', ' BL', ' BR'],
                        'In_Distribute_Four': [' TL', ' TR', ' BL', ' BR'],
                        'Distribute_Nine': [' TL', ' TC',  ' TR', ' CL', ' C', ' CR', ' BL', ' BC', ' BR'],
                    },
                    layout_start_sentence = "[",
                    layout_mid_sentence = "{}, ",
                    layout_end_sentence = "{}] ",
                    entity_sentence = "({}, {}, {}, {},{})", # the order of appearance in the sentence corresponds to the order of appearance in the entity_attributes dictionary
                    entity_attributes = {'Size': ['A', 'A,A', 'A,A,A', 'A,A,A,A', 'A,A,A,A,A', 'A,A,A,A,A,A'],
                                        'Color': ['B', 'B,B', 'B,B,B', 'B,B,B,B', 'B,B,B,B,B', 'B,B,B,B,B,B', 'B,B,B,B,B,B,B', 'B,B,B,B,B,B,B,B', 'B,B,B,B,B,B,B,B,B', 'B,B,B,B,B,B,B,B,B,B'],
                                        'Type': ['C', 'C,C', 'C,C,C', 'C,C,C,C', 'C,C,C,C,C', 'C,C,C,C,C,C'],
                                        'Angle' : ['D', 'D,D', 'D,D,D', 'D,D,D,D', 'D,D,D,D,D', 'D,D,D,D,D,D', 'D,D,D,D,D,D,D', 'D,D,D,D,D,D,D,D'],
                                        'bbox' : None, # special attribute determined by the layout mapping
                                        }
                )
elif output_type == 'pixel':
    panel_parser = lambda x : str(x.tolist())



# Parse files
samples = []
for f_name in tqdm.tqdm(get_xml_npz_files(input_list)):
    # Process xml file*
    if output_type != 'pixel':
        root = file_to_tree(f_name + ".xml")
        raw_panels = get_in_tree(root,'Panel')[0]

    # Process npz file
    ideal = extract_ideal(f_name + ".npz")
    if output_type == 'pixel':
        raw_panels = extract_raw(f_name + ".npz")

    # Generate samples
    panels = [[], []]
    for i, panel in enumerate(raw_panels):
        panel_str = panel_parser(panel)
        panels[i//PROBLEM_MATRIX].append(panel_str)

    sample = format_sample(*panels)
    sample["ideal"] = ideal
    samples.append(sample)



# Write to output file
with open(output_file_path, "w") as f:
    for sample in samples:
        f.write(json.dumps(sample) + "\n")
print("Saved dataset to: ", output_file_path)
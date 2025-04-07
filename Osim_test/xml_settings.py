# -*- coding: utf-8 -*-
"""

@author: Maksim

@mailto:
    
Created on Thu Jan 23 13:05:53 2025

version history:
    
"""

import xml.etree.ElementTree as ET
import dicttoxml
from xml.dom.minidom import parseString

# Function to convert XML tree to dictionary
def xml_to_dict(element):
    def _xml_to_dict(element):
        node = {}
        # Add element attributes
        if element.attrib:
            node['@attributes'] = element.attrib
        # Add element text
        if element.text and element.text.strip():
            node['#text'] = element.text.strip()
        # Add child elements
        for child in element:
            child_dict = _xml_to_dict(child)
            if child.tag not in node:
                node[child.tag] = child_dict
            else:
                if not isinstance(node[child.tag], list):
                    node[child.tag] = [node[child.tag]]
                node[child.tag].append(child_dict)
        return node
    
    return {element.tag: _xml_to_dict(element)}

# Function to read XML file and convert to dictionary
def read_xml_to_dict(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    return xml_to_dict(root)


# Function to convert dictionary to XML string
def dict_to_xml_str(data):
    xml_str = dicttoxml.dicttoxml(data, custom_root='root', attr_type=False)
    dom = parseString(xml_str)
    return dom.toprettyxml()

# Function to write XML string to file
def write_dict_to_xml(file_path, data):
    xml_str = dict_to_xml_str(data)
    with open(file_path, 'w') as file:
        file.write(xml_str)
        
        
# Example usage
xml_file_path = 'HSS_scale_settings.xml'
dict_file_path = 'output.xml'

# Read XML file and convert to dictionary
xml_dict = read_xml_to_dict(xml_file_path)
print("XML to Dictionary:")
print(xml_dict)

# Convert dictionary back to XML and write to file
write_dict_to_xml(dict_file_path, xml_dict)
print(f"Dictionary written back to XML file: {dict_file_path}")
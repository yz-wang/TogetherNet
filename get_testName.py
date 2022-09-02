import os
import xml.etree.ElementTree as ET

VOCdevkit_path  = 'test'
xmlfilepath     = os.path.join(VOCdevkit_path, 'Annotations')
saveBasePath    = VOCdevkit_path
temp_xml        = os.listdir(xmlfilepath)

total_xml       = []
for xml in temp_xml:
    if xml.endswith(".xml"):
        total_xml.append(xml)

num     = len(total_xml)
list    = range(num)
ftest       = open(os.path.join(saveBasePath,'test.txt'), 'w')

for i in list:
    name = total_xml[i][:-4] + '\n'
    ftest.write(name)

print("done")
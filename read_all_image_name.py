# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 19:36:00 2018

@author: yzzhao2
"""

import os

# read a folder, return the complete path
def get_files(path):
    ret = []
    for root, dirs, files in os.walk(path):  
        for filespath in files:
            fullpath = os.path.join(root,filespath)
            k = fullpath.split('\\')[-4:]
            j = k[0] + '/' + k[1] + '/' + k[2] + '/' + k[3]
            if fullpath.split('\\')[-4] == 'KAIST_spectral':
                k = fullpath.split('\\')[-3:]
                j = k[0] + '/' + k[1] + '/' + k[2]
            ret.append(j)
    return ret

# read a folder, return the image name
def get_jpgs(path):
    ret = [] 
    for root, dirs, files in os.walk(path):  
        for filespath in files: 
            ret.append(filespath)
    return ret

# save a list to a txt
def text_save(content, filename, mode = 'a'):
    # Try to save a list variable in txt file.
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i]) + '\n')
    file.close()

# read a txt expect EOF
def text_readlines(filename):
    # Try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        file = open(filename, 'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    # This for loop deletes the EOF (like \n)
    for i in range(len(content)):
        content[i] = content[i][:len(content[i])-1]
    file.close()
    return content

if __name__ == '__main__':

    # get file names
    fullname = get_files("C:\\Users\\ZHAO Yuzhi\\Desktop\\dataset\\KAIST_spectral")
    jpgname = get_jpgs("C:\\Users\\ZHAO Yuzhi\\Desktop\\dataset\\KAIST_spectral")

    # save the files
    text_save(fullname, "./test.txt")
    print("fullname saved")
    #text_save(jpgname, "./ILSVRC2012_train_sal_name.txt")
    print("jpgname saved")
    print("successfully saved")

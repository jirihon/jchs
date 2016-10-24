#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import cv2
import numpy as np
import sys
import os
import xml.etree.ElementTree as etree
import re
import glob

songs = [
    {'dir': '01_Predehra', 'slug': '01', 'title': 'Předehra'},
    {'dir': '02_Jak_ze_sna_procitam', 'slug': '02', 'title': 'Jak ze sna procitám'},
    {'dir': '03a_Proc_ten_shon', 'slug': '03a', 'title': 'Proč ten shon'},
    {'dir': '03b_Divna_mystifikace', 'slug': '03b', 'title': 'Divná mystifikace'},
    {'dir': '04_Vse_je_tak_jak_ma_byt', 'slug': '04', 'title': 'Vše je tak, jak má být'},
    {'dir': '05_Zemrit_by_mel', 'slug': '05', 'title': 'Zemřít by měl'},
    {'dir': '06_Hosanna', 'slug': '06', 'title': 'Hosanna'},
    {'dir': '07_Simon_Zelotes_Ubohy_Jeruzalem', 'slug': '07', 'title': 'Šimon Zélotes - Ubohý Jeruzalém'},
    {'dir': '08_Pilatuv_sen', 'slug': '08', 'title': 'Pilátův sen'},
    {'dir': '09_V_chramu', 'slug': '09', 'title': 'V chrámu'},
    {'dir': '10_Vse_je_tak_jak_ma_byt_II_Co_na_tom_je_tak_zleho', 'slug': '10', 'title': 'Vše je tak, jak má být II - Co na tom je tak zlého'},
    {'dir': '11_Zavrzen_na_veky_vekuv_Penize_zkropene_krvi', 'slug': '11', 'title': 'Zavržen na věky věkův - Peníze zkropené krví'}
]
instruments = [
    {'slug': 'Tbn', 'regex': re.compile('(T|\'I\')(rombone|[bh][nu])')},
    {'slug': 'Flt', 'regex': re.compile('(Flute|F[l1LI\]\[\}\{])')},
    {'slug': 'Clr', 'regex': re.compile('.*(Clarinet|C[l1LI\]\[\}\{])')},
    {'slug': 'Tpt', 'regex': re.compile('.*(Trumpet|Tp[tlL]|[B8].* m.)')}
]
score_x_shift = -30
label_x_offset = -150
first_label_extra_x_offset = -100
label_first_width = 270
label_other_width = 170
page_height_limit = 2590
score_width = 4250
top_y_offset = -120
bottom_y_offset = +100
white_color = (255, 255, 255)

tex_header = '''
\\documentclass[]{article}
\\usepackage[a4paper,landscape,top=2cm,bottom=1cm,left=0.9cm,right=0.8cm]{geometry}
\\usepackage[utf8]{inputenc}
\\usepackage{graphicx}
\\usepackage{layout}
\\usepackage{morefloats}
\\usepackage{fancyhdr}
\\pagestyle{fancy}
\\chead{%s}
\\renewcommand{\\headrulewidth}{0pt}
\\setlength{\\headsep}{0.5cm}
\\setlength{\\footskip}{0.5cm}

\\begin{document}
'''
tex_footer = '''
\\vfill
\\end{document}
'''
tex_image = '''
\\begin{figure}
\\centering
\\includegraphics[width=1\\linewidth]{%s}
\\end{figure}
'''


def preprocess(files):
    for path in files:
        filename, file_extension = os.path.splitext(path)
        print 'file:', filename, file_extension

        img = cv2.imread(path)
        height, width, channels = img.shape

        img_t = np.zeros((width, height, channels), img.dtype)
        cv2.transpose(img, img_t)
        cv2.flip(img_t, 1, img_t)

        img_gray = cv2.cvtColor(img_t, cv2.COLOR_BGR2GRAY)

        min_sum = sys.maxint
        opt_angle = 0
        angle = -1
        while angle < 1.1:
            rot_m = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
            img_gray_r = cv2.warpAffine(
                img_gray, rot_m, (height, width),
                borderMode=cv2.BORDER_CONSTANT, borderValue=white_color)
            sum = img_gray_r.sum(axis=1).min()
            if sum < min_sum:
                min_sum = sum
                opt_angle = angle
            angle += 0.1
        print 'opt_angle:', opt_angle

        rot_m = cv2.getRotationMatrix2D((width / 2, height / 2), opt_angle, 1)
        img_r = cv2.warpAffine(
            img_t, rot_m, (height, width),
            borderMode=cv2.BORDER_CONSTANT, borderValue=white_color)

        img_gray = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        # score_x = img_gray[0:width, 0:600].sum(axis=0).argmin() + score_x_shift
        x_sums = img_gray.sum(axis=0)
        sum_threshold = x_sums.mean() - 3 * x_sums.std()
        score_x = score_x_shift
        for x_sum in x_sums:
            if x_sum < sum_threshold:
                break
            score_x += 1

        print 'score_x:', score_x

        if filename.split('/')[-1] == 'page-000':
            label_width = label_first_width
        else:
            label_width = label_other_width
        label_left_x = max(0, score_x - label_width)

        cv2.imwrite('%s.png' % filename, img_r)
        img_instr = cv2.copyMakeBorder(
            img_r[0:width, label_left_x:score_x], 0, 0, label_left_x, 300,
            borderType=cv2.BORDER_CONSTANT, value=white_color)
        cv2.imwrite('%s-instr.png' % filename, img_instr)


def ocr(files):
    for path in files:
        filename, file_extension = os.path.splitext(path)
        print 'file:', filename, file_extension
        os.system('tesseract %s.png %s -l eng hocr' % (filename, filename))


def extract(files, instr, regex):
    staff_per_page_sum = 0
    staff_per_page_cnt = 0
    staff_height_sum = 0
    page_height_sum = 0
    page_cnt = 0
    staff_cnt = 0

    for path in files:
        filename, file_extension = os.path.splitext(path)
        print 'file:', filename, file_extension

        tree = etree.parse('%s-instr.hocr' % filename)
        root = tree.getroot()
        ns = {'x': 'http://www.w3.org/1999/xhtml'}

        left_x = 0
        right_x = 0
        top_y = 0
        bottom_y = 0
        label_x_sum = 0
        label_x_cnt = 0

        for par in root.findall("./x:body/x:div/x:div/x:p/x:span[@class='ocr_line']", ns):
            text = "".join(par.itertext()).strip()
            coords = par.attrib['title'].split(';')[0].split(' ')
            if re.match(regex, text):
                print text, coords
                if top_y == 0:
                    top_y = int(coords[2])
                bottom_y = int(coords[4])
                label_x_sum += int(coords[3])
                label_x_cnt += 1

        label_x = int(label_x_sum / label_x_cnt)

        img = cv2.imread('%s.png' % filename)
        height, width, _ = img.shape

        if filename.split('/')[-1] == 'page-000':
            left_x += first_label_extra_x_offset
            right_x += first_label_extra_x_offset

        top_y += top_y_offset
        bottom_y += bottom_y_offset
        left_x += label_x + label_x_offset
        right_x += left_x + score_width

        left_x = max(0, left_x)
        right_x = min(width, right_x)

        cv2.imwrite('%s-%s.png' % (filename, instr), img[top_y:bottom_y, left_x:right_x])

        height = bottom_y - top_y
        staff_height_sum += height
        staff_cnt += 1

        if page_height_sum + height > page_height_limit:
            print page_height_sum, height, page_height_sum + height
            page_height_sum = height
            staff_per_page_sum += staff_per_page_cnt
            staff_per_page_cnt = 1
            page_cnt += 1
        else:
            staff_per_page_cnt += 1
            page_height_sum += height

    staff_height = int(staff_height_sum / staff_cnt)

    if page_cnt > 0:
        staff_per_page = int(staff_per_page_sum / page_cnt)
    else:
        staff_per_page = staff_per_page_cnt

    missing_staff_cnt = staff_per_page - staff_per_page_cnt
    print staff_per_page, staff_per_page_cnt, page_cnt, staff_height

    img_empty = np.zeros((staff_height, score_width, 3), 'uint8')
    img_empty[:] = white_color

    filename_prefix = ''.join(filename.split('-')[:-1])
    os.system('rm %s-9*-%s.png' % (filename_prefix, instr))

    while missing_staff_cnt > 0:
        cv2.imwrite(
            '%s-9%02d-%s.png' % (filename_prefix, missing_staff_cnt, instr),
            img_empty)
        missing_staff_cnt -= 1


def export(files, song_slug, song_title, instr):
    f = open('tex/%s_%s.tex' % (song_slug, instr), 'w')
    f.write(tex_header % song_title)
    for path in files:
        f.write(tex_image % ('../%s' % path))

    f.write(tex_footer)
    f.close()

    outname = '%s_%s' % (song_slug, instr)
    os.system("cd tex; pdflatex %s.tex" % outname)
    os.system('mkdir -p %s' % instr)
    os.system('mv tex/%s.pdf %s/' % (outname, instr))
    os.system('rm tex/%s.log tex/%s.aux' % (outname, outname))


def unpdf():
    for song in songs:
        print 'song:', song['dir']
        os.system('mkdir %s' % song['dir'])
        os.system('pdfimages %s.pdf %s/page' % (song['dir'], song['dir']))


def list_song_files(pattern):
    files = []
    for song in songs:
        files.extend(glob.glob(pattern % song['dir']))
    files.sort()
    return files


action = sys.argv[1]

if action == 'preprocess':
    if len(sys.argv) == 2:
        files = list_song_files('%s/page-*.p*m')
    else:
        files = sys.argv[2:]
    preprocess(files)
elif action == 'ocr':
    files = []
    if len(sys.argv) == 2:
        files = list_song_files('%s/page-*-instr.png')
    else:
        files = sys.argv[2:]
    ocr(files)
elif action == 'extract':
    for song in songs:
        print 'song:', song['dir']
        files = glob.glob('%s/page-*.p*m' % song['dir'])
        files.sort()
        for instr in instruments:
            extract(files, instr['slug'], instr['regex'])
elif action == 'export':
    for song in songs:
        print 'song:', song['dir']
        for instr in instruments:
            files = glob.glob('%s/page-*-%s.png' % (song['dir'], instr['slug']))
            files.sort()
            export(files, song['slug'], song['title'], instr['slug'])
elif action == 'unpdf':
    unpdf()

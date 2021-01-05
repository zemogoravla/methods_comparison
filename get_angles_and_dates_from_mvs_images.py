import os
import glob
import random
import subprocess

import s2p
import rpcm
from utils import *

result_angles_filename = '/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/angles.csv'
result_dates_filename = '/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/dates.csv'

image_dir = 'http://menthe.ovh.hw.ipol.im/IARPA_data/cloud_optimized_geotif/'  # le falta la imagen del 19 de diciembre
image_dir = 'http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/'

image_filenames_dict = {
1:'http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/15NOV14WV031000014NOV15135121-P1BS-500171606160_05_P005_________AAE_0AAAAABAABC0.TIF',
2:'http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/05JAN15WV031000015JAN05135727-P1BS-500497282040_01_P001_________AAE_0AAAAABPABR0.TIF',
3:'http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/11JAN15WV031000015JAN11135414-P1BS-500497283010_01_P001_________AAE_0AAAAABPABS0.TIF',
4:'http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/23JAN15WV031000015JAN23134652-P1BS-500497282020_01_P001_________AAE_0AAAAABPABQ0.TIF',
5:'http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/06FEB15WV031000015FEB06141035-P1BS-500497283080_01_P001_________AAE_0AAAAABPABP0.TIF',
6:'http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/11FEB15WV031000015FEB11135123-P1BS-500497282030_01_P001_________AAE_0AAAAABPABR0.TIF',
7:'http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/12FEB15WV031000015FEB12140652-P1BS-500497283100_01_P001_________AAE_0AAAAABPABQ0.TIF',
8:'http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/08MAR15WV031000015MAR08134953-P1BS-500497284060_01_P001_________AAE_0AAAAABPABQ0.TIF',
9:'http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/15MAR15WV031000015MAR15140133-P1BS-500497284070_01_P001_________AAE_0AAAAABPABQ0.TIF',
10:'http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/21MAR15WV031000015MAR21135704-P1BS-500497282060_01_P001_________AAE_0AAAAABPABQ0.TIF',
11:'http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/22MAR15WV031000015MAR22141208-P1BS-500497285090_01_P001_________AAE_0AAAAABPABQ0.TIF',
12:'http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/02APR15WV031000015APR02134716-P1BS-500276959010_02_P001_________AAE_0AAAAABPABB0.TIF',
13:'http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/02APR15WV031000015APR02134802-P1BS-500276959010_02_P001_________AAE_0AAAAABPABC0.TIF',
14:'http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/03APR15WV031000015APR03140238-P1BS-500497283030_01_P001_________AAE_0AAAAABPABR0.TIF',
15:'http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/22APR15WV031000015APR22140347-P1BS-500497282070_01_P001_________AAE_0AAAAABPABS0.TIF',
16:'http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/04MAY15WV031000015MAY04135349-P1BS-500497283060_01_P001_________AAE_0AAAAABPABP0.TIF',
17:'http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/05MAY15WV031000015MAY05140810-P1BS-500497282090_01_P001_________AAE_0AAAAABPABM0.TIF',
18:'http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/12JUN15WV031000015JUN12140737-P1BS-500497284090_01_P001_________AAE_0AAAAABPABM0.TIF',
19:'http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/18JUN15WV031000015JUN18140207-P1BS-500497285040_01_P001_________AAE_0AAAAABPABR0.TIF',
20:'http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/19JUN15WV031100015JUN19141753-P1BS-500346924040_01_P001_________AAE_0AAAAABPABP0.TIF',
21:'http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/24JUN15WV031000015JUN24135730-P1BS-500497285080_01_P001_________AAE_0AAAAABPABO0.TIF',
22:'http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/30JUN15WV031000015JUN30135323-P1BS-500497282080_01_P001_________AAE_0AAAAABPABP0.TIF',
23:'http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/14JUL15WV031000015JUL14141509-P1BS-500497283020_01_P001_________AAE_0AAAAABPABO0.TIF',
24:'http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/19JUL15WV031000015JUL19135438-P1BS-500497285010_01_P001_________AAE_0AAAAABPABP0.TIF',
25:'http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/27AUG15WV031000015AUG27141600-P1BS-500497284050_01_P001_________AAE_0AAAAABPABP0.TIF',
26:'http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/01SEP15WV031000015SEP01135603-P1BS-500497284040_01_P001_________AAE_0AAAAABPABP0.TIF',
27:'http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/08SEP15WV031000015SEP08140733-P1BS-500497283070_01_P001_________AAE_0AAAAABPABP0.TIF',
28:'http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/14SEP15WV031000015SEP14140305-P1BS-500497285020_01_P001_________AAE_0AAAAABPABS0.TIF',
29:'http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/15SEP15WV031000015SEP15141840-P1BS-500497285060_01_P001_________AAE_0AAAAABPABO0.TIF',
30:'http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/27SEP15WV031000015SEP27140912-P1BS-500497284100_01_P001_________AAE_0AAAAABPABQ0.TIF',
31:'http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/03OCT15WV031000015OCT03140452-P1BS-500497283050_01_P001_________AAE_0AAAAABPABR0.TIF',
32:'http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/22OCT15WV031000015OCT22140432-P1BS-500497282010_01_P001_________AAE_0AAAAABPABS0.TIF',
33:'http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/23OCT15WV031100015OCT23141928-P1BS-500497285030_01_P001_________AAE_0AAAAABPABO0.TIF',
34:'http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/11DEC15WV030900015DEC11135506-P1BS-500510591010_01_P001_________AAE_0AAAAABPABJ0.TIF',
35:'http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/18DEC15WV031000015DEC18140455-P1BS-500515572010_01_P001_________AAE_0AAAAABPABJ0.TIF',
36:'http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/18DEC15WV031000015DEC18140510-P1BS-500515572040_01_P001_________AAE_0AAAAABPABJ0.TIF',
37:'http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/18DEC15WV031000015DEC18140522-P1BS-500515572020_01_P001_________AAE_0AAAAABPABJ0.TIF',
38:'http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/18DEC15WV031000015DEC18140533-P1BS-500515572050_01_P001_________AAE_0AAAAABPABJ0.TIF',
39:'http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/18DEC15WV031000015DEC18140544-P1BS-500515572060_01_P001_________AAE_0AAAAABPABJ0.TIF',
40:'http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/18DEC15WV031000015DEC18140554-P1BS-500515572030_01_P001_________AAE_0AAAAABPABJ0.TIF',
41:'http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/19DEC15WV031000015DEC19142039-P1BS-500514410020_01_P001_________AAE_0AAAAABPABW0.TIF',
42:'http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/25DEC15WV031000015DEC25141655-P1BS-500526006010_01_P001_________AAE_0AAAAABPAAZ0.TIF',
43:'http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/25DEC15WV031000015DEC25141705-P1BS-500526006020_01_P001_________AAE_0AAAAABPAAW0.TIF',
44:'http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/07JAN16WV031000016JAN07142142-P1BS-500537128030_01_P001_________AAE_0AAAAABPABE0.TIF',
45:'http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/07JAN16WV031000016JAN07142152-P1BS-500537128010_01_P001_________AAE_0AAAAABPABA0.TIF',
46:'http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/07JAN16WV031000016JAN07142202-P1BS-500537128020_01_P001_________AAE_0AAAAABPAAY0.TIF',
47:'http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/13JAN16WV031100016JAN13141501-P1BS-500541801010_01_P001_________AAE_0AAAAABPABJ0.TIF',
}

month_dict = {'JAN':1, 'FEB':2, 'MAR':3, 'APR':4, 'MAY':5, 'JUN':6,
              'JUL':7, 'AUG':8, 'SEP':9, 'OCT':10, 'NOV':11, 'DEC':12}

image_filename_key_list = list(image_filenames_dict.keys())
print(image_filename_key_list)


#-----------------------------------------------------------
# DATES
dates_list = []
for i, ref_key in enumerate(image_filename_key_list):
    print('procesando {:03d}/{:03d}   key:{}'.format(i + 1, len(image_filename_key_list), ref_key), flush=True)
    ref_filename = image_filenames_dict[ref_key]
    _, ref_image_name = os.path.split(ref_filename)
    ref_image_name_no_extension, _ = os.path.splitext(ref_image_name)

    image_name = ref_image_name_no_extension
    year = int(image_name[16:18]) + 2000
    month = month_dict[image_name[18:21]]
    day = int(image_name[21:23])
    hour = int(image_name[23:25])
    minute = int(image_name[25:27])
    second = int(image_name[27:29])
    #print(year, month, day, hour, minute, second )

    dates_list. append([ref_key, image_name, year, month, day, hour, minute, second])

with open(result_dates_filename, 'w') as f:
    f.write('key name year month day hour minutes seconds\n')
    for item in dates_list:
        f.write('%d %s %d %d %d %d %d %d\n' % tuple(item))

#--------------------------------------------------------------
# ANGLES

# all the possible image pairs
all_pairs = get_all_possible_pairs_from_list(image_filename_key_list)
pair_list = all_pairs
print('Pair list:', pair_list)

pair_angles_list = []

print('')
for i,pair in enumerate(pair_list):
    print('procesando {:03d}/{:03d}   pair:{}'.format(i+1, len(pair_list), pair), flush=True)

    ref_key = pair[0]
    sec_key = pair[1]
    ref_filename = image_filenames_dict[ref_key]
    sec_filename = image_filenames_dict[sec_key]
    _, ref_image_name = os.path.split(ref_filename)
    _, sec_image_name = os.path.split(sec_filename)
    ref_image_name_no_extension, _ = os.path.splitext(ref_image_name)
    sec_image_name_no_extension, _ = os.path.splitext(sec_image_name)

    # --------------------------------------------------------------------------
    # -----ANGLES---------------------------------------------------------------
    # angles [ref_zenith, ref_azimut, sec_zenith, sec_azimut, ref_sec_angle]
    # --------------------------------------------------------------------------
    angles = [ref_key, sec_key, ref_image_name_no_extension, sec_image_name_no_extension] + list(get_angles(ref_filename, sec_filename))
    pair_angles_list.append(angles)


with open(result_angles_filename, 'w') as f:
    f.write('ref_key sec_key ref sec ref_zenith ref_azimut sec_zenith sec_azimut ref_sec_angle\n')
    for item in pair_angles_list:
        f.write('%d %d %s %s %.3f %.3f %.3f %.3f %.3f\n' % tuple(item))



import os
import sys

import s2p
import rpcm
from utils import *

aoi_relative_padding = 0.1

mvs3d_geotiff_filename_list=[
'/home/agomez/Software/MultiStereo/COLMAP/COLMAP_FOR_SATELLITE/PRUEBAS/MVS3D/dataset/18DEC15WV031000015DEC18140455-P1BS-500515572010_01_P001_________AAE_0AAAAABPABJ0.NTF',
'/home/agomez/Software/MultiStereo/COLMAP/COLMAP_FOR_SATELLITE/PRUEBAS/MVS3D/dataset/18DEC15WV031000015DEC18140522-P1BS-500515572020_01_P001_________AAE_0AAAAABPABJ0.NTF',
'/home/agomez/Software/MultiStereo/COLMAP/COLMAP_FOR_SATELLITE/PRUEBAS/MVS3D/dataset/18DEC15WV031000015DEC18140544-P1BS-500515572060_01_P001_________AAE_0AAAAABPABJ0.NTF',
'/home/agomez/Software/MultiStereo/COLMAP/COLMAP_FOR_SATELLITE/PRUEBAS/MVS3D/dataset/22OCT15WV031000015OCT22140432-P1BS-500497282010_01_P001_________AAE_0AAAAABPABS0.NTF',
'/home/agomez/Software/MultiStereo/COLMAP/COLMAP_FOR_SATELLITE/PRUEBAS/MVS3D/dataset/18DEC15WV031000015DEC18140510-P1BS-500515572040_01_P001_________AAE_0AAAAABPABJ0.NTF',
'/home/agomez/Software/MultiStereo/COLMAP/COLMAP_FOR_SATELLITE/PRUEBAS/MVS3D/dataset/18DEC15WV031000015DEC18140533-P1BS-500515572050_01_P001_________AAE_0AAAAABPABJ0.NTF',
'/home/agomez/Software/MultiStereo/COLMAP/COLMAP_FOR_SATELLITE/PRUEBAS/MVS3D/dataset/18DEC15WV031000015DEC18140554-P1BS-500515572030_01_P001_________AAE_0AAAAABPABJ0.NTF',
'/home/agomez/Software/MultiStereo/COLMAP/COLMAP_FOR_SATELLITE/PRUEBAS/MVS3D/dataset/23OCT15WV031100015OCT23141928-P1BS-500497285030_01_P001_________AAE_0AAAAABPABO0.NTF'
    ]

mvs3d_geotiff_filename_list=[
'/media/agomez/FreeAgent GoFlex Drive/SATELITE/DATA/IARPA_DATA/GEOTIFF/02APR15WV031000015APR02134716-P1BS-500276959010_02_P001_________AAE_0AAAAABPABB0.TIF',
'/media/agomez/FreeAgent GoFlex Drive/SATELITE/DATA/IARPA_DATA/GEOTIFF/02APR15WV031000015APR02134802-P1BS-500276959010_02_P001_________AAE_0AAAAABPABC0.TIF',
'/media/agomez/FreeAgent GoFlex Drive/SATELITE/DATA/IARPA_DATA/GEOTIFF/03APR15WV031000015APR03140238-P1BS-500497283030_01_P001_________AAE_0AAAAABPABR0.TIF',
'/media/agomez/FreeAgent GoFlex Drive/SATELITE/DATA/IARPA_DATA/GEOTIFF/07JAN16WV031000016JAN07142142-P1BS-500537128030_01_P001_________AAE_0AAAAABPABE0.TIF',
'/media/agomez/FreeAgent GoFlex Drive/SATELITE/DATA/IARPA_DATA/GEOTIFF/07JAN16WV031000016JAN07142152-P1BS-500537128010_01_P001_________AAE_0AAAAABPABA0.TIF',
'/media/agomez/FreeAgent GoFlex Drive/SATELITE/DATA/IARPA_DATA/GEOTIFF/07JAN16WV031000016JAN07142202-P1BS-500537128020_01_P001_________AAE_0AAAAABPAAY0.TIF',
'/media/agomez/FreeAgent GoFlex Drive/SATELITE/DATA/IARPA_DATA/GEOTIFF/18DEC15WV031000015DEC18140455-P1BS-500515572010_01_P001_________AAE_0AAAAABPABJ0.TIF',
'/media/agomez/FreeAgent GoFlex Drive/SATELITE/DATA/IARPA_DATA/GEOTIFF/18DEC15WV031000015DEC18140510-P1BS-500515572040_01_P001_________AAE_0AAAAABPABJ0.TIF',
'/media/agomez/FreeAgent GoFlex Drive/SATELITE/DATA/IARPA_DATA/GEOTIFF/18DEC15WV031000015DEC18140522-P1BS-500515572020_01_P001_________AAE_0AAAAABPABJ0.TIF',
'/media/agomez/FreeAgent GoFlex Drive/SATELITE/DATA/IARPA_DATA/GEOTIFF/18DEC15WV031000015DEC18140533-P1BS-500515572050_01_P001_________AAE_0AAAAABPABJ0.TIF',
'/media/agomez/FreeAgent GoFlex Drive/SATELITE/DATA/IARPA_DATA/GEOTIFF/18DEC15WV031000015DEC18140544-P1BS-500515572060_01_P001_________AAE_0AAAAABPABJ0.TIF',
'/media/agomez/FreeAgent GoFlex Drive/SATELITE/DATA/IARPA_DATA/GEOTIFF/18DEC15WV031000015DEC18140554-P1BS-500515572030_01_P001_________AAE_0AAAAABPABJ0.TIF',
'/media/agomez/FreeAgent GoFlex Drive/SATELITE/DATA/IARPA_DATA/GEOTIFF/25DEC15WV031000015DEC25141655-P1BS-500526006010_01_P001_________AAE_0AAAAABPAAZ0.TIF',
'/media/agomez/FreeAgent GoFlex Drive/SATELITE/DATA/IARPA_DATA/GEOTIFF/25DEC15WV031000015DEC25141705-P1BS-500526006020_01_P001_________AAE_0AAAAABPAAW0.TIF'
]

image_filenames_dict_web = {
1:'/vsicurl/http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/15NOV14WV031000014NOV15135121-P1BS-500171606160_05_P005_________AAE_0AAAAABAABC0.TIF',
2:'/vsicurl/http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/05JAN15WV031000015JAN05135727-P1BS-500497282040_01_P001_________AAE_0AAAAABPABR0.TIF',
3:'/vsicurl/http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/11JAN15WV031000015JAN11135414-P1BS-500497283010_01_P001_________AAE_0AAAAABPABS0.TIF',
4:'/vsicurl/http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/23JAN15WV031000015JAN23134652-P1BS-500497282020_01_P001_________AAE_0AAAAABPABQ0.TIF',
5:'/vsicurl/http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/06FEB15WV031000015FEB06141035-P1BS-500497283080_01_P001_________AAE_0AAAAABPABP0.TIF',
6:'/vsicurl/http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/11FEB15WV031000015FEB11135123-P1BS-500497282030_01_P001_________AAE_0AAAAABPABR0.TIF',
7:'/vsicurl/http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/12FEB15WV031000015FEB12140652-P1BS-500497283100_01_P001_________AAE_0AAAAABPABQ0.TIF',
8:'/vsicurl/http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/08MAR15WV031000015MAR08134953-P1BS-500497284060_01_P001_________AAE_0AAAAABPABQ0.TIF',
9:'/vsicurl/http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/15MAR15WV031000015MAR15140133-P1BS-500497284070_01_P001_________AAE_0AAAAABPABQ0.TIF',
10:'/vsicurl/http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/21MAR15WV031000015MAR21135704-P1BS-500497282060_01_P001_________AAE_0AAAAABPABQ0.TIF',
11:'/vsicurl/http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/22MAR15WV031000015MAR22141208-P1BS-500497285090_01_P001_________AAE_0AAAAABPABQ0.TIF',
12:'/vsicurl/http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/02APR15WV031000015APR02134716-P1BS-500276959010_02_P001_________AAE_0AAAAABPABB0.TIF',
13:'/vsicurl/http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/02APR15WV031000015APR02134802-P1BS-500276959010_02_P001_________AAE_0AAAAABPABC0.TIF',
14:'/vsicurl/http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/03APR15WV031000015APR03140238-P1BS-500497283030_01_P001_________AAE_0AAAAABPABR0.TIF',
15:'/vsicurl/http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/22APR15WV031000015APR22140347-P1BS-500497282070_01_P001_________AAE_0AAAAABPABS0.TIF',
16:'/vsicurl/http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/04MAY15WV031000015MAY04135349-P1BS-500497283060_01_P001_________AAE_0AAAAABPABP0.TIF',
17:'/vsicurl/http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/05MAY15WV031000015MAY05140810-P1BS-500497282090_01_P001_________AAE_0AAAAABPABM0.TIF',
18:'/vsicurl/http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/12JUN15WV031000015JUN12140737-P1BS-500497284090_01_P001_________AAE_0AAAAABPABM0.TIF',
19:'/vsicurl/http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/18JUN15WV031000015JUN18140207-P1BS-500497285040_01_P001_________AAE_0AAAAABPABR0.TIF',
20:'/vsicurl/http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/19JUN15WV031100015JUN19141753-P1BS-500346924040_01_P001_________AAE_0AAAAABPABP0.TIF',
21:'/vsicurl/http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/24JUN15WV031000015JUN24135730-P1BS-500497285080_01_P001_________AAE_0AAAAABPABO0.TIF',
22:'/vsicurl/http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/30JUN15WV031000015JUN30135323-P1BS-500497282080_01_P001_________AAE_0AAAAABPABP0.TIF',
23:'/vsicurl/http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/14JUL15WV031000015JUL14141509-P1BS-500497283020_01_P001_________AAE_0AAAAABPABO0.TIF',
24:'/vsicurl/http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/19JUL15WV031000015JUL19135438-P1BS-500497285010_01_P001_________AAE_0AAAAABPABP0.TIF',
25:'/vsicurl/http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/27AUG15WV031000015AUG27141600-P1BS-500497284050_01_P001_________AAE_0AAAAABPABP0.TIF',
26:'/vsicurl/http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/01SEP15WV031000015SEP01135603-P1BS-500497284040_01_P001_________AAE_0AAAAABPABP0.TIF',
27:'/vsicurl/http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/08SEP15WV031000015SEP08140733-P1BS-500497283070_01_P001_________AAE_0AAAAABPABP0.TIF',
28:'/vsicurl/http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/14SEP15WV031000015SEP14140305-P1BS-500497285020_01_P001_________AAE_0AAAAABPABS0.TIF',
29:'/vsicurl/http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/15SEP15WV031000015SEP15141840-P1BS-500497285060_01_P001_________AAE_0AAAAABPABO0.TIF',
30:'/vsicurl/http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/27SEP15WV031000015SEP27140912-P1BS-500497284100_01_P001_________AAE_0AAAAABPABQ0.TIF',
31:'/vsicurl/http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/03OCT15WV031000015OCT03140452-P1BS-500497283050_01_P001_________AAE_0AAAAABPABR0.TIF',
32:'/vsicurl/http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/22OCT15WV031000015OCT22140432-P1BS-500497282010_01_P001_________AAE_0AAAAABPABS0.TIF',
33:'/vsicurl/http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/23OCT15WV031100015OCT23141928-P1BS-500497285030_01_P001_________AAE_0AAAAABPABO0.TIF',
34:'/vsicurl/http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/11DEC15WV030900015DEC11135506-P1BS-500510591010_01_P001_________AAE_0AAAAABPABJ0.TIF',
35:'/vsicurl/http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/18DEC15WV031000015DEC18140455-P1BS-500515572010_01_P001_________AAE_0AAAAABPABJ0.TIF',
36:'/vsicurl/http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/18DEC15WV031000015DEC18140510-P1BS-500515572040_01_P001_________AAE_0AAAAABPABJ0.TIF',
37:'/vsicurl/http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/18DEC15WV031000015DEC18140522-P1BS-500515572020_01_P001_________AAE_0AAAAABPABJ0.TIF',
38:'/vsicurl/http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/18DEC15WV031000015DEC18140533-P1BS-500515572050_01_P001_________AAE_0AAAAABPABJ0.TIF',
39:'/vsicurl/http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/18DEC15WV031000015DEC18140544-P1BS-500515572060_01_P001_________AAE_0AAAAABPABJ0.TIF',
40:'/vsicurl/http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/18DEC15WV031000015DEC18140554-P1BS-500515572030_01_P001_________AAE_0AAAAABPABJ0.TIF',
41:'/vsicurl/http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/19DEC15WV031000015DEC19142039-P1BS-500514410020_01_P001_________AAE_0AAAAABPABW0.TIF',
42:'/vsicurl/http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/25DEC15WV031000015DEC25141655-P1BS-500526006010_01_P001_________AAE_0AAAAABPAAZ0.TIF',
43:'/vsicurl/http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/25DEC15WV031000015DEC25141705-P1BS-500526006020_01_P001_________AAE_0AAAAABPAAW0.TIF',
44:'/vsicurl/http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/07JAN16WV031000016JAN07142142-P1BS-500537128030_01_P001_________AAE_0AAAAABPABE0.TIF',
45:'/vsicurl/http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/07JAN16WV031000016JAN07142152-P1BS-500537128010_01_P001_________AAE_0AAAAABPABA0.TIF',
46:'/vsicurl/http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/07JAN16WV031000016JAN07142202-P1BS-500537128020_01_P001_________AAE_0AAAAABPAAY0.TIF',
47:'/vsicurl/http://138.231.80.166:2334/iarpa-2016/cloud_optimized_geotif/13JAN16WV031100016JAN13141501-P1BS-500541801010_01_P001_________AAE_0AAAAABPABJ0.TIF',
}

image_filenames_dict_local = {
1:'/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/GEOTIFF/15NOV14WV031000014NOV15135121-P1BS-500171606160_05_P005_________AAE_0AAAAABAABC0.TIF',
2:'/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/GEOTIFF/05JAN15WV031000015JAN05135727-P1BS-500497282040_01_P001_________AAE_0AAAAABPABR0.TIF',
3:'/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/GEOTIFF/11JAN15WV031000015JAN11135414-P1BS-500497283010_01_P001_________AAE_0AAAAABPABS0.TIF',
4:'/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/GEOTIFF/23JAN15WV031000015JAN23134652-P1BS-500497282020_01_P001_________AAE_0AAAAABPABQ0.TIF',
5:'/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/GEOTIFF/06FEB15WV031000015FEB06141035-P1BS-500497283080_01_P001_________AAE_0AAAAABPABP0.TIF',
6:'/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/GEOTIFF/11FEB15WV031000015FEB11135123-P1BS-500497282030_01_P001_________AAE_0AAAAABPABR0.TIF',
7:'/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/GEOTIFF/12FEB15WV031000015FEB12140652-P1BS-500497283100_01_P001_________AAE_0AAAAABPABQ0.TIF',
8:'/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/GEOTIFF/08MAR15WV031000015MAR08134953-P1BS-500497284060_01_P001_________AAE_0AAAAABPABQ0.TIF',
9:'/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/GEOTIFF/15MAR15WV031000015MAR15140133-P1BS-500497284070_01_P001_________AAE_0AAAAABPABQ0.TIF',
10:'/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/GEOTIFF/21MAR15WV031000015MAR21135704-P1BS-500497282060_01_P001_________AAE_0AAAAABPABQ0.TIF',
11:'/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/GEOTIFF/22MAR15WV031000015MAR22141208-P1BS-500497285090_01_P001_________AAE_0AAAAABPABQ0.TIF',
12:'/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/GEOTIFF/02APR15WV031000015APR02134716-P1BS-500276959010_02_P001_________AAE_0AAAAABPABB0.TIF',
13:'/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/GEOTIFF/02APR15WV031000015APR02134802-P1BS-500276959010_02_P001_________AAE_0AAAAABPABC0.TIF',
14:'/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/GEOTIFF/03APR15WV031000015APR03140238-P1BS-500497283030_01_P001_________AAE_0AAAAABPABR0.TIF',
15:'/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/GEOTIFF/22APR15WV031000015APR22140347-P1BS-500497282070_01_P001_________AAE_0AAAAABPABS0.TIF',
16:'/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/GEOTIFF/04MAY15WV031000015MAY04135349-P1BS-500497283060_01_P001_________AAE_0AAAAABPABP0.TIF',
17:'/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/GEOTIFF/05MAY15WV031000015MAY05140810-P1BS-500497282090_01_P001_________AAE_0AAAAABPABM0.TIF',
18:'/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/GEOTIFF/12JUN15WV031000015JUN12140737-P1BS-500497284090_01_P001_________AAE_0AAAAABPABM0.TIF',
19:'/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/GEOTIFF/18JUN15WV031000015JUN18140207-P1BS-500497285040_01_P001_________AAE_0AAAAABPABR0.TIF',
20:'/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/GEOTIFF/19JUN15WV031100015JUN19141753-P1BS-500346924040_01_P001_________AAE_0AAAAABPABP0.TIF',
21:'/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/GEOTIFF/24JUN15WV031000015JUN24135730-P1BS-500497285080_01_P001_________AAE_0AAAAABPABO0.TIF',
22:'/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/GEOTIFF/30JUN15WV031000015JUN30135323-P1BS-500497282080_01_P001_________AAE_0AAAAABPABP0.TIF',
23:'/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/GEOTIFF/14JUL15WV031000015JUL14141509-P1BS-500497283020_01_P001_________AAE_0AAAAABPABO0.TIF',
24:'/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/GEOTIFF/19JUL15WV031000015JUL19135438-P1BS-500497285010_01_P001_________AAE_0AAAAABPABP0.TIF',
25:'/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/GEOTIFF/27AUG15WV031000015AUG27141600-P1BS-500497284050_01_P001_________AAE_0AAAAABPABP0.TIF',
26:'/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/GEOTIFF/01SEP15WV031000015SEP01135603-P1BS-500497284040_01_P001_________AAE_0AAAAABPABP0.TIF',
27:'/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/GEOTIFF/08SEP15WV031000015SEP08140733-P1BS-500497283070_01_P001_________AAE_0AAAAABPABP0.TIF',
28:'/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/GEOTIFF/14SEP15WV031000015SEP14140305-P1BS-500497285020_01_P001_________AAE_0AAAAABPABS0.TIF',
29:'/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/GEOTIFF/15SEP15WV031000015SEP15141840-P1BS-500497285060_01_P001_________AAE_0AAAAABPABO0.TIF',
30:'/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/GEOTIFF/27SEP15WV031000015SEP27140912-P1BS-500497284100_01_P001_________AAE_0AAAAABPABQ0.TIF',
31:'/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/GEOTIFF/03OCT15WV031000015OCT03140452-P1BS-500497283050_01_P001_________AAE_0AAAAABPABR0.TIF',
32:'/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/GEOTIFF/22OCT15WV031000015OCT22140432-P1BS-500497282010_01_P001_________AAE_0AAAAABPABS0.TIF',
33:'/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/GEOTIFF/23OCT15WV031100015OCT23141928-P1BS-500497285030_01_P001_________AAE_0AAAAABPABO0.TIF',
34:'/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/GEOTIFF/11DEC15WV030900015DEC11135506-P1BS-500510591010_01_P001_________AAE_0AAAAABPABJ0.TIF',
35:'/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/GEOTIFF/18DEC15WV031000015DEC18140455-P1BS-500515572010_01_P001_________AAE_0AAAAABPABJ0.TIF',
36:'/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/GEOTIFF/18DEC15WV031000015DEC18140510-P1BS-500515572040_01_P001_________AAE_0AAAAABPABJ0.TIF',
37:'/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/GEOTIFF/18DEC15WV031000015DEC18140522-P1BS-500515572020_01_P001_________AAE_0AAAAABPABJ0.TIF',
38:'/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/GEOTIFF/18DEC15WV031000015DEC18140533-P1BS-500515572050_01_P001_________AAE_0AAAAABPABJ0.TIF',
39:'/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/GEOTIFF/18DEC15WV031000015DEC18140544-P1BS-500515572060_01_P001_________AAE_0AAAAABPABJ0.TIF',
40:'/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/GEOTIFF/18DEC15WV031000015DEC18140554-P1BS-500515572030_01_P001_________AAE_0AAAAABPABJ0.TIF',
41:'/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/GEOTIFF/19DEC15WV031000015DEC19142039-P1BS-500514410020_01_P001_________AAE_0AAAAABPABW0.TIF',
42:'/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/GEOTIFF/25DEC15WV031000015DEC25141655-P1BS-500526006010_01_P001_________AAE_0AAAAABPAAZ0.TIF',
43:'/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/GEOTIFF/25DEC15WV031000015DEC25141705-P1BS-500526006020_01_P001_________AAE_0AAAAABPAAW0.TIF',
44:'/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/GEOTIFF/07JAN16WV031000016JAN07142142-P1BS-500537128030_01_P001_________AAE_0AAAAABPABE0.TIF',
45:'/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/GEOTIFF/07JAN16WV031000016JAN07142152-P1BS-500537128010_01_P001_________AAE_0AAAAABPABA0.TIF',
46:'/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/GEOTIFF/07JAN16WV031000016JAN07142202-P1BS-500537128020_01_P001_________AAE_0AAAAABPAAY0.TIF',
47:'/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/GEOTIFF/13JAN16WV031100016JAN13141501-P1BS-500541801010_01_P001_________AAE_0AAAAABPABJ0.TIF',
}

mvs3d_gt_crop_filename_list=[
    '/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/gt/mvs3d_gt_crop_01.tif',
    '/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/gt/mvs3d_gt_crop_02.tif',
    '/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/gt/mvs3d_gt_crop_03.tif',
    '/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/gt/mvs3d_gt_crop_04.tif',
    '/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/gt/mvs3d_gt_crop_05.tif',
    #
    # '/home/agomez/Documents/iie/satelite/DATA/IARPA_DATA/ground_truth/mvs3d_gt_crop_01.tif',
    # '/home/agomez/Documents/iie/satelite/DATA/IARPA_DATA/ground_truth/mvs3d_gt_crop_02.tif',
    # '/home/agomez/Documents/iie/satelite/DATA/IARPA_DATA/ground_truth/mvs3d_gt_crop_03.tif',
    # '/home/agomez/Documents/iie/satelite/DATA/IARPA_DATA/ground_truth/mvs3d_gt_crop_04.tif',
    # '/home/agomez/Documents/iie/satelite/DATA/IARPA_DATA/ground_truth/mvs3d_gt_crop_05.tif'
]

#output_dir = '/home/agomez/Documents/iie/satelite/DATA/IARPA_DATA/cropped'

# Near in time images
output_dir = '/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/cropped'

# Far in time images
keys = [2, 12, 20, 26, 33, 43]
output_dir = '/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/cropped_far_in_time'


# All images
output_dir = '/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/cropped_all'
keys = range(1,48)
image_filenames_dict = image_filenames_dict_web


for k in keys:

    image_filename = image_filenames_dict[k]
    rpc = rpcm.rpc_from_geotiff(image_filename)
    print(k, image_filename)

    _, image_name = os.path.split(image_filename)
    image_acquisition_date_string = image_name[16:29]

    for mvs3d_gt_crop_filename in mvs3d_gt_crop_filename_list:
        gt_filename = mvs3d_gt_crop_filename
        _, gt_name = os.path.split(gt_filename)
        gt_name_no_extension,_ = os.path.splitext(gt_name)

        gt_cropped_data_dir  = os.path.join(output_dir, gt_name_no_extension)
        if not os.path.isdir(gt_cropped_data_dir):
            os.makedirs(gt_cropped_data_dir)

        # get aoi from gt
        aoi, min_height, max_height, zone_hemisphere, zone_letter, zone_number, utm_bbx, lonlat_bbx = aoi_info_from_geotiff_gt(
            ref_filename=image_filename,
            gt_filename=gt_filename,
            padding=aoi_relative_padding)



        crop_image_filename = os.path.join(gt_cropped_data_dir, image_acquisition_date_string + '.tif')

        # 
        x, y, w, h = rpcm.utils.bounding_box_of_projected_aoi(rpc, aoi, z=min_height)
        s2p.common.image_crop_gdal(image_filename, x, y, w, h, crop_image_filename)  # no sirve para web



#!/usr/bin/env python
import argparse
import csv
import json
import io
import pathlib
import sys

from typing import Any

import sinter
import stim

src_path = pathlib.Path(__file__).parent.parent / 'src'
assert src_path.exists()
sys.path.append(str(src_path))

import cultiv
import gen

# Rewritten (and extended) data starting from what was included in https://arxiv.org/abs/1905.06903

litinsky_data_csv = """
state,c,lbl,p_phys,p_out,qubits,cycles
T,2020 Litinski,"T15₇,₃,₃",1e-4,4.4e-8,810,18.1
T,2020 Litinski,"T15₉,₃,₃",1e-4,9.3e-10,1150,18.1
T,2020 Litinski,"T15₁₁,₅,₅",1e-4,1.9e-11,2070,30.0
T,2020 Litinski,"T15₉,₃,₃×T20₁₅,₇,₉",1e-4,2.4e-15,16400,90.3
T,2020 Litinski,"T15₉,₃,₃×T15₂₅,₉,₉",1e-4,6.3e-25,18600,67.8
T,2020 Litinski,"T15₁₇,₇,₇",1e-3,4.5e-8,4620,42.6
T,2020 Litinski,"T15₁₃,₅,₅×T20₂₃,₁₁,₁₃",1e-3,1.4e-10,43300,130
T,2020 Litinski,"T15₁₃,₅,₅×T20₂₇,₁₃,₁₅",1e-3,2.6e-11,46800,157
T,2020 Litinski,"T15₁₁,₅,₅×T15₂₅,₁₁,₁₁",1e-3,2.7e-12,30700,82.5
T,2020 Litinski,"T15₁₃,₅,₅×T15₂₉,₁₁,₁₃",1e-3,3.3e-14,39100,97.5
T,2020 Litinski,"T15₁₇,₇,₇×T15₄₁,₁₇,₁₇",1e-3,4.5e-20,73400,128
T,2020 Litinski,"T15₉,₃,₃",1e-4,1.5e-9,762,36.2
T,2020 Litinski,"T15₉,₅,₅×T15₂₁,₉,₁₁",1e-3,6.1e-10,7780,469
CCZ,2020 Litinski,"T15₇,₃,₃×CCZ8₁₅,₇,₉",1e-4,7.2e-14,12400,36.1
CCZ,2020 Litinski,"T15₁₃,₇,₇×CCZ8₂₅,₁₅,₁₅",1e-3,5.2e-11,47000,60.0
T,2019 Litinski,"T15",1e-4,3.5e-11,3720,143
CCZ,2018 Gidney et al,"T15×CCZ8",1e-3,6.3e-11,126400,171
2T,2018 Gidney et al,"T15→2T",1e-3,6.3e-11,126400,202
T,2018 Fowler et al,"T15×T15",1e-3,1.0e-14,177000,202
T,2018 Fowler et al,"T15",1e-3,3.5e-8,14400,120
T,2012 Fowler et al,"T15×T15",1e-3,3.0e-15,800000,250
T[d=3],2024 Itogawa et al,"T15",1e-3,1e-4,25,4
T,2024 Hirano et al,"T15",1e-3,1e-8,3e5,1
T,2024 Hirano et al,"T15",1e-3,1e-10,6e5,1
T,2024 Hirano et al,"T15",1e-3,2e-11,2e6,1
""".strip()

# Data from https://arxiv.org/abs/2302.12292
hook_data_csv = """
shots,errors,discards,seconds,decoder,strong_id,json_metadata
  53936471,      1005,   3405822, 11235.0,internal_correlated,337c23d97678bbeaef38f696e0031ebf3bd2e27cb70d66f7063ba0c5a610300e,"{""b"":""hook_inject_X"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.0001,""post_d"":5,""post_q"":49,""post_r"":2,""q"":449,""r"":15}"
  51956164,      1000,   6379976, 10435.3,internal_correlated,83bd3cda92815d378b45bb67bcd42c05dde6d2e5c539d0791899c9a4d003a0cb,"{""b"":""hook_inject_X"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.0001,""post_d"":7,""post_q"":97,""post_r"":2,""q"":449,""r"":15}"
  26035235,      1016,   3181744,  4343.0,internal_correlated,2eda7cc9c4c4c40e7222f2969ea7ef069a2d38ef7ffed564c4c150839417e708,"{""b"":""hook_inject_X"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.0002,""post_d"":5,""post_q"":49,""post_r"":2,""q"":449,""r"":15}"
  28955612,      1006,   6674503,  4783.6,internal_correlated,9a402eeb62add7911b90a3176c55749ed6e63e92d1a04f63b82a887ac5b4bce8,"{""b"":""hook_inject_X"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.0002,""post_d"":7,""post_q"":97,""post_r"":2,""q"":449,""r"":15}"
  16350023,      1004,   2902698,  2837.7,internal_correlated,46ef37cad45cc1e05da4bfb313b0c2466c5888afdc2ec4c1258b3a66ddcb891a,"{""b"":""hook_inject_X"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.0003,""post_d"":5,""post_q"":49,""post_r"":2,""q"":449,""r"":15}"
  21084041,      1011,   6848845,  3152.8,internal_correlated,df4fa1af27356ce8eb481b4f6b44d59346ed69cda777e0892a501bd4c3933ed5,"{""b"":""hook_inject_X"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.0003,""post_d"":7,""post_q"":97,""post_r"":2,""q"":449,""r"":15}"
   9870204,      1014,   2741603,  1569.3,internal_correlated,c8cdd354d8e002cad96f09202715bbb356e3ef4184b48239f3e300404045ecd9,"{""b"":""hook_inject_X"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.0005,""post_d"":5,""post_q"":49,""post_r"":2,""q"":449,""r"":15}"
  15275741,      1001,   7336679,  1843.1,internal_correlated,1c6b90e7ebb29407385af0851571a3b47acb95e689e5065fc47cff2b808dd252,"{""b"":""hook_inject_X"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.0005,""post_d"":7,""post_q"":97,""post_r"":2,""q"":449,""r"":15}"
   7022678,      1011,   2571849,  1047.5,internal_correlated,19dd780c8a3f29753985a8df3561064e8fbfeb8a465af3552adb40e6a9d3142b,"{""b"":""hook_inject_X"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.0007,""post_d"":5,""post_q"":49,""post_r"":2,""q"":449,""r"":15}"
  13353495,      1015,   8009857,  2036.6,internal_correlated,dc38fcdd3ad67a79c97ff9f34769c03c242345830f9f32b663375519709f468b,"{""b"":""hook_inject_X"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.0007,""post_d"":7,""post_q"":97,""post_r"":2,""q"":449,""r"":15}"
   4263786,      1004,   2038050,   794.7,internal_correlated,7d17d65098bd3ae188245aef29987da6a2a2079041cf4fd744b1bde607842455,"{""b"":""hook_inject_X"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":5,""post_q"":49,""post_r"":2,""q"":449,""r"":15}"
  11457471,      1035,   8354715,  1556.5,internal_correlated,78360d83316c4f3f4de721e4c32339bc7b30aca5add9b38778dc3769514283d4,"{""b"":""hook_inject_X"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":7,""post_q"":97,""post_r"":2,""q"":449,""r"":15}"
   1434548,      1013,   1041817,   368.3,internal_correlated,daad49bcfc072011b9a2ccfc06db2fb3678ff0d9acfbdda854f41d9142801daf,"{""b"":""hook_inject_X"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.002,""post_d"":5,""post_q"":49,""post_r"":2,""q"":449,""r"":15}"
  10518719,      1087,   9742266,   914.0,internal_correlated,90ff2fa42569ec73b4c80beb880a13e28160437955621e3085197b01e110b2af,"{""b"":""hook_inject_X"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.002,""post_d"":7,""post_q"":97,""post_r"":2,""q"":449,""r"":15}"
    585930,      1001,    501546,   186.5,internal_correlated,4681ccca4ec2d5bc0fa5d8c9bd23404910cfeb3c0e49bcee4707f90cf4786d4f,"{""b"":""hook_inject_X"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.003,""post_d"":5,""post_q"":49,""post_r"":2,""q"":449,""r"":15}"
   7055049,      1093,   6912754,   451.4,internal_correlated,1433fb06332256ba9ec8bc581865f6a431b36db22c7d243fb4f7a84a140c8505,"{""b"":""hook_inject_X"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.003,""post_d"":7,""post_q"":97,""post_r"":2,""q"":449,""r"":15}"
    152279,      1014,    146064,    53.5,internal_correlated,e06437bee075e79333e69ae3e00f87e1946b1292e7ed3ff8dedcc770f422ab1d,"{""b"":""hook_inject_X"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.005,""post_d"":5,""post_q"":49,""post_r"":2,""q"":449,""r"":15}"
   4783814,      1034,   4776479,   233.1,internal_correlated,45dff86556999e79a5378e5c42a3cf5d1bb314832fe6cab4c1bdb8d315f92b72,"{""b"":""hook_inject_X"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.005,""post_d"":7,""post_q"":97,""post_r"":2,""q"":449,""r"":15}"
   1081601,      1016,   1079566,   108.4,internal_correlated,9791918d7e1fa1ae52e61d38671c653894bf2e3f54e3dfed2a07c91cfce9d9f7,"{""b"":""hook_inject_X"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.01,""post_d"":5,""post_q"":49,""post_r"":2,""q"":449,""r"":15}"
 100000000,       153,  99999691,  5551.8,internal_correlated,f97279cc889b1e98cc8ba43af806c101dc32b247991f2ff13e419671831966e1,"{""b"":""hook_inject_X"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.01,""post_d"":7,""post_q"":97,""post_r"":2,""q"":449,""r"":15}"
  42890942,      1023,   2725432, 14104.6,internal_correlated,729aedebf140a3c102d13d31e0d51edfb1963608c73a1a0660c06aac265db4c5,"{""b"":""hook_inject_Y"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.0001,""post_d"":5,""post_q"":49,""post_r"":2,""q"":463,""r"":15}"
  46089861,      1035,   5695822, 13345.7,internal_correlated,92379d7db08940eba1b49a1b7d6ba44b222d7e2cf83179cbb63d91c0c37a4913,"{""b"":""hook_inject_Y"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.0001,""post_d"":7,""post_q"":97,""post_r"":2,""q"":463,""r"":15}"
  19285330,      1005,   2369229,  5403.0,internal_correlated,3b9ec8a5b72b8cca425fd056604cd122892519cf580fb89b0889fad491b979b2,"{""b"":""hook_inject_Y"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.0002,""post_d"":5,""post_q"":49,""post_r"":2,""q"":463,""r"":15}"
  22089744,      1012,   5121860,  5803.9,internal_correlated,c9e28bdcd02f0fce9b25ec84f5789520b0325d90067c38a162931d7e878ca0a8,"{""b"":""hook_inject_Y"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.0002,""post_d"":7,""post_q"":97,""post_r"":2,""q"":463,""r"":15}"
  12271750,      1004,   2191379,  3340.9,internal_correlated,35cd4d767c8001a1de9f64e534739605a94df79ea82156ecdb8024cc89f82f33,"{""b"":""hook_inject_Y"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.0003,""post_d"":5,""post_q"":49,""post_r"":2,""q"":463,""r"":15}"
  16950367,      1055,   5539565,  4277.6,internal_correlated,977a529412c4a0b9289e5c1c2b4472d308984b30dec4273d4753e5a5173bcfd8,"{""b"":""hook_inject_Y"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.0003,""post_d"":7,""post_q"":97,""post_r"":2,""q"":463,""r"":15}"
   6465480,      1019,   1805190,  1876.7,internal_correlated,f1437661806605e0ea3cc8d75b558f4a83cd285befb8749cb5ea4457e547797f,"{""b"":""hook_inject_Y"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.0005,""post_d"":5,""post_q"":49,""post_r"":2,""q"":463,""r"":15}"
  10332377,      1010,   4986745,  2442.6,internal_correlated,7631676bd3cdf06859d5d94420ad5a8c2dbe79757b09c06296aab2f950ac7178,"{""b"":""hook_inject_Y"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.0005,""post_d"":7,""post_q"":97,""post_r"":2,""q"":463,""r"":15}"
   3977191,      1010,   1461682,  1165.2,internal_correlated,d2f79998b876d03972ecc54f42acf418f343e3dca00eefdfe0bfabbd60f61574,"{""b"":""hook_inject_Y"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.0007,""post_d"":5,""post_q"":49,""post_r"":2,""q"":463,""r"":15}"
   8054000,      1042,   4849157,  1786.1,internal_correlated,8d5aa99f3e0f321b43c01fbe9246e554a3af2790a80562f550345f87e46de81d,"{""b"":""hook_inject_Y"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.0007,""post_d"":7,""post_q"":97,""post_r"":2,""q"":463,""r"":15}"
   2083686,      1031,    999621,   723.5,internal_correlated,4b3858dd3fc618db089e63545c728aa750612a1d46926e1b9eb030ab61f0fac1,"{""b"":""hook_inject_Y"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":5,""post_q"":49,""post_r"":2,""q"":463,""r"":15}"
   6829621,      1007,   4997326,  1084.4,internal_correlated,aa41f800daf74b270ccd8062eae68517defbd0876bcce03387ac6b8caee410ec,"{""b"":""hook_inject_Y"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":7,""post_q"":97,""post_r"":2,""q"":463,""r"":15}"
    603085,      1048,    439071,   322.1,internal_correlated,41f613c109ffaee2c6414d92b96e4183e772c83fb37669c6a220917fe9b933f5,"{""b"":""hook_inject_Y"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.002,""post_d"":5,""post_q"":49,""post_r"":2,""q"":463,""r"":15}"
   4151786,      1003,   3850663,   606.4,internal_correlated,a8b4d31363d5493b5f561b62d12abbce529cd4edd7ede099c5a3f55c0da97ca0,"{""b"":""hook_inject_Y"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.002,""post_d"":7,""post_q"":97,""post_r"":2,""q"":463,""r"":15}"
    229003,      1002,    196644,   153.3,internal_correlated,560f4b8ab0018d4d605eb293b6e9a0644c59f4baa66b5c30e80f02f69465fca9,"{""b"":""hook_inject_Y"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.003,""post_d"":5,""post_q"":49,""post_r"":2,""q"":463,""r"":15}"
   2716166,      1067,   2662523,   318.8,internal_correlated,3e0003c32cc204fdfb61dd62a782e64ee5186640ff48f1bf1309f3b17ef62789,"{""b"":""hook_inject_Y"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.003,""post_d"":7,""post_q"":97,""post_r"":2,""q"":463,""r"":15}"
     78140,      1015,     75102,    57.3,internal_correlated,62f372bd178ec84622bf63fd94a6fb2742370eb6067bc81c2e44f495477b31df,"{""b"":""hook_inject_Y"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.005,""post_d"":5,""post_q"":49,""post_r"":2,""q"":463,""r"":15}"
   2183642,      1009,   2180399,   186.6,internal_correlated,4410fda83715305f6dc94683ea96c41cad713bfdaed59f989fa2430bd633775d,"{""b"":""hook_inject_Y"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.005,""post_d"":7,""post_q"":97,""post_r"":2,""q"":463,""r"":15}"
   1095428,      1000,   1093465,   185.0,internal_correlated,f3b4b4029014debf13daf454f08b5a9c7664bb2264ddc3c7a846bb5b44d43e84,"{""b"":""hook_inject_Y"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.01,""post_d"":5,""post_q"":49,""post_r"":2,""q"":463,""r"":15}"
 100000000,       138,  99999734,  9970.6,internal_correlated,3278117f0755a58a1a57a6933c95b03be9b988706631e0ffa3eb48552f5ee6f4,"{""b"":""hook_inject_Y"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.01,""post_d"":7,""post_q"":97,""post_r"":2,""q"":463,""r"":15}"
     40165,      1019,      3261,    15.9,internal_correlated,a36ed7adafd081374c7a7d984d91460f3e1d0d7a0417fc45b4af0df338f70888,"{""b"":""hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":2,""post_q"":7,""post_r"":2,""q"":449,""r"":15}"
     45036,      1096,      5526,    19.3,internal_correlated,1908f11ca32751a579716372bc1b6832939f3a925aa91447978125c4cb4d6502,"{""b"":""hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":2,""post_q"":7,""post_r"":3,""q"":449,""r"":15}"
     44580,      1020,      7422,    19.6,internal_correlated,e62d5bec4e4baf24d7bfa8ed919e7c99e0f0f0d5275bfc2f766f771c6333adc2,"{""b"":""hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":2,""post_q"":7,""post_r"":4,""q"":449,""r"":15}"
     51971,      1079,     10697,    17.2,internal_correlated,248d119e5cf8ff48fca4893142adc29893ee3fee34e9d68827f3bb1956bd4224,"{""b"":""hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":2,""post_q"":7,""post_r"":5,""q"":449,""r"":15}"
     51002,      1028,     12337,    17.0,internal_correlated,5737a753fc77fb56aca74571c94c9b461f0f81f971ae66031f485102d4057bac,"{""b"":""hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":2,""post_q"":7,""post_r"":6,""q"":449,""r"":15}"
     89819,      1014,      7758,    26.5,internal_correlated,15c03ef830cf83f375ad8ba740ec2e9cd5d9371d4af9673a42e80d9ad80b00d0,"{""b"":""hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":2,""post_q"":449,""post_r"":1,""q"":449,""r"":15}"
    338515,      1054,     66617,    96.3,internal_correlated,e659e3af393bb01e746de32cb69c3b91f8c0a8f8a34ce41a1cb12a323b342ded,"{""b"":""hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":3,""post_q"":17,""post_r"":2,""q"":449,""r"":15}"
    371904,      1011,    108108,    88.0,internal_correlated,e1c9428ed1aa6fae09737e21abd11fbdc10f5788d5b376ddc676f02702f8a176,"{""b"":""hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":3,""post_q"":17,""post_r"":3,""q"":449,""r"":15}"
    423380,      1022,    158190,   123.2,internal_correlated,499d10d4b223438d0fdf8a4809b491b48161d138f594122b6d49b1a52c777699,"{""b"":""hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":3,""post_q"":17,""post_r"":4,""q"":449,""r"":15}"
    515834,      1057,    230878,   111.7,internal_correlated,e7fdd88ad6695748f3228bdc4f10ad1af328c89d24de0a6c8c90a16cbf022a00,"{""b"":""hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":3,""post_q"":17,""post_r"":5,""q"":449,""r"":15}"
    504304,      1012,    257352,    96.2,internal_correlated,e5e486b4ed4aea1df5d56b4933b2446e48dd0db24ce2d47034b623d99d12154b,"{""b"":""hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":3,""post_q"":17,""post_r"":6,""q"":449,""r"":15}"
    592839,      1003,    124491,   161.6,internal_correlated,a1ea14c91cbab7360bd7c9c6a7bbd9042a925cf9f8a57448f93ccad5738e579f,"{""b"":""hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":3,""post_q"":449,""post_r"":1,""q"":449,""r"":15}"
    950213,      1120,    317828,   234.7,internal_correlated,56dd46e2d6ed16b7146f6259c6498e7f3eabdc28a9b30ab7c7574608682a5a63,"{""b"":""hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":4,""post_q"":31,""post_r"":2,""q"":449,""r"":15}"
   1159982,      1019,    549103,   245.7,internal_correlated,c4059698a500f1be231f002ec5ce1dfa9024aae472e38dd50a1a68971f75d534,"{""b"":""hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":4,""post_q"":31,""post_r"":3,""q"":449,""r"":15}"
   1445612,      1036,    840699,   286.7,internal_correlated,cda0cd2cef293b7b8eaaae015c0c2d52c7d214f2eca43a5ecee6ff2304e63ab0,"{""b"":""hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":4,""post_q"":31,""post_r"":4,""q"":449,""r"":15}"
   1681797,      1009,   1123873,   275.8,internal_correlated,79015ede30584f5ec4918af6f9911b751afe51f78e39182659237135b3833a08,"{""b"":""hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":4,""post_q"":31,""post_r"":5,""q"":449,""r"":15}"
   2075537,      1000,   1529518,   316.6,internal_correlated,de95ec7f2e001744198d7a8f686c90986de0c093e2394f2d4e6b190a1edc58e8,"{""b"":""hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":4,""post_q"":31,""post_r"":6,""q"":449,""r"":15}"
   1362826,      1020,    477527,   344.3,internal_correlated,dac252746001032462e87c836b7a261abf749c781ab31895701774dee16548fd,"{""b"":""hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":4,""post_q"":449,""post_r"":1,""q"":449,""r"":15}"
   2160000,      1006,   1037590,   422.6,internal_correlated,6ce1e1370be8c9244619897568db3d23197fd2ef5d2dabf6527c776e87335876,"{""b"":""hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":5,""post_q"":49,""post_r"":2,""q"":449,""r"":15}"
   4298577,      1063,   2759349,   747.4,internal_correlated,554b5addc3c14a03712e00c07c60940d37a35c1dd79645d8cdda5d81361c4cdd,"{""b"":""hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":5,""post_q"":49,""post_r"":3,""q"":449,""r"":15}"
   6373783,      1067,   4797643,   914.8,internal_correlated,337e0b107c3c5d2ff3bc2a535e2a2545055f7eb2965ee89aba5da630090f534a,"{""b"":""hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":5,""post_q"":49,""post_r"":4,""q"":449,""r"":15}"
   8523560,      1007,   7068288,  1099.9,internal_correlated,d2c87572c3e7f29219a7e021ed62ceda9c40e01cc3100a4a793e23af9be0625f,"{""b"":""hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":5,""post_q"":49,""post_r"":5,""q"":449,""r"":15}"
  12750108,      1051,  11245597,  1566.9,internal_correlated,f8fa6c053dd5c324e87dded6d5cb9c811e44bc129ef9dd2e4630d26160102fa8,"{""b"":""hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":5,""post_q"":49,""post_r"":6,""q"":449,""r"":15}"
   2949929,      1002,   1468691,   601.8,internal_correlated,51f600f26c5077b3c8d2ee8aef3077fe72d011c95aef6d7ea5f9a1b67cb012cf,"{""b"":""hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":5,""post_q"":449,""post_r"":1,""q"":449,""r"":15}"
   4216995,      1049,   2596823,   784.2,internal_correlated,ff7bd88d85da7f6251480f200c53dc06aff0c84b4a36a31153fdd52e9f832a43,"{""b"":""hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":6,""post_q"":71,""post_r"":2,""q"":449,""r"":15}"
  11146721,      1013,   8656981,  1525.8,internal_correlated,b9e091bf5bb899741147062077e3a7eef41d346f56036664fe2359edfc58c141,"{""b"":""hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":6,""post_q"":71,""post_r"":3,""q"":449,""r"":15}"
  18665994,      1058,  16238233,  2260.7,internal_correlated,ab353c31592ecce717aa1d1b157cc1c981e9fe7cda2547ef68114085185e7495,"{""b"":""hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":6,""post_q"":71,""post_r"":4,""q"":449,""r"":15}"
  31091615,      1007,  28736552,  3422.8,internal_correlated,2e7fac51dd32c8423f87d3a2491139c2907ffec1743f7ea87f133693bbd81d4a,"{""b"":""hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":6,""post_q"":71,""post_r"":5,""q"":449,""r"":15}"
  54067871,      1083,  51684937,  6095.0,internal_correlated,ce29cc9639205b3e5a30fa64302f415c6ee6b160780e4579fb8149ed98385de1,"{""b"":""hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":6,""post_q"":71,""post_r"":6,""q"":449,""r"":15}"
   4573046,      1023,   2885887,   813.1,internal_correlated,1fdd7b56cdaa8e941d89ec03203b1a289a95066255f4c65930e2b1ee4c08bb65,"{""b"":""hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":6,""post_q"":449,""post_r"":1,""q"":449,""r"":15}"
   7794747,      1176,   5704279,  1220.7,internal_correlated,42e61298c26b48b25d3d440a691559d1526facc78692ca8b276b5ddec62e3fe5,"{""b"":""hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":7,""post_q"":97,""post_r"":2,""q"":449,""r"":15}"
  23410313,      1019,  20430657,  2794.4,internal_correlated,d7df7f77564004a3bb4217d750f9f9241b806c4db8160515f4485dc03b7caef8,"{""b"":""hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":7,""post_q"":97,""post_r"":3,""q"":449,""r"":15}"
  54868417,      1104,  51547916,  6111.2,internal_correlated,3e2c23f30871deecca6106848ae324762e776d27245c7cace6e5becb5f342724,"{""b"":""hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":7,""post_q"":97,""post_r"":4,""q"":449,""r"":15}"
 100000000,       942,  97120290, 10929.5,internal_correlated,237820b93ab5f644bb7ab7c9512ef10b7e527a0086f7cee2487fd5c427134aa0,"{""b"":""hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":7,""post_q"":97,""post_r"":5,""q"":449,""r"":15}"
 100000000,       453,  98628568, 10731.9,internal_correlated,a554bcb8f97995d494a9ab93e2166b32e13adba4f06df781abf4c67469ebd325,"{""b"":""hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":7,""post_q"":97,""post_r"":6,""q"":449,""r"":15}"
   7063325,      1004,   5260258,  1052.8,internal_correlated,1213490bcd28955444c132198020bab584301336b08d5266f665b092b8a31109,"{""b"":""hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":7,""post_q"":449,""post_r"":1,""q"":449,""r"":15}"
     41622,      1015,      4396,    32.4,internal_correlated,8aaa7ab622660fa281542eb5a8ab992e99af2360daa3e70624af9c0a57695567,"{""b"":""li_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":2,""post_q"":9,""post_r"":2,""q"":841,""r"":15}"
     45246,      1013,      7150,    33.1,internal_correlated,db827e0cec9fc24b08520e564d553d1ab4f53c3a885ccc055258a520277168f9,"{""b"":""li_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":2,""post_q"":9,""post_r"":3,""q"":841,""r"":15}"
     50059,      1010,     10875,    32.4,internal_correlated,3e62d12ef2889bb8146ff3ae6aaaf7a286c88adff44d3df11330d00505f896d7,"{""b"":""li_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":2,""post_q"":9,""post_r"":4,""q"":841,""r"":15}"
     48732,      1046,     13026,    32.8,internal_correlated,c5a284e985d6d4d4eb9d30dc95740e5392c41fe24868ff2bc50796c112508fdf,"{""b"":""li_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":2,""post_q"":9,""post_r"":5,""q"":841,""r"":15}"
     55208,      1039,     17206,    36.1,internal_correlated,24dc9822990579acfd6b6f6b4be606c852a3c8db6ec21de5fcb87623abf1448f,"{""b"":""li_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":2,""post_q"":9,""post_r"":6,""q"":841,""r"":15}"
     84000,      1003,      9155,    62.9,internal_correlated,5db644a92d4ce9676abf132607b74bb2606b06cf7eb9cbdd3af6cd6b25424bb3,"{""b"":""li_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":2,""post_q"":841,""post_r"":1,""q"":841,""r"":15}"
    198401,      1071,     56358,   124.8,internal_correlated,e154f9cf0d5e804e65c22f0f6ba2c3548062d8f269c0ed14914ce736f9642402,"{""b"":""li_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":3,""post_q"":25,""post_r"":2,""q"":841,""r"":15}"
    247200,      1006,    100855,   125.9,internal_correlated,06a6eebc0e8e46638a4d54eebfaa9a5c8bf38282bd0f5d1cac8a3ae5c6f01021,"{""b"":""li_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":3,""post_q"":25,""post_r"":3,""q"":841,""r"":15}"
    293661,      1010,    150281,   136.2,internal_correlated,d4e77382ba9784d46ca6cf7078b2646b11f6bcb85ec5cd8c1c787ee832aea8ab,"{""b"":""li_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":3,""post_q"":25,""post_r"":4,""q"":841,""r"":15}"
    403124,      1079,    240899,   152.0,internal_correlated,55ef87a05787b2744ffe93006341f40243b542e0edac7d304e4e5862d326028d,"{""b"":""li_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":3,""post_q"":25,""post_r"":5,""q"":841,""r"":15}"
    432000,      1088,    288351,   139.8,internal_correlated,89b390f470c706734b93ba4c54fa2e2a0774118158544981992944dca350a214,"{""b"":""li_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":3,""post_q"":25,""post_r"":6,""q"":841,""r"":15}"
    226940,      1011,     66313,   143.1,internal_correlated,0216810b67764ea898b3002cec542a4b092b3917b1941fb9b2b89ee6312c8f66,"{""b"":""li_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":3,""post_q"":841,""post_r"":1,""q"":841,""r"":15}"
    344111,      1014,    167848,   148.9,internal_correlated,683fdc501145848abc03395347fd5c10ca108fe11ae70328f7afd9fdff3049fe,"{""b"":""li_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":4,""post_q"":49,""post_r"":2,""q"":841,""r"":15}"
    565075,      1025,    368202,   219.8,internal_correlated,6d99e70140a5e4e375ef9d0a0d14e6d52a7bf7408a17bf6ac85dafff1233f1e1,"{""b"":""li_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":4,""post_q"":49,""post_r"":3,""q"":841,""r"":15}"
    845778,      1118,    644253,   252.9,internal_correlated,e941f08ccb104d9400909cc6ab5d74d26012999dc419dec859d12e7f4579aef3,"{""b"":""li_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":4,""post_q"":49,""post_r"":4,""q"":841,""r"":15}"
   1271463,      1119,   1064815,   321.2,internal_correlated,808e9489b3acf93742b07300d433bb19e5e521e20ee184115225ca6050c423a3,"{""b"":""li_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":4,""post_q"":49,""post_r"":5,""q"":841,""r"":15}"
   1689385,      1001,   1502247,   392.7,internal_correlated,8486c9904a0c891c479919c63b6fa26eea8262521a6fba7e7b156ca543e812c4,"{""b"":""li_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":4,""post_q"":49,""post_r"":6,""q"":841,""r"":15}"
    464914,      1200,    231389,   186.5,internal_correlated,8a9a7691e8716888240a1e2ff9bd793f6271a6d136fd95d02c5d1ff41da9bd16,"{""b"":""li_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":4,""post_q"":841,""post_r"":1,""q"":841,""r"":15}"
    576000,      1015,    388317,   204.1,internal_correlated,c1bf894e9c96b28688313010196038e7c9724dd4b23ecf1e60a0bbcb0bc71619,"{""b"":""li_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":5,""post_q"":81,""post_r"":2,""q"":841,""r"":15}"
   1296253,      1067,   1072947,   363.1,internal_correlated,34183f8fe62d8ab4a020e990f2d56c88ae0da9e876d658ee19263cbc155ca7aa,"{""b"":""li_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":5,""post_q"":81,""post_r"":3,""q"":841,""r"":15}"
   2350230,      1044,   2136583,   498.2,internal_correlated,a9f64c7c01e899e439e84e8263d3c7a55e1504c3c77f8548b64ab2ee7f26fed8,"{""b"":""li_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":5,""post_q"":81,""post_r"":4,""q"":841,""r"":15}"
   4169213,      1003,   3967663,   855.8,internal_correlated,cb15c033ba12e49db05ac90cdab273735ed0c1902c382ce138adc687b4478805,"{""b"":""li_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":5,""post_q"":81,""post_r"":5,""q"":841,""r"":15}"
   8443274,      1043,   8227227,  1732.3,internal_correlated,3313cae22bbf474d659e19ce17d4117ab895dfa111f1a1c0b26470a88759e365,"{""b"":""li_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":5,""post_q"":81,""post_r"":6,""q"":841,""r"":15}"
    604007,      1007,    411081,   188.9,internal_correlated,b40ff6b7f3a3c47e3e7a5222dfccef2f9372e81f4302763ebb164a4b44676d20,"{""b"":""li_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":5,""post_q"":841,""post_r"":1,""q"":841,""r"":15}"
   1080000,      1027,    879677,   281.6,internal_correlated,83b581be8caebf501fb01b54a7a79d0980842b5ced0d1d8e05754baba19c50a8,"{""b"":""li_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":6,""post_q"":121,""post_r"":2,""q"":841,""r"":15}"
   2896306,      1001,   2690569,   598.5,internal_correlated,5919ade5d70849942d5ba17796715f4227b2b4d3e5517403eb82838d15fa9313,"{""b"":""li_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":6,""post_q"":121,""post_r"":3,""q"":841,""r"":15}"
   8364156,      1100,   8135428,  1680.9,internal_correlated,45e1be1660ef7b8de637b93681e50b62fb9618f8ccd14004bffa844cd2dfb31b,"{""b"":""li_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":6,""post_q"":121,""post_r"":4,""q"":841,""r"":15}"
  22158573,      1146,  21925851,  4478.1,internal_correlated,28f6c3eb61a3cefdd1da2331cace490fc0fa9dcbf5f491cfd7804e8e02d9c8c2,"{""b"":""li_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":6,""post_q"":121,""post_r"":5,""q"":841,""r"":15}"
  49568284,      1008,  49367352,  9967.1,internal_correlated,36abd905ba46d0c6c059d815f1a88f723dbd81ea02493864c47bf09c41be809d,"{""b"":""li_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":6,""post_q"":121,""post_r"":6,""q"":841,""r"":15}"
   1168307,      1057,    957663,   296.5,internal_correlated,31c2e09b207120b0730dfa59df2ce332d8ac4959a1d41265eda4c32bb4c8e265,"{""b"":""li_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":6,""post_q"":841,""post_r"":1,""q"":841,""r"":15}"
   2116839,      1005,   1917643,   490.4,internal_correlated,bb6b8f8b9b334b99ca857cbda612a1721c5f4a25d2ca6a5e3648056a93fade7c,"{""b"":""li_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":7,""post_q"":169,""post_r"":2,""q"":841,""r"":15}"
   8638171,      1034,   8425685,  1646.9,internal_correlated,0398c4015aa4a7ebf404a5bbafddc3f31553c21d2fa8c884402ab889f77e2184,"{""b"":""li_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":7,""post_q"":169,""post_r"":3,""q"":841,""r"":15}"
  36529967,      1065,  36295299,  7249.6,internal_correlated,ad2d2b2c91be1a11eb615d87e288c69eb2b5c4f289f0b4169c8e7b6bae1c837e,"{""b"":""li_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":7,""post_q"":169,""post_r"":4,""q"":841,""r"":15}"
 100000000,       779,  99831046, 20406.2,internal_correlated,3dffcfb96001566d5d2d0001766dc9fbde8c10d2a52a0c0287bd543d7f9ce4e9,"{""b"":""li_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":7,""post_q"":169,""post_r"":5,""q"":841,""r"":15}"
 100000000,       193,  99955500, 20564.0,internal_correlated,c5114e7d5fe98b79d8d809d3c9ffba33eb69dd481212c1cfebdbd1707fc7c51f,"{""b"":""li_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":7,""post_q"":169,""post_r"":6,""q"":841,""r"":15}"
   2280507,      1030,   2073434,   500.3,internal_correlated,14674d8530fd0a7635a432f7cbde5801a4876f24b5670e38a246478ec8d62a65,"{""b"":""li_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":7,""post_q"":841,""post_r"":1,""q"":841,""r"":15}"
    793367,      1003,    133391,   231.0,internal_correlated,7ca1feb20d52ea18c6142bdc15a500e7385f8c0d2c1a2100e18eb700cce7ab7d,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":2,""post_q"":449,""post_r"":1,""q"":449,""r"":15}"
    793881,      1001,    133295,   274.9,internal_correlated,7bf79e38c34fc1c0bc27487060a820ce59f3b723529abef992f9c6e0aebfab76,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":2,""post_q"":449,""post_r"":2,""q"":449,""r"":15}"
   1151839,      1001,    293575,   359.8,internal_correlated,ba79cf1e4a655422869f3e4b1cfff78479719aa3c23b0dbf9e9689e17796aa62,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":2,""post_q"":449,""post_r"":3,""q"":449,""r"":15}"
   1473801,      1099,    490457,   416.3,internal_correlated,6dec56f0e58459edba547ae4ab60b27b7aba3f7e8d47d730b3c52c4183c14296,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":2,""post_q"":449,""post_r"":4,""q"":449,""r"":15}"
   1464117,      1013,    587957,   483.8,internal_correlated,ee32b85ba07ed787c4c66b179dd990adba4546fadecedb5f341dc50d7878231f,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":2,""post_q"":449,""post_r"":5,""q"":449,""r"":15}"
   1781180,      1065,    826878,   547.1,internal_correlated,5040161f934646c3b15e9c6decf14c0ae0684baea034a4589b4da386b856905d,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":2,""post_q"":449,""post_r"":6,""q"":449,""r"":15}"
   1789166,      1012,    538075,   497.6,internal_correlated,ee2e54cf65c3b43b127f8cf5e008c406fadf960bb36b0393ea82b559b1fe67be,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":3,""post_q"":449,""post_r"":1,""q"":449,""r"":15}"
   1675183,      1008,    504696,   471.5,internal_correlated,0cf8b70b1825c6accd83a4842acbd758f1546f052f3ca72849506a3dc80eb61d,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":3,""post_q"":449,""post_r"":2,""q"":449,""r"":15}"
   3298712,      1000,   1433368,   886.2,internal_correlated,0e37365016c9bb9527861c3191aa522be84c9ed3bb760b3ae85a20469d0837bf,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":3,""post_q"":449,""post_r"":3,""q"":449,""r"":15}"
   4496746,      1007,   2438737,  1171.7,internal_correlated,8d1c42b6895e97e33e9ed0446b636c32e25cd1fa2d270027ed0504a02d581c59,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":3,""post_q"":449,""post_r"":4,""q"":449,""r"":15}"
   5828385,      1080,   3668842,  1382.0,internal_correlated,755d85b2e2440b9869430a885d8c64b70794aa8d2a81bc10652fdbcd87d57317,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":3,""post_q"":449,""post_r"":5,""q"":449,""r"":15}"
   6886825,      1015,   4821776,  1586.2,internal_correlated,ad89a1b98490706ef5e516c47c9262c5b91a7c762290e09e9b27b4c52b401679,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":3,""post_q"":449,""post_r"":6,""q"":449,""r"":15}"
   2928812,      1013,   1311928,   694.9,internal_correlated,ee678ceb6c2fc8d9e85832ee4887c384d59ffd6d1ca3075f1682f1ab3de3b65b,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":4,""post_q"":449,""post_r"":1,""q"":449,""r"":15}"
   3236575,      1098,   1446912,   792.0,internal_correlated,8d6108171dc637442122dc3297020f1b6a2426830033b86d7e028e0e3f61b482,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":4,""post_q"":449,""post_r"":2,""q"":449,""r"":15}"
   7670221,      1046,   4673162,  1807.1,internal_correlated,18a75c2920e861eadb4e9b2fae3d99aaf791799bec2934e3c808193c1c442288,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":4,""post_q"":449,""post_r"":3,""q"":449,""r"":15}"
  10742184,      1015,   7763882,  2212.4,internal_correlated,7d17f9efdeb5586c9944ef741955a78341d8a3512e0bcd1b84c1c4ef494a5850,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":4,""post_q"":449,""post_r"":4,""q"":449,""r"":15}"
  14835707,      1038,  11924214,  2683.2,internal_correlated,5ded96bf55912dcc6d74b510dce814dec2d9ee589e9fa5318f7c32c285adbcc5,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":4,""post_q"":449,""post_r"":5,""q"":449,""r"":15}"
  22677872,      1045,  19524216,  4128.8,internal_correlated,e7dcc6f367d45c7053612fd45dbdae3e7dd5760538fd00dea5c56ef9b3670499,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":4,""post_q"":449,""post_r"":6,""q"":449,""r"":15}"
   4202509,      1003,   2458709,   826.7,internal_correlated,ee556c261936e66d272e156a10a7d8e52395a331825f82384f0df9be6b32aa40,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":5,""post_q"":449,""post_r"":1,""q"":449,""r"":15}"
   4342102,      1029,   2542949,   904.6,internal_correlated,42e55a80f20d14375a072e81b58e341d24565e3437d2bc51e24a4aa2fed207b9,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":5,""post_q"":449,""post_r"":2,""q"":449,""r"":15}"
  13392277,      1022,  10059850,  2442.0,internal_correlated,dead0ae6ca83e2e42afe67bafba075dad7d074a079c72ad34f8505e2f5575736,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":5,""post_q"":449,""post_r"":3,""q"":449,""r"":15}"
  22882400,      1024,  19464597,  3857.9,internal_correlated,84e63a832b02828d1d5dd08d36890d47bade8387bb0f95b14a128b913ad6e2ec,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":5,""post_q"":449,""post_r"":4,""q"":449,""r"":15}"
  41144151,      1083,  37448223,  6483.3,internal_correlated,0167cde773a6a5120c63af6d02d89f9fba36f9bfaa8403a78baf691a75875925,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":5,""post_q"":449,""post_r"":5,""q"":449,""r"":15}"
  63466581,      1040,  60042446,  9633.3,internal_correlated,0a35a4d459d0f4bc826b2f302e35295776b39d35f32a9dd8a064f0848ed9ac7d,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":5,""post_q"":449,""post_r"":6,""q"":449,""r"":15}"
   6676682,      1077,   4722425,  1100.0,internal_correlated,12511bbfbbefeba94ac2a215ed4e077c0ea2cceb442d9d287f5f854921bc8c00,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":6,""post_q"":449,""post_r"":1,""q"":449,""r"":15}"
   6735176,      1107,   4762200,  1210.8,internal_correlated,8c24e6c506086016fa1915b3ca3667f721b7289f36e75246f890246060edcc9b,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":6,""post_q"":449,""post_r"":2,""q"":449,""r"":15}"
  23847937,      1019,  20407890,  3636.7,internal_correlated,9876af68fbca31a2969710ea46f1149b20bec6946fb233e6216e7f2673eda3c6,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":6,""post_q"":449,""post_r"":3,""q"":449,""r"":15}"
  50663821,      1059,  47055389,  7292.9,internal_correlated,a5e4b6306ff4b9199af1ab29a4a5c9dd71c7db77a3ee7d9405ae0c16b68d027a,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":6,""post_q"":449,""post_r"":4,""q"":449,""r"":15}"
  99910409,      1017,  96398748, 14240.8,internal_correlated,d00543fc41e3686500332bc8cc907a7567aaa658e50eca86b16baee8d481d71c,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":6,""post_q"":449,""post_r"":5,""q"":449,""r"":15}"
 100000000,       524,  98262830, 14348.6,internal_correlated,a753cb45b8818ff4432524d49c246aa5f9fb82dc2fa13fb87f308f900c13e5e6,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":6,""post_q"":449,""post_r"":6,""q"":449,""r"":15}"
   9354957,      1010,   7517109,  1347.3,internal_correlated,7f9b08dfd5f51449dd4ec0cc215afec2c7b8412c03ed7fd339ba6a47924502e1,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":7,""post_q"":449,""post_r"":1,""q"":449,""r"":15}"
  10608907,      1120,   8525874,  1639.7,internal_correlated,e765d829b0b8663474348a42503965628809e56e02d78d6aa95a04325d55ccad,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":7,""post_q"":449,""post_r"":2,""q"":449,""r"":15}"
  48646309,      1083,  44905309,  6227.2,internal_correlated,a2dbd3c04578dbf7c10ac4f6c90412399b709c32d3191757bf621531db49cf86,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":7,""post_q"":449,""post_r"":3,""q"":449,""r"":15}"
 100000000,       927,  96975762, 12949.4,internal_correlated,6a7dd95e78cd9bebeb38838a12af02a9e0a4ebb9cae390faf1d0c385759a26dc,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":7,""post_q"":449,""post_r"":4,""q"":449,""r"":15}"
 100000000,       336,  98812599, 12839.1,internal_correlated,12aebeb5bb008c1c7b2e71ac37524a873541719b952d63d5802da42554265c31,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":7,""post_q"":449,""post_r"":5,""q"":449,""r"":15}"
 100000000,       132,  99533708, 13807.9,internal_correlated,d95769a2352ee2eff224be783f946b6554131b4fcbc33ce589e3937c5649398b,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":7,""post_q"":449,""post_r"":6,""q"":449,""r"":15}"
  14991420,      1009,  13135321,  1859.2,internal_correlated,5189b4ef7ec3cb81c89c6ce9708ee5850c5719b5f2df707274ca796903290fb3,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":8,""post_q"":449,""post_r"":1,""q"":449,""r"":15}"
  15211359,      1019,  13329110,  2005.8,internal_correlated,0fe40cc6993258c6c0e8ab6790077337d9247bc352d858f96deaccc790f98228,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":8,""post_q"":449,""post_r"":2,""q"":449,""r"":15}"
  95058564,      1023,  91500373, 11545.8,internal_correlated,ebf5c1888b0a5c44b91a272070afb4e0d6ed6776ec8a967938201523dabb26fc,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":8,""post_q"":449,""post_r"":3,""q"":449,""r"":15}"
 100000000,       339,  98865349, 12932.8,internal_correlated,883a4783b1d36756760bea35c77a88a7ec3e7228c35bef75a38ff2325cf85dbd,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":8,""post_q"":449,""post_r"":4,""q"":449,""r"":15}"
 100000000,        97,  99655746, 13589.1,internal_correlated,19a5b02523748227fd99c12ce45e2cb3ba8f5a01f52842c6c0f6c599cb310139,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":8,""post_q"":449,""post_r"":5,""q"":449,""r"":15}"
 100000000,        21,  99895332,  9090.8,internal_correlated,335e251456aa13d69d26c23c449f0f04f7d6a2842c768f13b32c285eefd863ae,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":8,""post_q"":449,""post_r"":6,""q"":449,""r"":15}"
  23978702,      1010,  22202151,  2705.5,internal_correlated,a7246882616cd8f2806afa8e007a868f55c790fcfc6172f62c1c0d181e5b73bd,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":9,""post_q"":449,""post_r"":1,""q"":449,""r"":15}"
  25764529,      1035,  23853936,  3105.2,internal_correlated,faa8f45569ac8e90f112cca643be7b1eb12f25b09c39351e09017752be5765c0,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":9,""post_q"":449,""post_r"":2,""q"":449,""r"":15}"
 100000000,       476,  98327969, 12069.1,internal_correlated,733df43d69dcf8f1c9d5df8e8eb3ace4ae41526b4817f7dc6b62df1cad3d2d35,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":9,""post_q"":449,""post_r"":3,""q"":449,""r"":15}"
 100000000,        95,  99620579, 12495.1,internal_correlated,5d672852cc8b8b43a7335ae938c744d9b3126e1668c32ee75d1c008c215bd27a,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":9,""post_q"":449,""post_r"":4,""q"":449,""r"":15}"
 100000000,        22,  99914476, 13827.6,internal_correlated,6566eb419afdef4a82d4090c3a636d0983193d7a450d8091bfd0874bee246994,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":9,""post_q"":449,""post_r"":5,""q"":449,""r"":15}"
 100000000,        11,  99980660, 12985.1,internal_correlated,1ee620f169a704aa5e5ccbddf6bf082787f94e406b0b0ce34818aa41a9a9f2da,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":9,""post_q"":449,""post_r"":6,""q"":449,""r"":15}"
  44911205,      1033,  43035143,  5326.4,internal_correlated,bbfe19fa6a021652e6e1d3c75370613d51515d4342fcda5c398f6c031a277942,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":10,""post_q"":449,""post_r"":1,""q"":449,""r"":15}"
  47129778,      1090,  45158839,  5420.5,internal_correlated,4ca8fff00862f7679a5825b17cce51fb0764d1386410e4763d0e9bc82caedc8f,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":10,""post_q"":449,""post_r"":2,""q"":449,""r"":15}"
 100000000,       202,  99317606, 11901.1,internal_correlated,1f5e99c7b209350ea50f2e0f3ac91fcc860f2cbe0d28d8e1de6cbd60ec184e77,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":10,""post_q"":449,""post_r"":3,""q"":449,""r"":15}"
 100000000,        44,  99888179, 12473.0,internal_correlated,466389343475680c4c40478b434a5269213d4337443f2f4b3d580ca3283c5f32,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":10,""post_q"":449,""post_r"":4,""q"":449,""r"":15}"
 100000000,         6,  99981759, 13513.8,internal_correlated,a2c934e3c552d73419663023c85b4bdde9adc5d6b2e320a856b3fd0afa26a6a3,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":10,""post_q"":449,""post_r"":5,""q"":449,""r"":15}"
 100000000,         1,  99996984, 13899.7,internal_correlated,f538186a9a73717e1eca7e04cd046a354fafc272907d601db938bbddbed52f0a,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":10,""post_q"":449,""post_r"":6,""q"":449,""r"":15}"
  80492766,      1014,  78691383,  8621.5,internal_correlated,0f94ba95d42736d59e0bc268b5814d4e3160c166d1221b927d575e2f722e5ef8,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":11,""post_q"":449,""post_r"":1,""q"":449,""r"":15}"
  90840303,      1107,  88810322, 10155.1,internal_correlated,b6fe0bf467372aeb91ee220c7e966c7bb61d791f80bb821e1c4d52d2fb99a4d7,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":11,""post_q"":449,""post_r"":2,""q"":449,""r"":15}"
 100000000,        70,  99743657, 11928.1,internal_correlated,4e951184ffd56677a2cbcd817207d3250067ab9b408543da78f7ec24912de216,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":11,""post_q"":449,""post_r"":3,""q"":449,""r"":15}"
 100000000,         9,  99970633, 12713.8,internal_correlated,7c842eefa108f1be00ae9b52c1be38b9befb195e86c62df89def271d97896570,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":11,""post_q"":449,""post_r"":4,""q"":449,""r"":15}"
 100000000,         0,  99996694, 13220.8,internal_correlated,d6ac0adcc67992fbee156caf98b6dbfe77fad880dde7290e3380ba7d8439bf72,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":11,""post_q"":449,""post_r"":5,""q"":449,""r"":15}"
 100000000,         0,  99999564, 13861.6,internal_correlated,fc18872288c557840220a4b00d4219960e0c71c72c271f12b1cac336f9cb38de,"{""b"":""pregrown_hook_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":11,""post_q"":449,""post_r"":6,""q"":449,""r"":15}"
     14223,      1038,       464,    13.6,internal_correlated,a360d8ece3dd834c20e1f1a0a6f6d58e8922de570c8f2c531417a7cf12dcfaf1,"{""b"":""zz_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":2,""post_q"":9,""post_r"":1,""q"":841,""r"":15}"
     15816,      1024,      1683,    15.9,internal_correlated,edd9cd3da37c84218e4fa7211470fe46c6ae2e2a7eaaf35b29107789e058e748,"{""b"":""zz_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":2,""post_q"":9,""post_r"":2,""q"":841,""r"":15}"
     16380,      1006,      2710,    16.8,internal_correlated,53a7cf0350c732052019327c7378134b34a9039b61c8f0439c19039a1d751ad4,"{""b"":""zz_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":2,""post_q"":9,""post_r"":3,""q"":841,""r"":15}"
     18670,      1053,      4024,    13.3,internal_correlated,79f97b933a10d5021ccf6ace036cae9ee265907b9fc3ef3e86346d3790c0c0e4,"{""b"":""zz_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":2,""post_q"":9,""post_r"":4,""q"":841,""r"":15}"
     18990,      1014,      5142,    13.5,internal_correlated,1e48cda5b4ae40c53a33b43763d288e3bc5f07230a3e762ffd0da13afcbee1fa,"{""b"":""zz_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":2,""post_q"":9,""post_r"":5,""q"":841,""r"":15}"
     21497,      1071,      6663,    15.3,internal_correlated,75152b2f084968c5aac28ff9f5b236239e09e9075804a637fb29caa96b16c944,"{""b"":""zz_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":2,""post_q"":9,""post_r"":6,""q"":841,""r"":15}"
     42192,      1059,      3331,    28.7,internal_correlated,e230fbfa562cc95e8968a17a38e7a33f16801c995ede0ac38bd647922a231b7c,"{""b"":""zz_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":3,""post_q"":25,""post_r"":1,""q"":841,""r"":15}"
    191987,      1004,     54460,    80.8,internal_correlated,be0a3beca6cd9a96e904d3ccba6da205bac8231cb3eecd0e5fb16f053f4cfa72,"{""b"":""zz_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":3,""post_q"":25,""post_r"":2,""q"":841,""r"":15}"
    233710,      1007,     95424,    76.0,internal_correlated,38e5eb7e68ede0bd9103ca7cd0ad9c4924d499d02139ec8d5870fc3a64c101e9,"{""b"":""zz_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":3,""post_q"":25,""post_r"":3,""q"":841,""r"":15}"
    289132,      1000,    146782,    78.9,internal_correlated,2c6094633aba4def835496aa9f63ed76e97fc3f0c121d22f29007d3c80d6c93f,"{""b"":""zz_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":3,""post_q"":25,""post_r"":4,""q"":841,""r"":15}"
    373879,      1082,    221676,   141.1,internal_correlated,791ae929325eb7c988eb84563a4f8536e9a55812bac66ed2039daeb7630d1b1a,"{""b"":""zz_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":3,""post_q"":25,""post_r"":5,""q"":841,""r"":15}"
    444305,      1032,    294546,   161.7,internal_correlated,42832260f26af27072b7bcde5c16ac3f8368a001bdea5185907d3a34637f25f4,"{""b"":""zz_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":3,""post_q"":25,""post_r"":6,""q"":841,""r"":15}"
     50069,      1013,      7637,    32.0,internal_correlated,ace8228ec4d74cae43058e347c79c48a05886a88c7d3abfb65fbf048a80441d9,"{""b"":""zz_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":4,""post_q"":49,""post_r"":1,""q"":841,""r"":15}"
    721670,      1037,    352185,   204.3,internal_correlated,cfba3e06c9ba00712f236b68cee878179ca2b7d2a98bb83ba3af9be0b0f9e69b,"{""b"":""zz_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":4,""post_q"":49,""post_r"":2,""q"":841,""r"":15}"
   1078389,      1018,    698657,   208.2,internal_correlated,f49b0dd432423423dd3bbc58d82aed248cc8933558b1bd088753d9d80ebb81be,"{""b"":""zz_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":4,""post_q"":49,""post_r"":3,""q"":841,""r"":15}"
   1785402,      1076,   1353800,   277.1,internal_correlated,85549e5264ab96b5961d68b9e25ce497dea8e14c9ff0280ae7b520ba1a0f432d,"{""b"":""zz_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":4,""post_q"":49,""post_r"":4,""q"":841,""r"":15}"
   2427893,      1008,   2024559,   603.7,internal_correlated,7e1625523a823682e4fb6248d5cca887fc6dbe23b48179d1306c14a433b0300d,"{""b"":""zz_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":4,""post_q"":49,""post_r"":5,""q"":841,""r"":15}"
   3446413,      1028,   3052677,   794.3,internal_correlated,25dd29c43d819a7f3e6531d0966fc8c029403b3e3a5b6541c1c6970615ef308c,"{""b"":""zz_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":4,""post_q"":49,""post_r"":6,""q"":841,""r"":15}"
     57806,      1012,     14324,    36.4,internal_correlated,d473d8eb5d886b491b48f60e91b931a2e84e3392a6d2d0f970c98234447d40ee,"{""b"":""zz_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":5,""post_q"":81,""post_r"":1,""q"":841,""r"":15}"
   1740475,      1003,   1172144,   315.4,internal_correlated,8319f871c3359d6e5179ab4d102a1910263d45de36512b1dfda944e8840e3ed0,"{""b"":""zz_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":5,""post_q"":81,""post_r"":2,""q"":841,""r"":15}"
   4020777,      1010,   3320177,   547.0,internal_correlated,412bd70245b787d9b515b63db757879eb8f77be8f5aa017a9c599d253fc10442,"{""b"":""zz_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":5,""post_q"":81,""post_r"":3,""q"":841,""r"":15}"
   7473070,      1007,   6773772,  1022.9,internal_correlated,c9839febcf1b4d91aa389e20e9850f6fe05043b1509cf2d66afef3a0636de016,"{""b"":""zz_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":5,""post_q"":81,""post_r"":4,""q"":841,""r"":15}"
  14440724,      1059,  13717149,  2918.1,internal_correlated,9fd829fbdbff35f89fff54e6c9f51e926371606f4e05be20b53791fa2a9c1715,"{""b"":""zz_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":5,""post_q"":81,""post_r"":5,""q"":841,""r"":15}"
  27223913,      1098,  26491298,  5385.1,internal_correlated,aececd02b653a3e7603e50771fb6c43e2ff1b55a9d71e741bda57011b83bc25d,"{""b"":""zz_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":5,""post_q"":81,""post_r"":6,""q"":841,""r"":15}"
     66423,      1022,     23436,    32.2,internal_correlated,62c60254c732347f024f5e52c5b85ec51e3485abbc23adb878451e9d52fa7087,"{""b"":""zz_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":6,""post_q"":121,""post_r"":1,""q"":841,""r"":15}"
   3697606,      1137,   3011623,   510.8,internal_correlated,fa28e8e572b9f08527ac4947069fea5dd84ed875a79d9fcbd5744cc9e3f35749,"{""b"":""zz_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":6,""post_q"":121,""post_r"":2,""q"":841,""r"":15}"
  12031573,      1074,  11158776,  1225.1,internal_correlated,dbff8aeed4f5a8634d99e4aaed11a77117289fb5a39c9135b89160ffacf538a1,"{""b"":""zz_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":6,""post_q"":121,""post_r"":3,""q"":841,""r"":15}"
  30091579,      1094,  29236130,  3682.8,internal_correlated,1f5be4c002cc82b1941316dae5882fb0aea712bc1b3a40e7b661fd1b853d34ed,"{""b"":""zz_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":6,""post_q"":121,""post_r"":4,""q"":841,""r"":15}"
  69556794,      1021,  68780550, 13197.8,internal_correlated,d1dd33b280d5371b5cf2191a95e7dd4ccf406b3a98d81b0cb38a50f6b15a063d,"{""b"":""zz_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":6,""post_q"":121,""post_r"":5,""q"":841,""r"":15}"
 100000000,       574,  99561021, 18843.0,internal_correlated,5fe53100f526649d0235a55125843531528b74dfdc63db471537adf62bb653c4,"{""b"":""zz_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":6,""post_q"":121,""post_r"":6,""q"":841,""r"":15}"
     90520,      1103,     41660,    40.0,internal_correlated,f34d4dd49ef18b3bb37d2a56555390fbbce38eb03a193dda384f8f2ca3682c35,"{""b"":""zz_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":7,""post_q"":169,""post_r"":1,""q"":841,""r"":15}"
   6579684,      1035,   5960705,  1153.4,internal_correlated,7fd64d1b3be7920ed30d0282336db4e8835fd4858fdfd706b65caa5c1f918b45,"{""b"":""zz_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":7,""post_q"":169,""post_r"":2,""q"":841,""r"":15}"
  36259997,      1109,  35342812,  3709.2,internal_correlated,5e3ee874844023b0597faddc253feb76d0c4535b8139cfed40aa022df43b11f4,"{""b"":""zz_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":7,""post_q"":169,""post_r"":3,""q"":841,""r"":15}"
 100000000,       871,  99317826, 20948.9,internal_correlated,263f6fe32ca7cd24338f7c6d10e50e677aeb3cb2d901b57e66910edf3a7f6f1e,"{""b"":""zz_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":7,""post_q"":169,""post_r"":4,""q"":841,""r"":15}"
 100000000,       214,  99816891, 18957.9,internal_correlated,0ef74bae7e85cfca69db5b1f359cd503ac2353ff9a363c1aa6d335fdc5b499d5,"{""b"":""zz_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":7,""post_q"":169,""post_r"":5,""q"":841,""r"":15}"
 100000000,        59,  99950691, 19400.8,internal_correlated,782fdbfd562b93d8d6506ea5dcc0542e8f00c907db43562672e6b3e53f3c7206,"{""b"":""zz_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":7,""post_q"":169,""post_r"":6,""q"":841,""r"":15}"
     46619,      1044,      1422,    33.2,internal_correlated,5581f9fac1b60ed5b9cfb56f78e0de3f15602c268a5f30c81b881e6bc51dfa3e,"{""b"":""zz_tweaked_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":2,""post_q"":9,""post_r"":1,""q"":841,""r"":15}"
     56012,      1001,      6157,    44.1,internal_correlated,1943140b82a937d7e437614d5c1183ceb4d83ca4dd418002e78588426e8fb7ed,"{""b"":""zz_tweaked_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":2,""post_q"":9,""post_r"":2,""q"":841,""r"":15}"
     56760,      1001,      9274,    40.2,internal_correlated,d86b14e457f4984206a127046e67555aaf257a7eaafa74f3c4e0e50b8c74e0e2,"{""b"":""zz_tweaked_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":2,""post_q"":9,""post_r"":3,""q"":841,""r"":15}"
     60412,      1001,     13078,    46.6,internal_correlated,be8948cd2aa2f700b10d2e50c7099bc53117b212dce11ab8553d3f1554b3356e,"{""b"":""zz_tweaked_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":2,""post_q"":9,""post_r"":4,""q"":841,""r"":15}"
     68758,      1016,     18433,    41.7,internal_correlated,d2f56c0d5fae3954c77f442c2db900e11066da105210b278fee97de32bd95014,"{""b"":""zz_tweaked_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":2,""post_q"":9,""post_r"":5,""q"":841,""r"":15}"
     70988,      1030,     22099,    44.4,internal_correlated,01e118e7eaf02ce9034455b0e6e19b6d2801226f4fa87c46d36c3fe25b18cb44,"{""b"":""zz_tweaked_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":2,""post_q"":9,""post_r"":6,""q"":841,""r"":15}"
     57589,      1007,      4638,    40.6,internal_correlated,368bb95e369c199479fc44b851b5c5e0a4b0d380980176586890dbb3e3bd0d42,"{""b"":""zz_tweaked_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":3,""post_q"":25,""post_r"":1,""q"":841,""r"":15}"
    564928,      1048,    160360,   263.0,internal_correlated,25a83a8f4c4858be85a452bf3098b597511de812db4132697295b1c3b82b4a68,"{""b"":""zz_tweaked_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":3,""post_q"":25,""post_r"":2,""q"":841,""r"":15}"
    700425,      1033,    284987,   321.0,internal_correlated,47240509ea5323df7a0353a5c8ad71fdd779eb5491cb8c251f1cd7afd092434b,"{""b"":""zz_tweaked_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":3,""post_q"":25,""post_r"":3,""q"":841,""r"":15}"
    799428,      1016,    405714,   319.7,internal_correlated,ef7ae4f8607efc2d4c809bb87c6a6adea0c41b2a362b8cea89e02c3c10cb8022,"{""b"":""zz_tweaked_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":3,""post_q"":25,""post_r"":4,""q"":841,""r"":15}"
    970796,      1016,    575227,   343.5,internal_correlated,fb5fd45441e947f3351f86d844f26fc16f0501bd0e06cbd9ff5daeea6980a590,"{""b"":""zz_tweaked_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":3,""post_q"":25,""post_r"":5,""q"":841,""r"":15}"
   1146220,      1005,    759786,   389.6,internal_correlated,c1debc92bfbf871e91e1d3f3d4e9d134c00a0ec54c4cd479008ead74c4171c17,"{""b"":""zz_tweaked_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":3,""post_q"":25,""post_r"":6,""q"":841,""r"":15}"
     65048,      1080,     10006,    49.7,internal_correlated,5c05830188567369edfa7c49f2b081bce5597d0620260c71dc3df81fb4e55e2d,"{""b"":""zz_tweaked_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":4,""post_q"":49,""post_r"":1,""q"":841,""r"":15}"
   1611310,      1096,    784508,   648.0,internal_correlated,0104d1b435548719ff6f3591212a6a31a4854920eb38ac16518d1b10154fec6e,"{""b"":""zz_tweaked_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":4,""post_q"":49,""post_r"":2,""q"":841,""r"":15}"
   2542910,      1015,   1646912,   805.9,internal_correlated,06599f89ac2ab558f53a86a73c25b7eb76dc80baf8bee8999725499203ea963d,"{""b"":""zz_tweaked_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":4,""post_q"":49,""post_r"":3,""q"":841,""r"":15}"
   4131989,      1111,   3131326,  1127.6,internal_correlated,04cf9cb3b992889d4fb83d7d095b55086fbae97cb4991ac655021ed6d55081ba,"{""b"":""zz_tweaked_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":4,""post_q"":49,""post_r"":4,""q"":841,""r"":15}"
   5946585,      1130,   4955087,  1431.3,internal_correlated,b4762df1027093f318c87f8f36fdb16ffcfbd47e9c1e8fd07a2d6077314bdcf4,"{""b"":""zz_tweaked_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":4,""post_q"":49,""post_r"":5,""q"":841,""r"":15}"
   7661881,      1005,   6784745,  1700.1,internal_correlated,e8053dc555b3490e0df7d1fd63b6df02f1d66777f42076e83eef26b83914e1d8,"{""b"":""zz_tweaked_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":4,""post_q"":49,""post_r"":6,""q"":841,""r"":15}"
     78454,      1101,     19246,    51.3,internal_correlated,ee3a8c52517310cbf303d4aec3f092ba8c57b888f24da1459907e8c0390db03f,"{""b"":""zz_tweaked_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":5,""post_q"":81,""post_r"":1,""q"":841,""r"":15}"
   3302526,      1004,   2218973,  1072.0,internal_correlated,4054b2634aa9263ef7be4fa85d8088e095c4595ccf27f52e8d3115c35d00465e,"{""b"":""zz_tweaked_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":5,""post_q"":81,""post_r"":2,""q"":841,""r"":15}"
   8566418,      1068,   7065107,  2099.3,internal_correlated,debce26f4be2e200fc8d8bbf4385db839b887a3a0b725265bc838b5260afa6f2,"{""b"":""zz_tweaked_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":5,""post_q"":81,""post_r"":3,""q"":841,""r"":15}"
  15096950,      1049,  13675752,  3004.8,internal_correlated,485ef8c42f3f1183c1fc7e9605eb97e73f39d27a60e6fe0af62ed7c95051f503,"{""b"":""zz_tweaked_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":5,""post_q"":81,""post_r"":4,""q"":841,""r"":15}"
  28523434,      1042,  27086742,  5889.0,internal_correlated,ec99e39675c505f477752db60b0911c3fbe7469efde6af0250457be09b36a6c0,"{""b"":""zz_tweaked_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":5,""post_q"":81,""post_r"":5,""q"":841,""r"":15}"
  54778764,      1031,  53299559, 10070.6,internal_correlated,67076b290092684f385a3c9149eeb8f0f91bed9ba58450abcc210c46cdf550a0,"{""b"":""zz_tweaked_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":5,""post_q"":81,""post_r"":6,""q"":841,""r"":15}"
     83481,      1002,     29316,    56.1,internal_correlated,5f9038542154130ca063ebef72affd0c297827593f0af1f69a7726c156547b2b,"{""b"":""zz_tweaked_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":6,""post_q"":121,""post_r"":1,""q"":841,""r"":15}"
   6700800,      1085,   5447981,  1594.1,internal_correlated,3822a9c799540ee9f68d598049228b81a074d3dda5f836130554190d40c48f6b,"{""b"":""zz_tweaked_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":6,""post_q"":121,""post_r"":2,""q"":841,""r"":15}"
  22725833,      1009,  21063614,  4456.7,internal_correlated,1c4513e58f728aed973b66c0b7e56ce3bf915186372babacc73d70c929236b14,"{""b"":""zz_tweaked_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":6,""post_q"":121,""post_r"":3,""q"":841,""r"":15}"
  66555163,      1082,  64647257, 12621.0,internal_correlated,0069e10b1b2f77680338ccd356d3d713473bf0cb44fddf329efb811f7baca697,"{""b"":""zz_tweaked_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":6,""post_q"":121,""post_r"":4,""q"":841,""r"":15}"
 100000000,       673,  98877943, 18573.2,internal_correlated,5da94a967ea5004025f02d67d38bf43f48a2697cdad160e293687596e4496d57,"{""b"":""zz_tweaked_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":6,""post_q"":121,""post_r"":5,""q"":841,""r"":15}"
 100000000,       280,  99560562, 17103.6,internal_correlated,79e320906c7e2f1c8536f0d5f00984d3e7b5068945654aad7f7e1d9ff8f6f6a1,"{""b"":""zz_tweaked_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":6,""post_q"":121,""post_r"":6,""q"":841,""r"":15}"
     96704,      1009,     44386,    48.4,internal_correlated,3a404ca7a89770a2135087082b08e7dcd53bbbc40c2a51567e1c2bd8784adfe2,"{""b"":""zz_tweaked_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":7,""post_q"":169,""post_r"":1,""q"":841,""r"":15}"
  14177300,      1127,  12827512,  2779.2,internal_correlated,4f067315d8f7a8ac5b5bc039135b9ce31584397c20a5309caa2ccd132db6bed5,"{""b"":""zz_tweaked_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":7,""post_q"":169,""post_r"":2,""q"":841,""r"":15}"
  66424187,      1016,  64728066, 12029.8,internal_correlated,d5a27148fb250aac7fa7b55b39af9211920162570a64632b45674d1abc0a9659,"{""b"":""zz_tweaked_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":7,""post_q"":169,""post_r"":3,""q"":841,""r"":15}"
 100000000,       382,  99313884, 18613.5,internal_correlated,2bbf8d63209e4edc1e9a10eab8bdcc798c93eddabf8c404677f81267d91cb9d9,"{""b"":""zz_tweaked_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":7,""post_q"":169,""post_r"":4,""q"":841,""r"":15}"
 100000000,       108,  99814589, 19048.9,internal_correlated,65906963863edeaa1626e611f6867ea6595725e4d5b859088663c4e1d34fdced,"{""b"":""zz_tweaked_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":7,""post_q"":169,""post_r"":5,""q"":841,""r"":15}"
 100000000,        41,  99950259, 13631.1,internal_correlated,b4a5805492c2eabd9df7a7561b6a9291c8fbf8df0893bb5d8826295199b87468,"{""b"":""zz_tweaked_inject_Y_magic_verify"",""d"":15,""gates"":""cz"",""noise"":""SI1000"",""p"":0.001,""post_d"":7,""post_q"":169,""post_r"":6,""q"":841,""r"":15}"
""".strip()


# Data from https://arxiv.org/abs/2302.07395
y_data_csv = """
shots,errors,discards,seconds,decoder,strong_id,json_metadata
    310582,      1000,         0,   0.381,internal_correlated,3fa38268e206a0a127d29acf5c4a4a052221212874ab9c5b4dd91f46fa7c065c,"{""b"":""X"",""d"":3,""noise"":""SI1000"",""p"":0.001,""q"":17,""r"":3,""rb"":0}"
     35306,      1008,         0,   0.401,internal_correlated,327022b732c0a7e0909f787676176f7ec9b827e9bf14be0e78883bb90f97ed0f,"{""b"":""X"",""d"":3,""noise"":""SI1000"",""p"":0.003,""q"":17,""r"":3,""rb"":0}"
   1459043,      1001,         0,    6.07,internal_correlated,b8d514b09b7e49778f8e87260612a595d86766c9eda8d1d320b6a3cbc9084be0,"{""b"":""X"",""d"":5,""noise"":""SI1000"",""p"":0.001,""q"":49,""r"":5,""rb"":0}"
     53259,      1031,         0,   0.915,internal_correlated,4b23a05a6fdb99c801b8ef82d4bd2e426b596243077a2da15eb292a3245ba429,"{""b"":""X"",""d"":5,""noise"":""SI1000"",""p"":0.003,""q"":49,""r"":5,""rb"":0}"
   8411811,      1004,         0,    95.9,internal_correlated,a5aaf3b51a9162c68c3d2e5ff959e678020b4ebbbb6143c7912020467bf7c576,"{""b"":""X"",""d"":7,""noise"":""SI1000"",""p"":0.001,""q"":97,""r"":7,""rb"":0}"
     87153,      1061,         0,    6.62,internal_correlated,a35ea41c1dca97735bda9890e117d23026c947f61f9bdef1861ce9a8322c55be,"{""b"":""X"",""d"":7,""noise"":""SI1000"",""p"":0.003,""q"":97,""r"":7,""rb"":0}"
  58510590,      1049,         0,  1672.6,internal_correlated,be074faf61cbcc4469ad5b6e7cb740e4218e4b7a459e70aa16dca6cbc3c311f9,"{""b"":""X"",""d"":9,""noise"":""SI1000"",""p"":0.001,""q"":161,""r"":9,""rb"":0}"
    142320,      1031,         0,    25.8,internal_correlated,481a0b5b9a3eb95db7e6c66bc7e76c524d60a9eede566dc425430e554d6360e8,"{""b"":""X"",""d"":9,""noise"":""SI1000"",""p"":0.003,""q"":161,""r"":9,""rb"":0}"
 376985106,      1001,         0, 23602.6,internal_correlated,207ef531a4546b375ef50c26dbd5552e1eee7c7463a7a42ae4a07ba5812c3d75,"{""b"":""X"",""d"":11,""noise"":""SI1000"",""p"":0.001,""q"":241,""r"":11,""rb"":0}"
    264588,      1102,         0,   121.1,internal_correlated,cd72e8a0acc615983af0e6b3972aace55b736ecc9cc1ad42bebb247810092019,"{""b"":""X"",""d"":11,""noise"":""SI1000"",""p"":0.003,""q"":241,""r"":11,""rb"":0}"
1000000000,       329,         0,122953.9,internal_correlated,07642b3d70ef2acc3eea82cec0d98e782a69a3102786262064ce7f39ac9b5b6c,"{""b"":""X"",""d"":13,""noise"":""SI1000"",""p"":0.001,""q"":337,""r"":13,""rb"":0}"
    369616,      1009,         0,   388.3,internal_correlated,67240968f811ae9298992f7a5483d035b872cc787ac34daf4737297a67ca7397,"{""b"":""X"",""d"":13,""noise"":""SI1000"",""p"":0.003,""q"":337,""r"":13,""rb"":0}"
1000000000,        56,         0,227675.5,internal_correlated,d2a2c1af6b235bf2981187a282fa11fb307c9aa9ee9e93a076334d48a2a77602,"{""b"":""X"",""d"":15,""noise"":""SI1000"",""p"":0.001,""q"":449,""r"":15,""rb"":0}"
    692138,      1034,         0,  1442.8,internal_correlated,40521f4e01e53f17277a794b2f6c4c7d6a73e2a2a341472fe984a325732ec33c,"{""b"":""X"",""d"":15,""noise"":""SI1000"",""p"":0.003,""q"":449,""r"":15,""rb"":0}"
1000000000,         7,         0,393299.0,internal_correlated,9f8d1db53792b049e71b6e7c907041f838274cbd653711c7e54e325c37098afb,"{""b"":""X"",""d"":17,""noise"":""SI1000"",""p"":0.001,""q"":577,""r"":17,""rb"":0}"
   1198406,      1037,         0,  4355.3,internal_correlated,6435932f1abba09e2959f652a902936a77a7534656e9e915345f1437cf79bc65,"{""b"":""X"",""d"":17,""noise"":""SI1000"",""p"":0.003,""q"":577,""r"":17,""rb"":0}"
    463823,      1039,         0,   0.660,internal_correlated,0d5a232b3de764f9efdc60b4a8136ca3bcf7ae0008a91a92f06c2123e7b35e27,"{""b"":""X_magic_idle"",""d"":3,""noise"":""SI1000"",""p"":0.001,""q"":17,""r"":3,""rb"":0}"
     54197,      1001,         0,   0.193,internal_correlated,926587e6e460664ba5e88a9702bb3dc66f0df079f943d95d5f4ea4d6f7dbcbe8,"{""b"":""X_magic_idle"",""d"":3,""noise"":""SI1000"",""p"":0.003,""q"":17,""r"":3,""rb"":0}"
   5037815,      1001,         0,    15.0,internal_correlated,0efafa9d34b9d9db5b85301daffce038d2ba7287c0456a00e813bf169420a6ea,"{""b"":""X_magic_idle"",""d"":5,""noise"":""SI1000"",""p"":0.001,""q"":49,""r"":3,""rb"":0}"
    178232,      1020,         0,    1.82,internal_correlated,0ed0fd8055837d346df2e64d5370b2dbd53197b14b80771738a6a88aee95ea0a,"{""b"":""X_magic_idle"",""d"":5,""noise"":""SI1000"",""p"":0.003,""q"":49,""r"":3,""rb"":0}"
  57595791,      1043,         0,   217.4,internal_correlated,274ee909f814d45809fa0c54006c775539711532445edb7944fcd1e975fdd875,"{""b"":""X_magic_idle"",""d"":7,""noise"":""SI1000"",""p"":0.001,""q"":97,""r"":3,""rb"":0}"
    622488,      1138,         0,    11.0,internal_correlated,8d4ad79b4c3f7397ebba0e58048afacb716f69c7ac1282765377314d70a371be,"{""b"":""X_magic_idle"",""d"":7,""noise"":""SI1000"",""p"":0.003,""q"":97,""r"":3,""rb"":0}"
 642408423,      1026,         0,  3657.3,internal_correlated,6e7fdd6fcb1d4c49f54ce9604f72df15a50ce740f62bfc8cc18baadbbb275fe9,"{""b"":""X_magic_idle"",""d"":9,""noise"":""SI1000"",""p"":0.001,""q"":161,""r"":3,""rb"":0}"
   1758549,      1004,         0,    87.7,internal_correlated,02620ccc0ed567a8807cfd0ab04f437bb4a8c4e3d3b63705e77ec89e52c968a7,"{""b"":""X_magic_idle"",""d"":9,""noise"":""SI1000"",""p"":0.003,""q"":161,""r"":3,""rb"":0}"
1000000000,       125,         0,  9643.2,internal_correlated,c62f7e8928afccf0e310de4beeedd0cf94bb6c876fd766022cb2071d50f81ffe,"{""b"":""X_magic_idle"",""d"":11,""noise"":""SI1000"",""p"":0.001,""q"":241,""r"":3,""rb"":0}"
   5988691,      1034,         0,   506.0,internal_correlated,1cc1b4bfd1a6b1d7dbee47c3afd1aefd04d3ef5258e4a884db7f2ed4a0e63a99,"{""b"":""X_magic_idle"",""d"":11,""noise"":""SI1000"",""p"":0.003,""q"":241,""r"":3,""rb"":0}"
1000000000,        15,         0, 13820.1,internal_correlated,9ced9687d300836364ec0d32edd261c7c8a75d1ba3015106c3eeb2d2c6178646,"{""b"":""X_magic_idle"",""d"":13,""noise"":""SI1000"",""p"":0.001,""q"":337,""r"":3,""rb"":0}"
  19191302,      1017,         0,  2773.0,internal_correlated,8502a10443106ce3486ac5da797af59e4b8d41994d28870af49f6c3f07ba04bf,"{""b"":""X_magic_idle"",""d"":13,""noise"":""SI1000"",""p"":0.003,""q"":337,""r"":3,""rb"":0}"
1000000000,         1,         0, 19347.6,internal_correlated,a3fc92a98d7256ddff8c9e04df4e030db8c44b3eb3b6aea3d90f47701e682cc6,"{""b"":""X_magic_idle"",""d"":15,""noise"":""SI1000"",""p"":0.001,""q"":449,""r"":3,""rb"":0}"
  61936896,      1003,         0, 12821.2,internal_correlated,ef50c41c54794ff79edef84e6cf9f23afb774f8456087833ea68badd49dc2b50,"{""b"":""X_magic_idle"",""d"":15,""noise"":""SI1000"",""p"":0.003,""q"":449,""r"":3,""rb"":0}"
1000000000,         0,         0, 24715.7,internal_correlated,befcc098f381a49c849b589911a15a0d42c34864de01f902c010b729d19fcf18,"{""b"":""X_magic_idle"",""d"":17,""noise"":""SI1000"",""p"":0.001,""q"":577,""r"":3,""rb"":0}"
 196989417,      1004,         0, 32959.0,internal_correlated,e84b871ece6f05730a6e48239ab0f3e07aa6542b38719656ae29d680489626d8,"{""b"":""X_magic_idle"",""d"":17,""noise"":""SI1000"",""p"":0.003,""q"":577,""r"":3,""rb"":0}"
     82066,      1045,         0,   0.233,internal_correlated,c0a3cb08c7ecf3fa32402e1b0d4b70bdfcd33de0fc1b3d8b85cfbd23da5b7b59,"{""b"":""Y"",""d"":3,""noise"":""SI1000"",""p"":0.001,""q"":19,""r"":3,""rb"":0}"
     88410,      1043,         0,   0.315,internal_correlated,f4465ed24249d2aedb7e82226e2a73e9ed4969d24e4bba640b3574c12b2a812c,"{""b"":""Y"",""d"":3,""noise"":""SI1000"",""p"":0.001,""q"":19,""r"":3,""rb"":1}"
     90246,      1053,         0,   0.400,internal_correlated,57a3d420250a2efdcafd9ee4884a744a37970b5574c64778e36ceeb4ecf63b47,"{""b"":""Y"",""d"":3,""noise"":""SI1000"",""p"":0.001,""q"":19,""r"":3,""rb"":2}"
     85931,      1008,         0,   0.410,internal_correlated,0830d504f4e738fde9dd058fd379e857056b29dec7dd75766dd4cf1f368211b7,"{""b"":""Y"",""d"":3,""noise"":""SI1000"",""p"":0.001,""q"":19,""r"":3,""rb"":3}"
     87031,      1000,         0,   0.452,internal_correlated,406ddc7e6d628c936be173bfbdfb3d1bf51fc78452f5ed76f97b6d6692118edf,"{""b"":""Y"",""d"":3,""noise"":""SI1000"",""p"":0.001,""q"":19,""r"":3,""rb"":4}"
     87476,      1052,         0,   0.505,internal_correlated,c77905c10d4987d5e70bba018bc641486fd3aa290be84acef35983daa3d8880c,"{""b"":""Y"",""d"":3,""noise"":""SI1000"",""p"":0.001,""q"":19,""r"":3,""rb"":5}"
     85038,      1034,         0,   0.578,internal_correlated,3ddae04d1db410933d45cdcb11ebff1ab119b6e4ed52d21bd81255786f3c87c6,"{""b"":""Y"",""d"":3,""noise"":""SI1000"",""p"":0.001,""q"":19,""r"":3,""rb"":6}"
     90651,      1051,         0,   0.709,internal_correlated,8c981d82cf819fc043ab4f24862e4bd29120c3a397957ebc2fe7da51a5017b66,"{""b"":""Y"",""d"":3,""noise"":""SI1000"",""p"":0.001,""q"":19,""r"":3,""rb"":7}"
     99000,      1177,         0,   0.702,internal_correlated,df2bd6093e26c7804ea0e86e3d66828e2d02394015dd163ab56e03b97a65e029,"{""b"":""Y"",""d"":3,""noise"":""SI1000"",""p"":0.001,""q"":19,""r"":3,""rb"":8}"
     89396,      1020,         0,   0.826,internal_correlated,2e0ce91edab50d4c88629b3e1cc8735295081c150c3d1521741f32961b4111cc,"{""b"":""Y"",""d"":3,""noise"":""SI1000"",""p"":0.001,""q"":19,""r"":3,""rb"":9}"
     82709,      1005,         0,   0.735,internal_correlated,d01b7854fa2ec8823a8a4f39ab7588f30e4131f5fff940b8008d2c0da7a5a24a,"{""b"":""Y"",""d"":3,""noise"":""SI1000"",""p"":0.001,""q"":19,""r"":3,""rb"":10}"
     12202,      1084,         0,   0.183,internal_correlated,34cca25fa6bda00c7d0ae6af8978372d182ee3a5a7448dc121952ffaf835114c,"{""b"":""Y"",""d"":3,""noise"":""SI1000"",""p"":0.003,""q"":19,""r"":3,""rb"":1}"
    213800,      1059,         0,    2.05,internal_correlated,77e0d4cf7eecdda4f49bebe8140f1e8bfd711f854c2e7697fde6459f0757225e,"{""b"":""Y"",""d"":5,""noise"":""SI1000"",""p"":0.001,""q"":53,""r"":5,""rb"":0}"
    347458,      1022,         0,    3.62,internal_correlated,1f42b26a912adf1bb23151b0ecab5103327c70453b295d5ed244bb658c1c3668,"{""b"":""Y"",""d"":5,""noise"":""SI1000"",""p"":0.001,""q"":53,""r"":5,""rb"":1}"
    397512,      1002,         0,    4.93,internal_correlated,af80f3f1ce2e33b9cb17d620083f2a21a9288b90963bf1f5a27b4cf9b7e2ec23,"{""b"":""Y"",""d"":5,""noise"":""SI1000"",""p"":0.001,""q"":53,""r"":5,""rb"":2}"
    428318,      1017,         0,    6.48,internal_correlated,2c04310d1e5ca3f2987a72fa7f618eedfe369cb22f854617da1c1833586a3b43,"{""b"":""Y"",""d"":5,""noise"":""SI1000"",""p"":0.001,""q"":53,""r"":5,""rb"":3}"
    402208,      1003,         0,    7.08,internal_correlated,c0ebb0c68d25f2267d4bf5a8dcbab15e2c0f17ab96d73eb45b72c7ea64da67ef,"{""b"":""Y"",""d"":5,""noise"":""SI1000"",""p"":0.001,""q"":53,""r"":5,""rb"":4}"
    423880,      1047,         0,    7.66,internal_correlated,cb42856d505222c13b97fca0e0bf88434238a13d1b099733135194a444e455c9,"{""b"":""Y"",""d"":5,""noise"":""SI1000"",""p"":0.001,""q"":53,""r"":5,""rb"":5}"
    444411,      1089,         0,    9.14,internal_correlated,ff7ac928f2770c0da144b6de9e43721aa3d97f755033d1fb7c4d07efd24d766e,"{""b"":""Y"",""d"":5,""noise"":""SI1000"",""p"":0.001,""q"":53,""r"":5,""rb"":6}"
    393508,      1002,         0,    8.83,internal_correlated,d05b52af5e6a80c2c6141cc3f2bffed5814065cf4776c05bb8f1b0574455a9b2,"{""b"":""Y"",""d"":5,""noise"":""SI1000"",""p"":0.001,""q"":53,""r"":5,""rb"":7}"
    406951,      1039,         0,    10.1,internal_correlated,25927d07485d308fc828acf2e81a48de9af509f362dbf6737679f22e36d50bba,"{""b"":""Y"",""d"":5,""noise"":""SI1000"",""p"":0.001,""q"":53,""r"":5,""rb"":8}"
    423584,      1016,         0,    11.2,internal_correlated,8674160803139529ba47e1acf8ae16d1d1a69a70a1b9974ac8030954987c63e0,"{""b"":""Y"",""d"":5,""noise"":""SI1000"",""p"":0.001,""q"":53,""r"":5,""rb"":9}"
    414379,      1022,         0,    12.2,internal_correlated,2da61ab600c685116228a437707e12520073c45008e0fef1567879fc2cfe9969,"{""b"":""Y"",""d"":5,""noise"":""SI1000"",""p"":0.001,""q"":53,""r"":5,""rb"":10}"
     15458,      1002,         0,    1.78,internal_correlated,1407dc28081eee60f637748f4c419bd7056c4f2537d83199dab0767431c8df2a,"{""b"":""Y"",""d"":5,""noise"":""SI1000"",""p"":0.003,""q"":53,""r"":5,""rb"":2}"
    361978,      1005,         0,    9.42,internal_correlated,b096a8b44e4cac423e4fc12b0e9a995ed414e3eb7ac5cbc7b3a27bc80c10f7ef,"{""b"":""Y"",""d"":7,""noise"":""SI1000"",""p"":0.001,""q"":103,""r"":7,""rb"":0}"
   1115857,      1000,         0,    31.6,internal_correlated,83f15f6ef6b1875c0987204067e841fa152221cac84a6c1a227cd67a38440a36,"{""b"":""Y"",""d"":7,""noise"":""SI1000"",""p"":0.001,""q"":103,""r"":7,""rb"":1}"
   1914124,      1001,         0,    60.7,internal_correlated,1812b2a57ac2b7f9ad2004223f2231958ddba50c1a72d59b22ffa172f5a8d59d,"{""b"":""Y"",""d"":7,""noise"":""SI1000"",""p"":0.001,""q"":103,""r"":7,""rb"":2}"
   2288046,      1004,         0,    83.2,internal_correlated,c704a243ba11f4cce5cb8d1b670e827c2b6189cca2a67f223acd921f3403f722,"{""b"":""Y"",""d"":7,""noise"":""SI1000"",""p"":0.001,""q"":103,""r"":7,""rb"":3}"
   2348948,      1002,         0,    97.8,internal_correlated,78e3a974765e70e6908f79a8478d99f3ffddc5f5808e16a17751a0026002b885,"{""b"":""Y"",""d"":7,""noise"":""SI1000"",""p"":0.001,""q"":103,""r"":7,""rb"":4}"
   2488836,      1022,         0,   109.1,internal_correlated,e5d8ecfd720080cd97f1fc9db52cefe17259c47ace30d6347032242458619236,"{""b"":""Y"",""d"":7,""noise"":""SI1000"",""p"":0.001,""q"":103,""r"":7,""rb"":5}"
   2296712,      1002,         0,   116.4,internal_correlated,6ff2a8d4c956654fc33521fbd7566e2b4d47756769165f5a6c2f2f17f4ac7e5e,"{""b"":""Y"",""d"":7,""noise"":""SI1000"",""p"":0.001,""q"":103,""r"":7,""rb"":6}"
   2296889,      1002,         0,   129.1,internal_correlated,cfe52b780b8c5989583237b11dafc7c6cd09103d29437eaa37c80dbf1c4070a6,"{""b"":""Y"",""d"":7,""noise"":""SI1000"",""p"":0.001,""q"":103,""r"":7,""rb"":7}"
   2394920,      1001,         0,   143.0,internal_correlated,7e48332888d308aad928ab355113daf81b904a612995468aeb6501b6afa9e7e9,"{""b"":""Y"",""d"":7,""noise"":""SI1000"",""p"":0.001,""q"":103,""r"":7,""rb"":8}"
   2335843,      1003,         0,   153.3,internal_correlated,9ceb015618680b2aae7a5c466c48f83379078369643563a0d7383c7872d41025,"{""b"":""Y"",""d"":7,""noise"":""SI1000"",""p"":0.001,""q"":103,""r"":7,""rb"":9}"
   2565219,      1002,         0,   181.6,internal_correlated,febaaefddcbcc97487969d0a6cdee2e30ff0c6686131452f9ff3ee297b9e5ee1,"{""b"":""Y"",""d"":7,""noise"":""SI1000"",""p"":0.001,""q"":103,""r"":7,""rb"":10}"
     21600,      1017,         0,    5.43,internal_correlated,58759b279b7c7772f160811e72f8504cc162a63ab6f82da71437df68a135f14c,"{""b"":""Y"",""d"":7,""noise"":""SI1000"",""p"":0.003,""q"":103,""r"":7,""rb"":3}"
  25603082,      1004,         0,  1155.6,internal_correlated,f37543dff5fe8ca7fe7c708724125ce813af9e74e752fb2acb13e2de2e413fad,"{""b"":""Y"",""d"":9,""noise"":""SI1000"",""p"":0.001,""q"":169,""r"":0,""rb"":4}"
  24336555,      1010,         0,  1309.7,internal_correlated,db788d46dd2e4f16833f6d6da988b671c081c24661a2b5cd085347cee33b3f51,"{""b"":""Y"",""d"":9,""noise"":""SI1000"",""p"":0.001,""q"":169,""r"":1,""rb"":4}"
  22129443,      1007,         0,  1253.2,internal_correlated,aa7e2e57c110d6b0f76dd56b08ed438e91e1b976d0c2de8d43d8a83cdf4d46e4,"{""b"":""Y"",""d"":9,""noise"":""SI1000"",""p"":0.001,""q"":169,""r"":2,""rb"":4}"
  19535734,      1003,         0,  1197.2,internal_correlated,5ba9dbaa9fc439f772d9cc36d25dd4632ab5dd1a2e0beed13dea4d1a31f83371,"{""b"":""Y"",""d"":9,""noise"":""SI1000"",""p"":0.001,""q"":169,""r"":3,""rb"":4}"
  19675924,      1030,         0,  1328.6,internal_correlated,1987148a469fa22a43f0c7cdbfa3203cb7aa331c64efc5f73fdae5da1da299f4,"{""b"":""Y"",""d"":9,""noise"":""SI1000"",""p"":0.001,""q"":169,""r"":4,""rb"":4}"
  18853385,      1019,         0,  1401.6,internal_correlated,0e3f01dc769efa729b0ce0bbc2749693f5ece63ae12ee3d470256c8d1f7f5664,"{""b"":""Y"",""d"":9,""noise"":""SI1000"",""p"":0.001,""q"":169,""r"":5,""rb"":4}"
  16073719,      1031,         0,  1296.5,internal_correlated,954fa3c7fb37e2f1ab9bf0cea199e36e1d7f211910c22607d3534894bfb33da6,"{""b"":""Y"",""d"":9,""noise"":""SI1000"",""p"":0.001,""q"":169,""r"":6,""rb"":4}"
  15872022,      1002,         0,  1297.8,internal_correlated,3295a01fdbeae0f1d8de8e996cd7a8e49f67c03f9291e6749c7d2d6e788e615e,"{""b"":""Y"",""d"":9,""noise"":""SI1000"",""p"":0.001,""q"":169,""r"":7,""rb"":4}"
  16073719,      1053,         0,  1405.5,internal_correlated,f7dccadc310917760679ff97f06cf8d6b3d3b904496e76533cb32d31843bc416,"{""b"":""Y"",""d"":9,""noise"":""SI1000"",""p"":0.001,""q"":169,""r"":8,""rb"":4}"
    441543,      1035,         0,    23.8,internal_correlated,34ab80bd17be55c71bd9c4f152e57d98ca65808d82ca9392cb8642e766041b64,"{""b"":""Y"",""d"":9,""noise"":""SI1000"",""p"":0.001,""q"":169,""r"":9,""rb"":0}"
   2269233,      1029,         0,   142.5,internal_correlated,0caaae37b2abdde48311476e837ea51a6b773a64ee45e14142da2e2684723ac4,"{""b"":""Y"",""d"":9,""noise"":""SI1000"",""p"":0.001,""q"":169,""r"":9,""rb"":1}"
   6684885,      1000,         0,   484.6,internal_correlated,0b3b70f50951e518736be16174dd624b3f3a0f76b224092a9ce48aad7210b933,"{""b"":""Y"",""d"":9,""noise"":""SI1000"",""p"":0.001,""q"":169,""r"":9,""rb"":2}"
  12158835,      1016,         0,   957.2,internal_correlated,209529da4b9585c48609728528b510601f72df1e5d58eaed08b3b5038dce9caf,"{""b"":""Y"",""d"":9,""noise"":""SI1000"",""p"":0.001,""q"":169,""r"":9,""rb"":3}"
  14245126,      1005,         0,  1150.3,internal_correlated,b7248f5d5a7966fa4bef43ae4805da6b2b3fff070d4dcafb2f1e7b87a5587078,"{""b"":""Y"",""d"":9,""noise"":""SI1000"",""p"":0.001,""q"":169,""r"":9,""rb"":4}"
  15412410,      1015,         0,  1327.2,internal_correlated,d8915b43b7b98920a7b647d012b6be7147233f578fe082c4bd486984fc36f242,"{""b"":""Y"",""d"":9,""noise"":""SI1000"",""p"":0.001,""q"":169,""r"":9,""rb"":5}"
  15312978,      1011,         0,  1333.7,internal_correlated,453b5f855f25e2b07f51a6a157df99dadc255cac2186ec236da20956c6ee64e5,"{""b"":""Y"",""d"":9,""noise"":""SI1000"",""p"":0.001,""q"":169,""r"":9,""rb"":6}"
  15383437,      1042,         0,  1448.1,internal_correlated,45ec7beebbaeadeac79c38ebd72708aa50cc372f96ba911273af5ded35a4d9bd,"{""b"":""Y"",""d"":9,""noise"":""SI1000"",""p"":0.001,""q"":169,""r"":9,""rb"":7}"
  15646491,      1027,         0,  1537.3,internal_correlated,c00a59538475a3e7c7114b9e044763faaeb6826dac30977fa0455bebe4876df8,"{""b"":""Y"",""d"":9,""noise"":""SI1000"",""p"":0.001,""q"":169,""r"":9,""rb"":8}"
  16742746,      1040,         0,  1627.2,internal_correlated,b82d7420ed8b2bd850b575fe708a1f01ed251088b5f59dc8ebcd8afd90599dca,"{""b"":""Y"",""d"":9,""noise"":""SI1000"",""p"":0.001,""q"":169,""r"":9,""rb"":9}"
  15836452,      1002,         0,  2317.1,internal_correlated,b22267554b1bc74e6da04abd4905a7eb0f2060c5a31361a492aaef6f418bacad,"{""b"":""Y"",""d"":9,""noise"":""SI1000"",""p"":0.001,""q"":169,""r"":9,""rb"":10}"
  13185893,      1005,         0,  1351.8,internal_correlated,31dc6b3abb8074dc5b5abfb12dedae52cb67487381e345b65469d116b03211fb,"{""b"":""Y"",""d"":9,""noise"":""SI1000"",""p"":0.001,""q"":169,""r"":10,""rb"":4}"
     38729,      1084,         0,    25.2,internal_correlated,f1be687d54a24def2d55ea37ea72633e1d1a825d6d781d29d96b87c73eeeadef,"{""b"":""Y"",""d"":9,""noise"":""SI1000"",""p"":0.003,""q"":169,""r"":9,""rb"":4}"
    457552,      1015,         0,    51.5,internal_correlated,788af7cafcdb78bd98c3bf0080770dc63a17f33ba1e22fb2b78a8bbb4585529f,"{""b"":""Y"",""d"":11,""noise"":""SI1000"",""p"":0.001,""q"":251,""r"":11,""rb"":0}"
   3012015,      1037,         0,   391.1,internal_correlated,d37a057d68a42d56c7affccb41fa6daad6ba9cb1ad2aa9cf939adc74f3026dd0,"{""b"":""Y"",""d"":11,""noise"":""SI1000"",""p"":0.001,""q"":251,""r"":11,""rb"":1}"
  14963190,      1017,         0,  2368.3,internal_correlated,2c71f63ae4f69990f21d1a01a7991b337c09d192d4e59ba9879cdcf8f4217443,"{""b"":""Y"",""d"":11,""noise"":""SI1000"",""p"":0.001,""q"":251,""r"":11,""rb"":2}"
  46290322,      1040,         0,  7418.7,internal_correlated,2991978e9021589f29c79f09c6018d6083e340d430c33d9de30bf1ea7b831db8,"{""b"":""Y"",""d"":11,""noise"":""SI1000"",""p"":0.001,""q"":251,""r"":11,""rb"":3}"
  80326374,      1055,         0, 14248.0,internal_correlated,9585ad3c181e43df3b3489746590a6a1893b3412147eb52fed92c744625f1770,"{""b"":""Y"",""d"":11,""noise"":""SI1000"",""p"":0.001,""q"":251,""r"":11,""rb"":4}"
  95593201,      1045,         0, 18641.3,internal_correlated,bfec82ee0d2da8dbb06443549aff5bd95eae1589e6e3beea867279419e9197e1,"{""b"":""Y"",""d"":11,""noise"":""SI1000"",""p"":0.001,""q"":251,""r"":11,""rb"":5}"
 103319286,      1010,         0, 22176.6,internal_correlated,99992cb1278fbc7c44744fc3e14651e202b3dd08220393d8495f66c1dde4b792,"{""b"":""Y"",""d"":11,""noise"":""SI1000"",""p"":0.001,""q"":251,""r"":11,""rb"":6}"
 101490252,      1017,         0, 23874.3,internal_correlated,69f7bb76c6b3f72d9f9d1a8264945ad3369bb35f3fc3ade5c536e584de68cf9d,"{""b"":""Y"",""d"":11,""noise"":""SI1000"",""p"":0.001,""q"":251,""r"":11,""rb"":7}"
 102475952,      1011,         0, 26473.6,internal_correlated,02bc9aa838d1f3080ec6b9882afd94f358ef5e64885b78c107e0b10a4625e445,"{""b"":""Y"",""d"":11,""noise"":""SI1000"",""p"":0.001,""q"":251,""r"":11,""rb"":8}"
 102579832,      1029,         0, 28725.5,internal_correlated,59946046851ee12a10fa2740a4f77a1768218cd86d03877309602f5cbe3091b9,"{""b"":""Y"",""d"":11,""noise"":""SI1000"",""p"":0.001,""q"":251,""r"":11,""rb"":9}"
 110752708,      1018,         0, 33977.1,internal_correlated,51a035bf5ef492adbc52614459f3bf76ede7d027468a36c3a2c215e45a333746,"{""b"":""Y"",""d"":11,""noise"":""SI1000"",""p"":0.001,""q"":251,""r"":11,""rb"":10}"
     53278,      1003,         0,    89.2,internal_correlated,05a35bd2aac93c02a8a4d4453dc16ec36d0c918fbb91658a04f708ef50f43a4f,"{""b"":""Y"",""d"":11,""noise"":""SI1000"",""p"":0.003,""q"":251,""r"":11,""rb"":5}"
    465443,      1002,         0,   101.0,internal_correlated,a70590d93b3f274fc6fd2e15b9af35d9edfd5118545688fab0aa0e9654f9a09e,"{""b"":""Y"",""d"":13,""noise"":""SI1000"",""p"":0.001,""q"":349,""r"":13,""rb"":0}"
   3270049,      1041,         0,   774.9,internal_correlated,248a4e3c30e9316da317754f3a7c2c6b98add6b2f3fa60d1ea9334a1d8ac4067,"{""b"":""Y"",""d"":13,""noise"":""SI1000"",""p"":0.001,""q"":349,""r"":13,""rb"":1}"
  21477518,      1014,         0,  5659.2,internal_correlated,a4605b2f2d9f0d0ce0befad5e4b98239cd0ea32a03ddc887d61d0bd1297d48f6,"{""b"":""Y"",""d"":13,""noise"":""SI1000"",""p"":0.001,""q"":349,""r"":13,""rb"":2}"
  99382020,      1014,         0, 28292.0,internal_correlated,e2c24dc6823874608ca7e2e562107d20c9533043d220128e9d741a95476dea81,"{""b"":""Y"",""d"":13,""noise"":""SI1000"",""p"":0.001,""q"":349,""r"":13,""rb"":3}"
 315269304,      1004,         0,100889.5,internal_correlated,674ba48404cb1b59e5a64e269ec808aad7a3a9867431fb481881cdf083e0d131,"{""b"":""Y"",""d"":13,""noise"":""SI1000"",""p"":0.001,""q"":349,""r"":13,""rb"":4}"
 534421097,      1005,         0,190490.8,internal_correlated,1e64661b27cc3d0767cec5a22d4cf65039f86a54a27a7630bebb39c620a42894,"{""b"":""Y"",""d"":13,""noise"":""SI1000"",""p"":0.001,""q"":349,""r"":13,""rb"":5}"
 689448426,      1008,         0,271514.8,internal_correlated,6d59fb17dbba79d51a6209bf3f15ec55aedc26c6cbc76148cfb3444782029b78,"{""b"":""Y"",""d"":13,""noise"":""SI1000"",""p"":0.001,""q"":349,""r"":13,""rb"":6}"
 760212917,      1004,         0,330922.0,internal_correlated,7c27eabf41f2d8a970fcc64ac6108cd036a3e227db007a91ca34d5645ef68161,"{""b"":""Y"",""d"":13,""noise"":""SI1000"",""p"":0.001,""q"":349,""r"":13,""rb"":7}"
 726528858,      1002,         0,360563.3,internal_correlated,d4dbc67713d080462bfc81e42fde38ad3b25320cbb5461ce6948fb7bcea28c72,"{""b"":""Y"",""d"":13,""noise"":""SI1000"",""p"":0.001,""q"":349,""r"":13,""rb"":8}"
 749927342,      1016,         0,402663.3,internal_correlated,aa1907e4e3a675c7fa627597db9ecce9c7e9299f1359f0c26ce418ecafa07d4d,"{""b"":""Y"",""d"":13,""noise"":""SI1000"",""p"":0.001,""q"":349,""r"":13,""rb"":9}"
 709519482,      1010,         0,401779.0,internal_correlated,ef6e78a6edd9219644f36bef693aabb52fe39394b457e0fe63fb2511968ca5cd,"{""b"":""Y"",""d"":13,""noise"":""SI1000"",""p"":0.001,""q"":349,""r"":13,""rb"":10}"
     90651,      1073,         0,   339.1,internal_correlated,b04116b0586eab63b984949f38c32c66d72bbf90b88472eee2ac6deca03d047d,"{""b"":""Y"",""d"":13,""noise"":""SI1000"",""p"":0.003,""q"":349,""r"":13,""rb"":6}"
1000000000,       131,         0,317935.5,internal_correlated,ef0f22bfd380b1293ded247a84e1945443b5d14c0279c38827b4c2920f565866,"{""b"":""Y"",""d"":15,""noise"":""SI1000"",""p"":0.001,""q"":463,""r"":0,""rb"":7}"
1000000000,       129,         0,332254.2,internal_correlated,f8a2e9b2b81baacbc5fe4e3575860b08689c3dfd97d350fea0a7afc827a7f5e2,"{""b"":""Y"",""d"":15,""noise"":""SI1000"",""p"":0.001,""q"":463,""r"":1,""rb"":7}"
1000000000,       141,         0,357816.3,internal_correlated,4170d3a6c767e8989314112eed5e719d9b7be6fa1ff868a21b9bbcb0705a0228,"{""b"":""Y"",""d"":15,""noise"":""SI1000"",""p"":0.001,""q"":463,""r"":2,""rb"":7}"
1000000000,       131,         0,383374.1,internal_correlated,d5619d3c002cb173ca5ee7e272de634b97fb769a131fbb4137c5997c0cd0d75d,"{""b"":""Y"",""d"":15,""noise"":""SI1000"",""p"":0.001,""q"":463,""r"":3,""rb"":7}"
1000000000,       130,         0,409755.4,internal_correlated,e53c8609c14a08b94e85720813f2fc668ffb3fa376743699aa767cd7fa8c72cf,"{""b"":""Y"",""d"":15,""noise"":""SI1000"",""p"":0.001,""q"":463,""r"":4,""rb"":7}"
1000000000,       148,         0,437015.7,internal_correlated,ea4db1443d80fbd8557c7e29a06eba8d9ea33c17d763abfb4a15cb160ff590dc,"{""b"":""Y"",""d"":15,""noise"":""SI1000"",""p"":0.001,""q"":463,""r"":5,""rb"":7}"
1000000000,       159,         0,463789.1,internal_correlated,563b91853f380c705b00d806e51f72e278c50aa0e614ad0b694d4e582a3bd742,"{""b"":""Y"",""d"":15,""noise"":""SI1000"",""p"":0.001,""q"":463,""r"":6,""rb"":7}"
1000000000,       159,         0,491895.7,internal_correlated,a5927f79d1f9b3b7c6c085f1615b7ead7922b997d1df316ee87a3365caa2b91d,"{""b"":""Y"",""d"":15,""noise"":""SI1000"",""p"":0.001,""q"":463,""r"":7,""rb"":7}"
1000000000,       194,         0,520063.3,internal_correlated,8398bd69b7e19b5f4e05433b274328aba8ff8c02971404f2549c783f59663abf,"{""b"":""Y"",""d"":15,""noise"":""SI1000"",""p"":0.001,""q"":463,""r"":8,""rb"":7}"
1000000000,       197,         0,546753.5,internal_correlated,0706964396c3cd3b56e17fc552bf959cb3bc02090702595d6c7559c3c093ce41,"{""b"":""Y"",""d"":15,""noise"":""SI1000"",""p"":0.001,""q"":463,""r"":9,""rb"":7}"
1000000000,       204,         0,591530.2,internal_correlated,b6042108a396ba0555c150d013e5e26c8bc3426666b12c60e848717786fb313d,"{""b"":""Y"",""d"":15,""noise"":""SI1000"",""p"":0.001,""q"":463,""r"":10,""rb"":7}"
    511052,      1026,         0,   189.4,internal_correlated,70e8ae04614e823e892e722d3450de8c1b4ad58d0c8824abfa957d649a1f8d8c,"{""b"":""Y"",""d"":15,""noise"":""SI1000"",""p"":0.001,""q"":463,""r"":15,""rb"":0}"
   3362929,      1014,         0,  1372.5,internal_correlated,9d1ee9f9a4ce7a5a0bf2724b78b74af758c898ae3484a2f707e45fa719455ac4,"{""b"":""Y"",""d"":15,""noise"":""SI1000"",""p"":0.001,""q"":463,""r"":15,""rb"":1}"
  22847382,      1028,         0, 10679.6,internal_correlated,b57ae42c00aa9fe2f0b62940adca497d372ec28c4b0ce013b32c93e06765bd78,"{""b"":""Y"",""d"":15,""noise"":""SI1000"",""p"":0.001,""q"":463,""r"":15,""rb"":2}"
 140890329,      1011,         0, 72006.1,internal_correlated,1d75f8ee14e20552560e14347edfad49f1b8a63edb6a48b018d58405301a7248,"{""b"":""Y"",""d"":15,""noise"":""SI1000"",""p"":0.001,""q"":463,""r"":15,""rb"":3}"
 640807276,      1003,         0,357606.9,internal_correlated,874415c55bf8d1d7fd989545e30d4568231c085a61fecf738e839736bd85c5bc,"{""b"":""Y"",""d"":15,""noise"":""SI1000"",""p"":0.001,""q"":463,""r"":15,""rb"":4}"
1000000000,       500,         0,615146.2,internal_correlated,cddb98870e6d73b2b8a2b96542c98c43644b02444efea615ffc801f61d5fedcf,"{""b"":""Y"",""d"":15,""noise"":""SI1000"",""p"":0.001,""q"":463,""r"":15,""rb"":5}"
1000000000,       273,         0,681095.5,internal_correlated,b312bb6de073558319cbfd2786c007cc4bff354c38b6a2d340b2a747b7840fcd,"{""b"":""Y"",""d"":15,""noise"":""SI1000"",""p"":0.001,""q"":463,""r"":15,""rb"":6}"
1000000000,       230,         0,748608.2,internal_correlated,a94fd1de790c908721db6e8966e2fa218168c4ee42ce1ebc237a892ebbd120f2,"{""b"":""Y"",""d"":15,""noise"":""SI1000"",""p"":0.001,""q"":463,""r"":15,""rb"":7}"
1000000000,       209,         0,818132.8,internal_correlated,d320c608622efddcc4ef66c84ec2b98d38e5e677f1141855b0192c4e83c1ad13,"{""b"":""Y"",""d"":15,""noise"":""SI1000"",""p"":0.001,""q"":463,""r"":15,""rb"":8}"
1000000000,       190,         0,894403.6,internal_correlated,649b060ec6c78f5b569d6cbbc8e6eedd370831cb5e8ee7499756dabe4c241f7d,"{""b"":""Y"",""d"":15,""noise"":""SI1000"",""p"":0.001,""q"":463,""r"":15,""rb"":9}"
1000000000,       228,         0,974561.0,internal_correlated,03d3ef022a067890d9191d90d90d87b1aeca886ca43001f4cd8eb9e23416b220,"{""b"":""Y"",""d"":15,""noise"":""SI1000"",""p"":0.001,""q"":463,""r"":15,""rb"":10}"
    160366,      1080,         0,  1127.3,internal_correlated,9331f47306d3fd7bdaebd631b06bb983c44606240b9d6eb42ae34e894ba9e712,"{""b"":""Y"",""d"":15,""noise"":""SI1000"",""p"":0.003,""q"":463,""r"":15,""rb"":7}"
    458271,      1016,         0,   260.1,internal_correlated,ef5033a1a68553452a8a6f14d5632f6f5c91c98b56aaba7adbce5c3293251971,"{""b"":""Y"",""d"":17,""noise"":""SI1000"",""p"":0.001,""q"":593,""r"":17,""rb"":0}"
   3368061,      1008,         0,  2236.3,internal_correlated,db0b3653068225ab69391e16266f207c77de3f5554ef78b3f05f309b3355cee6,"{""b"":""Y"",""d"":17,""noise"":""SI1000"",""p"":0.001,""q"":593,""r"":17,""rb"":1}"
  24463417,      1018,         0, 18617.3,internal_correlated,1a23a93758e79ae37cd4a954b0d9232b15d0f716c9cd5175707c67def164ce2d,"{""b"":""Y"",""d"":17,""noise"":""SI1000"",""p"":0.001,""q"":593,""r"":17,""rb"":2}"
 151047143,      1004,         0,126111.1,internal_correlated,bdb14220823cce94cf1518490e4efcbb997cf0d8f3e43e05efe80e8e519eeef9,"{""b"":""Y"",""d"":17,""noise"":""SI1000"",""p"":0.001,""q"":593,""r"":17,""rb"":3}"
 918799702,      1005,         0,840588.1,internal_correlated,d1d126cebcd5eb78885f92c63fb3d46dbbdc001422f13ee1759ce7b4370308a4,"{""b"":""Y"",""d"":17,""noise"":""SI1000"",""p"":0.001,""q"":593,""r"":17,""rb"":4}"
1000000000,       232,         0,1023004.3,internal_correlated,fd3e5b62864ccafd9d58203361086a30e85148aff2ee38da07ea74f625f25c9a,"{""b"":""Y"",""d"":17,""noise"":""SI1000"",""p"":0.001,""q"":593,""r"":17,""rb"":5}"
1000000000,        56,         0,1143707.8,internal_correlated,9659f36387571b456c8e64a67b95b61fdfff2e06bc9136d4d7173df67fec64d2,"{""b"":""Y"",""d"":17,""noise"":""SI1000"",""p"":0.001,""q"":593,""r"":17,""rb"":6}"
1000000000,        37,         0,1225659.5,internal_correlated,458568fc9ade03c33d602496d6a7fe5c9d12a1f36c93f5ff75db0ee235db61aa,"{""b"":""Y"",""d"":17,""noise"":""SI1000"",""p"":0.001,""q"":593,""r"":17,""rb"":7}"
1000000000,        28,         0,1316585.7,internal_correlated,92e57447a2c35046a08f06c7296a4ccdbe14c332b397f2f69e677c0384549cfa,"{""b"":""Y"",""d"":17,""noise"":""SI1000"",""p"":0.001,""q"":593,""r"":17,""rb"":8}"
1000000000,        28,         0,1420891.9,internal_correlated,ff3efa01c35f07124d499ac832add872ab8cdcaaf74ac273a91fd0aead0a84ff,"{""b"":""Y"",""d"":17,""noise"":""SI1000"",""p"":0.001,""q"":593,""r"":17,""rb"":9}"
1000000000,        24,         0,1530924.1,internal_correlated,9e300ddcff66408b61f1c0013b42f62c3ad32a885f5dfb3999df15c3e96fc330,"{""b"":""Y"",""d"":17,""noise"":""SI1000"",""p"":0.001,""q"":593,""r"":17,""rb"":10}"
    273034,      1106,         0,  3164.3,internal_correlated,3c505d18fd11ee9a0eef13ee51e43260eb33d8cd3332b79b17eacfba1e0426b3,"{""b"":""Y"",""d"":17,""noise"":""SI1000"",""p"":0.003,""q"":593,""r"":17,""rb"":8}"
   1282404,      1006,         0,    62.7,internal_correlated,efab721696fc954e61730fdaca08dc6738f6f34b805eedc881daf261986be114,"{""b"":""Y_braid"",""d"":9,""noise"":""SI1000"",""p"":0.001,""q"":173,""r"":0,""rb"":4}"
   6151842,      1024,         0,   319.3,internal_correlated,4c2d28ee976d0539e9388e60dc7a5b31e261260c81383adca4e608b5afd5a3ed,"{""b"":""Y_braid"",""d"":9,""noise"":""SI1000"",""p"":0.001,""q"":173,""r"":1,""rb"":4}"
  15070584,      1017,         0,   849.6,internal_correlated,179d1a3dfa1f0ff995645d702084fc40829fe1927ff1e4800170a17720c63fd3,"{""b"":""Y_braid"",""d"":9,""noise"":""SI1000"",""p"":0.001,""q"":173,""r"":2,""rb"":4}"
  19313739,      1001,         0,  1165.9,internal_correlated,f5e5ac6051eba2ac979bbf8509e6ce5266345b5c54a8ffdb2fda6b08242417f0,"{""b"":""Y_braid"",""d"":9,""noise"":""SI1000"",""p"":0.001,""q"":173,""r"":3,""rb"":4}"
  18565725,      1046,         0,  1234.7,internal_correlated,b2d8708940051c4952058bdc18a1f34f7f4bedc762dd19c14c819d6130ece732,"{""b"":""Y_braid"",""d"":9,""noise"":""SI1000"",""p"":0.001,""q"":173,""r"":4,""rb"":4}"
  16891986,      1003,         0,  1224.6,internal_correlated,a1f1bca0052e6ee1636911b497e3e5c0ee16be2354faeae64e9d1ba5b11f5a34,"{""b"":""Y_braid"",""d"":9,""noise"":""SI1000"",""p"":0.001,""q"":173,""r"":5,""rb"":4}"
  16475951,      1001,         0,  1332.2,internal_correlated,46004f20ee0fecb0730c0f17c8c625bb0a5c4650e989b65f48768beead3c62ef,"{""b"":""Y_braid"",""d"":9,""noise"":""SI1000"",""p"":0.001,""q"":173,""r"":6,""rb"":4}"
  14731530,      1005,         0,  1204.9,internal_correlated,dbc6d73f874cce900444e40b6fc9a0848b1ace513f7f84dfbdb50d5cd212c722,"{""b"":""Y_braid"",""d"":9,""noise"":""SI1000"",""p"":0.001,""q"":173,""r"":7,""rb"":4}"
  14858423,      1005,         0,  1300.4,internal_correlated,594e49a9fdeecec4f982b37fa18237e3790450da3ba7b8b2878830ec5665c991,"{""b"":""Y_braid"",""d"":9,""noise"":""SI1000"",""p"":0.001,""q"":173,""r"":8,""rb"":4}"
  14859260,      1044,         0,  1350.4,internal_correlated,ed623b45a55ac25ce4b0f87f9d6f437e32909c9ff3d5dbe2170944c002bb5e41,"{""b"":""Y_braid"",""d"":9,""noise"":""SI1000"",""p"":0.001,""q"":173,""r"":9,""rb"":4}"
  12903409,      1010,         0,  1345.7,internal_correlated,273480cbe12244d3c36cc0237e95ef8e23fde9872c791277fa5594470f8cf1de,"{""b"":""Y_braid"",""d"":9,""noise"":""SI1000"",""p"":0.001,""q"":173,""r"":10,""rb"":4}"
   1406733,      1008,         0,   467.1,internal_correlated,70aa9b65f1efc56c9c409d8aa5b40b081b448230876d6d1bb1718f8f85e4a4a7,"{""b"":""Y_braid"",""d"":15,""noise"":""SI1000"",""p"":0.001,""q"":470,""r"":0,""rb"":7}"
   8653350,      1143,         0,  2743.1,internal_correlated,3341d3994bb39a37c2695793a261dcd935dcf0e3240a84670363fc4ff964a090,"{""b"":""Y_braid"",""d"":15,""noise"":""SI1000"",""p"":0.001,""q"":470,""r"":1,""rb"":7}"
  53615398,      1107,         0, 19339.0,internal_correlated,9077ebe12526575f470deacd75b1d127a23f4b7dd302a1e8324edb6434dab5ca,"{""b"":""Y_braid"",""d"":15,""noise"":""SI1000"",""p"":0.001,""q"":470,""r"":2,""rb"":7}"
 314770919,      1004,         0, 92706.7,internal_correlated,47e1ff133743e24f31d17b51e06dd1574b1cac7b5a576377fa932e4f6d7ee702,"{""b"":""Y_braid"",""d"":15,""noise"":""SI1000"",""p"":0.001,""q"":470,""r"":3,""rb"":7}"
1000000000,       648,         0,276234.8,internal_correlated,cd201a769228bc832a809af855c2b67077c6451352b756b1f72fb1c5992143c2,"{""b"":""Y_braid"",""d"":15,""noise"":""SI1000"",""p"":0.001,""q"":470,""r"":4,""rb"":7}"
1000000000,       231,         0,292304.3,internal_correlated,12ebd18405f5c1600928cefb5fcb8e80e0264369bf9a0bfc467689e20b5bd9e1,"{""b"":""Y_braid"",""d"":15,""noise"":""SI1000"",""p"":0.001,""q"":470,""r"":5,""rb"":7}"
1000000000,       200,         0,309841.4,internal_correlated,18558b777b5535262f36bd0460fdf0df1251b1da0fc911abe37a3e60dd3cc42a,"{""b"":""Y_braid"",""d"":15,""noise"":""SI1000"",""p"":0.001,""q"":470,""r"":6,""rb"":7}"
1000000000,       165,         0,328035.8,internal_correlated,6713d6ee2b216b993d75fd0970f388491915624c2cb98a862c697c6c2cb1fde1,"{""b"":""Y_braid"",""d"":15,""noise"":""SI1000"",""p"":0.001,""q"":470,""r"":7,""rb"":7}"
1000000000,       193,         0,345309.3,internal_correlated,49bc4a4b0b73adba23771cf8016f7ee471122effe5b8a9760d4469bc54238bcb,"{""b"":""Y_braid"",""d"":15,""noise"":""SI1000"",""p"":0.001,""q"":470,""r"":8,""rb"":7}"
1000000000,       183,         0,362889.1,internal_correlated,e8844727b5609360e21b8f982b42c54db964fa4803d5392de579f8f43267a8c6,"{""b"":""Y_braid"",""d"":15,""noise"":""SI1000"",""p"":0.001,""q"":470,""r"":9,""rb"":7}"
1000000000,       225,         0,592281.2,internal_correlated,3277fab7442ef752afc25f673ead46f122397ac617ebb0a64defe8387ca3b997,"{""b"":""Y_braid"",""d"":15,""noise"":""SI1000"",""p"":0.001,""q"":470,""r"":10,""rb"":7}"
    127343,      1006,         0,   0.541,internal_correlated,37161e640388a03612eb36de1a043b63caa0017705ba87b28bcd569f33526b72,"{""b"":""Y_folded"",""d"":3,""noise"":""SI1000"",""p"":0.001,""q"":23,""r"":3,""rb"":0}"
     15501,      1085,         0,   0.184,internal_correlated,b6ce8ad682fb54e70e053bdfc33fc69b0cd65dcf63751c572e7bed29878ac742,"{""b"":""Y_folded"",""d"":3,""noise"":""SI1000"",""p"":0.003,""q"":23,""r"":3,""rb"":0}"
    599400,      1010,         0,    5.25,internal_correlated,5278c3ca53591f19a312c0732fac7a59939e383d367460415540791fe9e3a72d,"{""b"":""Y_folded"",""d"":5,""noise"":""SI1000"",""p"":0.001,""q"":59,""r"":5,""rb"":0}"
     21182,      1032,         0,    1.28,internal_correlated,86abff9eac6ef99bb9f3fd2ecc966e43fd520891f1e66e5db5ade2824193e437,"{""b"":""Y_folded"",""d"":5,""noise"":""SI1000"",""p"":0.003,""q"":59,""r"":5,""rb"":0}"
   3479702,      1003,         0,    80.0,internal_correlated,693d194b524598ec61cde014cddd61b54beb4f522f25d16b6b40946f152dae61,"{""b"":""Y_folded"",""d"":7,""noise"":""SI1000"",""p"":0.001,""q"":111,""r"":7,""rb"":0}"
     31150,      1004,         0,    5.46,internal_correlated,59ae1b8baea67f823b8a789e579f7ec84ff2033b27932eb0cfa4952a2ba2d190,"{""b"":""Y_folded"",""d"":7,""noise"":""SI1000"",""p"":0.003,""q"":111,""r"":7,""rb"":0}"
  25002182,      1060,         0,  1216.0,internal_correlated,503954550cf458ddd35686b16ce49a50dbcdf570cd765258f18ec45a1dffe5d5,"{""b"":""Y_folded"",""d"":9,""noise"":""SI1000"",""p"":0.001,""q"":179,""r"":9,""rb"":0}"
     55200,      1132,         0,    23.4,internal_correlated,4a90faec5fbceedc587a0e1853d940217459b4a48bda055c2037e5e0e80e5fb6,"{""b"":""Y_folded"",""d"":9,""noise"":""SI1000"",""p"":0.003,""q"":179,""r"":9,""rb"":0}"
 151816899,      1010,         0, 15503.2,internal_correlated,202063a932839339acc7e1d01efa217928339fee433fc38abf789a932d8a4cf2,"{""b"":""Y_folded"",""d"":11,""noise"":""SI1000"",""p"":0.001,""q"":263,""r"":11,""rb"":0}"
     79277,      1008,         0,    81.3,internal_correlated,7e44f15a58659cbea15bda4a9e70d8d253ae53b3d429c408c6daadb6b91227e8,"{""b"":""Y_folded"",""d"":11,""noise"":""SI1000"",""p"":0.003,""q"":263,""r"":11,""rb"":0}"
1000000000,       966,         0,189429.7,internal_correlated,9bf6a8736a0eb0e1cfe2403ade89a28bfbce7f226b1ecf9cc489f83a138033bf,"{""b"":""Y_folded"",""d"":13,""noise"":""SI1000"",""p"":0.001,""q"":363,""r"":13,""rb"":0}"
    123451,      1020,         0,   253.4,internal_correlated,9a4a9380d10aa558b3b1fce074dbef5ec8293f4fa44d17fd7e93b6e4ab528946,"{""b"":""Y_folded"",""d"":13,""noise"":""SI1000"",""p"":0.003,""q"":363,""r"":13,""rb"":0}"
1000000000,       128,         0,333836.9,internal_correlated,36375614630472a26a7f1e95e766f06ecc07381b96467ce9837c4decf1020319,"{""b"":""Y_folded"",""d"":15,""noise"":""SI1000"",""p"":0.001,""q"":479,""r"":15,""rb"":0}"
    233795,      1062,         0,   903.4,internal_correlated,ed8a4aeecb222a3074cf4aa327ba9f9b2cc136e55c9c4b93e47050136725cad4,"{""b"":""Y_folded"",""d"":15,""noise"":""SI1000"",""p"":0.003,""q"":479,""r"":15,""rb"":0}"
1000000000,        23,         0,560357.7,internal_correlated,7fe1e53751e38574c03a0bf81f3098bd8e93f23981f76c396d1f9327cd82e2a7,"{""b"":""Y_folded"",""d"":17,""noise"":""SI1000"",""p"":0.001,""q"":611,""r"":17,""rb"":0}"
    361767,      1036,         0,  2307.9,internal_correlated,314251ff8d6b2bb619a2b75844f38139677218cd2a1bdc4ff50d52aa6e309923,"{""b"":""Y_folded"",""d"":17,""noise"":""SI1000"",""p"":0.003,""q"":611,""r"":17,""rb"":0}"
    256916,      1013,         0,   0.417,internal_correlated,5116ca9c21710b56f945ae7d3c84893937af98b2af58333ddf6a5fa0487d3968,"{""b"":""Y_magic_idle"",""d"":3,""noise"":""SI1000"",""p"":0.001,""q"":17,""r"":3,""rb"":0}"
     31251,      1016,         0,   0.109,internal_correlated,7e328a620f8a2a4f9bf77c8cbdd2a677b7be934881a397f7faf206d3e6ee656d,"{""b"":""Y_magic_idle"",""d"":3,""noise"":""SI1000"",""p"":0.003,""q"":17,""r"":3,""rb"":0}"
   2504393,      1018,         0,    6.75,internal_correlated,3cfb2332045dad2a2748050305bbb16505e9e66d0c7b524f3f0cea73b6f8eaea,"{""b"":""Y_magic_idle"",""d"":5,""noise"":""SI1000"",""p"":0.001,""q"":49,""r"":3,""rb"":0}"
     89347,      1004,         0,   0.881,internal_correlated,4889fcf2ee545c472974149d80f9af9355e696087c90df546940c8d91b24cbce,"{""b"":""Y_magic_idle"",""d"":5,""noise"":""SI1000"",""p"":0.003,""q"":49,""r"":3,""rb"":0}"
  30119861,      1095,         0,   165.2,internal_correlated,c0220e1650b33e5f1c5cdea0ef102a3d45769287701fb58b439746554f12f54b,"{""b"":""Y_magic_idle"",""d"":7,""noise"":""SI1000"",""p"":0.001,""q"":97,""r"":3,""rb"":0}"
    289749,      1021,         0,    6.60,internal_correlated,bfceb38f4fbec34e1a6e05b2cbceff790d749bf5b9424dc0980c1bfaf407d27a,"{""b"":""Y_magic_idle"",""d"":7,""noise"":""SI1000"",""p"":0.003,""q"":97,""r"":3,""rb"":0}"
 333048840,      1011,         0,  3308.6,internal_correlated,0cbf85912a06cda6324100873e020ea6f8b667d42ba3c3706ecb0d9a0fdd33fa,"{""b"":""Y_magic_idle"",""d"":9,""noise"":""SI1000"",""p"":0.001,""q"":161,""r"":3,""rb"":0}"
    892323,      1014,         0,    41.7,internal_correlated,fbbb6aaa1da86cfb3f26ef7caa09bc6acd7e2d06ca41131a8645784b7a72d22e,"{""b"":""Y_magic_idle"",""d"":9,""noise"":""SI1000"",""p"":0.003,""q"":161,""r"":3,""rb"":0}"
1000000000,       250,         0, 16501.2,internal_correlated,269345f9715ef9214a1068494ced9041bcdc0f2bf5139957cbb92a266763a8e0,"{""b"":""Y_magic_idle"",""d"":11,""noise"":""SI1000"",""p"":0.001,""q"":241,""r"":3,""rb"":0}"
   2846508,      1007,         0,   231.6,internal_correlated,c7fb4ff8eb366f26cd4134c14dadcf799ab9071882083a74e7d10dfa7d807545,"{""b"":""Y_magic_idle"",""d"":11,""noise"":""SI1000"",""p"":0.003,""q"":241,""r"":3,""rb"":0}"
1000000000,        23,         0, 25894.2,internal_correlated,d2c9db0c54f9145e8318857f9c87463e7af42d06e5ee4a985c13396beef3d9d6,"{""b"":""Y_magic_idle"",""d"":13,""noise"":""SI1000"",""p"":0.001,""q"":337,""r"":3,""rb"":0}"
   9022085,      1005,         0,  1149.3,internal_correlated,f16e59004c9e8c310461c4a38688a8cf5328b8638dd21819eb16196da87824c6,"{""b"":""Y_magic_idle"",""d"":13,""noise"":""SI1000"",""p"":0.003,""q"":337,""r"":3,""rb"":0}"
1000000000,         0,         0, 32711.7,internal_correlated,1b8a09ddff1440b0e7f36fa4f44ae021308f32273b3cfeffba6361088b047013,"{""b"":""Y_magic_idle"",""d"":15,""noise"":""SI1000"",""p"":0.001,""q"":449,""r"":3,""rb"":0}"
  30087635,      1028,         0,  5701.8,internal_correlated,7d7b1b2407dd947b86ca8ac6855eeeb52b1e6b860fd2dde21458c13a056308b4,"{""b"":""Y_magic_idle"",""d"":15,""noise"":""SI1000"",""p"":0.003,""q"":449,""r"":3,""rb"":0}"
1000000000,         0,         0, 40341.7,internal_correlated,db853bff4b22c75a0cb80bf93d26d7aa250a2c90850c3ea020e5cec0eca7e58c,"{""b"":""Y_magic_idle"",""d"":17,""noise"":""SI1000"",""p"":0.001,""q"":577,""r"":3,""rb"":0}"
  94732012,      1010,         0, 24959.5,internal_correlated,b4229de8b97635559beb2ad25e8106f17806a5f3ae015a70db355806e38cb1a4,"{""b"":""Y_magic_idle"",""d"":17,""noise"":""SI1000"",""p"":0.003,""q"":577,""r"":3,""rb"":0}"
    240132,      1102,         0,   0.538,internal_correlated,52f3ff11aa696dba0a91aa8a02d544f85fd567df6a90d07fd12d8520e4673e5f,"{""b"":""Y_magic_transition"",""d"":3,""noise"":""SI1000"",""p"":0.001,""q"":19,""r"":1,""rb"":1}"
     32708,      1053,         0,   0.335,internal_correlated,bc2c400430458ed9f0aea0949939103c869dea7bfcd822eb5d029a027e087a1e,"{""b"":""Y_magic_transition"",""d"":3,""noise"":""SI1000"",""p"":0.003,""q"":19,""r"":1,""rb"":1}"
   1490343,      1005,         0,    4.93,internal_correlated,d8528c9f1fc3f9d86bd76231bce83fa0bfdc1a67b15ab8fe3b60d5fd33107c94,"{""b"":""Y_magic_transition"",""d"":5,""noise"":""SI1000"",""p"":0.001,""q"":53,""r"":1,""rb"":1}"
     70139,      1003,         0,    1.04,internal_correlated,e97ca48c3ebf54289889db478de0851781a161c0b46726fcb7db229b05dcdd5e,"{""b"":""Y_magic_transition"",""d"":5,""noise"":""SI1000"",""p"":0.003,""q"":53,""r"":1,""rb"":1}"
  14046013,      1002,         0,    93.6,internal_correlated,1a6b1726de6e6663372cc51eca6148d03e321a1308b12d32fee7c81276a68e14,"{""b"":""Y_magic_transition"",""d"":7,""noise"":""SI1000"",""p"":0.001,""q"":103,""r"":1,""rb"":1}"
    181395,      1078,         0,    5.66,internal_correlated,277a030b27dbf65b0079219a1e1c324cdfc7a22358efd662336fd0d7c8ecf0f4,"{""b"":""Y_magic_transition"",""d"":7,""noise"":""SI1000"",""p"":0.003,""q"":103,""r"":1,""rb"":1}"
 116265866,      1001,         0,  1437.5,internal_correlated,9cc25f43bc04db42fe64140be2e85c0c5691b55f02aa5f7857d0c251217a6327,"{""b"":""Y_magic_transition"",""d"":9,""noise"":""SI1000"",""p"":0.001,""q"":169,""r"":1,""rb"":1}"
    440932,      1040,         0,    28.5,internal_correlated,8998994beac3ee9c4f2ba783f35f7ae93227272b4261efb9f61b0d4faf231536,"{""b"":""Y_magic_transition"",""d"":9,""noise"":""SI1000"",""p"":0.003,""q"":169,""r"":1,""rb"":1}"
1000000000,       851,         0, 17623.4,internal_correlated,c320d0d15f5b2ef128f7767bd197008e7cb321202538d246dd99257cbf8046dc,"{""b"":""Y_magic_transition"",""d"":11,""noise"":""SI1000"",""p"":0.001,""q"":251,""r"":1,""rb"":1}"
   1097791,      1000,         0,   125.0,internal_correlated,8617e3fdf3eaa5e33814ae671640dee231f64339473f47c1465481f8814164bc,"{""b"":""Y_magic_transition"",""d"":11,""noise"":""SI1000"",""p"":0.003,""q"":251,""r"":1,""rb"":1}"
1000000000,       113,         0, 26130.9,internal_correlated,737b19159582d6845757ffe8e693332e240ca8f5d4af16fc7bcdcf6e675ce049,"{""b"":""Y_magic_transition"",""d"":13,""noise"":""SI1000"",""p"":0.001,""q"":349,""r"":1,""rb"":1}"
   3290230,      1059,         0,   604.6,internal_correlated,1ad98591f172ac2048b662a13b19182a17877326bc4d6e3f0d50844bb0e7ddeb,"{""b"":""Y_magic_transition"",""d"":13,""noise"":""SI1000"",""p"":0.003,""q"":349,""r"":1,""rb"":1}"
1000000000,        11,         0, 35175.4,internal_correlated,00d5cc3d84ed2532463b74bba5a4a9e87cadb5f930d2c900810e8e6358e98b24,"{""b"":""Y_magic_transition"",""d"":15,""noise"":""SI1000"",""p"":0.001,""q"":463,""r"":1,""rb"":1}"
   9019756,      1076,         0,  2491.3,internal_correlated,5f43f01df411d6b638cc1d31dd1eef52f9c49763ab55dfa975eb8a6fec366e13,"{""b"":""Y_magic_transition"",""d"":15,""noise"":""SI1000"",""p"":0.003,""q"":463,""r"":1,""rb"":1}"
1000000000,         2,         0, 46898.2,internal_correlated,89283a2db44a118add7b78f91e1c546988379087eb02963748b0fead6cce0ed6,"{""b"":""Y_magic_transition"",""d"":17,""noise"":""SI1000"",""p"":0.001,""q"":593,""r"":1,""rb"":1}"
  24694269,      1056,         0,  9416.1,internal_correlated,ad488bf6cf8638236f28da39acc524999f7ea0149553edfaeb49836661e7a42f,"{""b"":""Y_magic_transition"",""d"":17,""noise"":""SI1000"",""p"":0.003,""q"":593,""r"":1,""rb"":1}"
    301919,      1019,         0,   0.665,internal_correlated,8eb0c9384d67af89df0aa3d76b6a70cb9f6c7270b2503b66e5c96de0d21900e2,"{""b"":""Z"",""d"":3,""noise"":""SI1000"",""p"":0.001,""q"":17,""r"":3,""rb"":0}"
     35968,      1000,         0,   0.116,internal_correlated,371f8a6f8cfcc603e6cbf9920c3bdfc024fbdea886652f9fa9f80a14b4216768,"{""b"":""Z"",""d"":3,""noise"":""SI1000"",""p"":0.003,""q"":17,""r"":3,""rb"":0}"
   1377748,      1014,         0,    5.81,internal_correlated,38b9bdbbfb214933cbcf2283e870d4ff1f0937da01cd97a6c999b2ca09268380,"{""b"":""Z"",""d"":5,""noise"":""SI1000"",""p"":0.001,""q"":49,""r"":5,""rb"":0}"
     57319,      1094,         0,    1.57,internal_correlated,4e1f1536d71c1d26107a7d88c01ff661e9962860e3c3b3724bf59639a8019a96,"{""b"":""Z"",""d"":5,""noise"":""SI1000"",""p"":0.003,""q"":49,""r"":5,""rb"":0}"
   8289257,      1043,         0,    97.3,internal_correlated,5ba8a8d81599ded112284f53919bf21f98f3a550a03858f555bab84ee9758308,"{""b"":""Z"",""d"":7,""noise"":""SI1000"",""p"":0.001,""q"":97,""r"":7,""rb"":0}"
     82861,      1006,         0,    5.49,internal_correlated,8e75a474b0da0d44a84e52696ff0ef8e58d6d90605ea2016dec937412c9f6b5e,"{""b"":""Z"",""d"":7,""noise"":""SI1000"",""p"":0.003,""q"":97,""r"":7,""rb"":0}"
  56701221,      1002,         0,  1221.3,internal_correlated,026fda95f5fc8e684f44986eb9f007feaa20675a0eb3a00893ce0392eb7216a1,"{""b"":""Z"",""d"":9,""noise"":""SI1000"",""p"":0.001,""q"":161,""r"":9,""rb"":0}"
    137262,      1052,         0,    24.4,internal_correlated,5c310568d89c0b40eb25b7b09dcb4585443eb49acbd05637609a2bfcb44cbc77,"{""b"":""Z"",""d"":9,""noise"":""SI1000"",""p"":0.003,""q"":161,""r"":9,""rb"":0}"
 404336647,      1027,         0, 25180.9,internal_correlated,73854575a5894ce382955a6733f133a640567f1f6c7df412ac1c6603e1f80f9d,"{""b"":""Z"",""d"":11,""noise"":""SI1000"",""p"":0.001,""q"":241,""r"":11,""rb"":0}"
    247879,      1031,         0,   116.3,internal_correlated,aa0a87b419778d866791dfa2ff2b88a993fca12c8906180951d9c5b85a466569,"{""b"":""Z"",""d"":11,""noise"":""SI1000"",""p"":0.003,""q"":241,""r"":11,""rb"":0}"
1000000000,       348,         0,125019.6,internal_correlated,e14b6e1aa7d8959e438c997e4267e0be5e552afd0eda00c794e66d1ba52b1a79,"{""b"":""Z"",""d"":13,""noise"":""SI1000"",""p"":0.001,""q"":337,""r"":13,""rb"":0}"
    410358,      1052,         0,   428.9,internal_correlated,0813cf87c76af49f7b4af80ab9924402c1509a9247b5ee77466debf996d1cfac,"{""b"":""Z"",""d"":13,""noise"":""SI1000"",""p"":0.003,""q"":337,""r"":13,""rb"":0}"
1000000000,        38,         0,223637.0,internal_correlated,c1dbc4025ffb5d11fdbef936067b0490400a5f6831c42fd975be29869fff57a7,"{""b"":""Z"",""d"":15,""noise"":""SI1000"",""p"":0.001,""q"":449,""r"":15,""rb"":0}"
    732717,      1073,         0,  1517.7,internal_correlated,a3a4e7830f4afc98de49e8a2c0eba08a420586077962902073913f9c24effca5,"{""b"":""Z"",""d"":15,""noise"":""SI1000"",""p"":0.003,""q"":449,""r"":15,""rb"":0}"
1000000000,         8,         0,391773.6,internal_correlated,3fb7e69f102a47aa7e834acf3130939532341271744acedda943f7cb5d5dd0e2,"{""b"":""Z"",""d"":17,""noise"":""SI1000"",""p"":0.001,""q"":577,""r"":17,""rb"":0}"
   1213416,      1009,         0,  4415.9,internal_correlated,14d0e1309b0ac05823a2f1b4fcacfc49f814e28cef8f94bf6bd4832eb8fce61f,"{""b"":""Z"",""d"":17,""noise"":""SI1000"",""p"":0.003,""q"":577,""r"":17,""rb"":0}"
    496350,      1008,         0,   0.667,internal_correlated,2e06a2025e4c3235d2cf2c77ace37fb69f973cf2dbbe92ffde8949ff5855b287,"{""b"":""Z_magic_idle"",""d"":3,""noise"":""SI1000"",""p"":0.001,""q"":17,""r"":3,""rb"":0}"
     57979,      1001,         0,   0.189,internal_correlated,780066f2bbc0a0d6f3bfadb218838d4106e60985d93b96cb9617ce44f602f272,"{""b"":""Z_magic_idle"",""d"":3,""noise"":""SI1000"",""p"":0.003,""q"":17,""r"":3,""rb"":0}"
   4743459,      1003,         0,    14.1,internal_correlated,81050a08cd521cd95f5c6b6d68d2f25e8c1e40db93b7caab749df561f7902b27,"{""b"":""Z_magic_idle"",""d"":5,""noise"":""SI1000"",""p"":0.001,""q"":49,""r"":3,""rb"":0}"
    178794,      1015,         0,    1.83,internal_correlated,3be055222050f649a07b5824347e2d0ca5f2117e2c1a2d0e24faea75be664519,"{""b"":""Z_magic_idle"",""d"":5,""noise"":""SI1000"",""p"":0.003,""q"":49,""r"":3,""rb"":0}"
  55221433,      1008,         0,   207.7,internal_correlated,963bccbaa7fa4f515b56011fcf6c285e08336faa455c31ecd7aac67bfc6b59b7,"{""b"":""Z_magic_idle"",""d"":7,""noise"":""SI1000"",""p"":0.001,""q"":97,""r"":3,""rb"":0}"
    620439,      1138,         0,    15.1,internal_correlated,6b1a79316f053c674465bac9f23bce71ec8a1f85da5937f7d5653d9919cab8b3,"{""b"":""Z_magic_idle"",""d"":7,""noise"":""SI1000"",""p"":0.003,""q"":97,""r"":3,""rb"":0}"
 685195822,      1007,         0,  3853.0,internal_correlated,46b7a23fbe89aa11c325df06ec19cf79f7b4d95e69841c7e72ab98985dc9184e,"{""b"":""Z_magic_idle"",""d"":9,""noise"":""SI1000"",""p"":0.001,""q"":161,""r"":3,""rb"":0}"
   1726128,      1002,         0,    67.5,internal_correlated,0bb8fefd4423e6691c3fcd1db07f6daebc79557b95aa0279724cc8291cf27c8f,"{""b"":""Z_magic_idle"",""d"":9,""noise"":""SI1000"",""p"":0.003,""q"":161,""r"":3,""rb"":0}"
1000000000,       123,         0,  9460.5,internal_correlated,950c4323f7065ea88812a231b82826419371d2fc9ffc0a34e5cea56e27796b63,"{""b"":""Z_magic_idle"",""d"":11,""noise"":""SI1000"",""p"":0.001,""q"":241,""r"":3,""rb"":0}"
   5731501,      1004,         0,   536.2,internal_correlated,f52da33ac1f91b4cbc82edfbeaba1abbf8eb26cf5b4446fba6d7fad605139662,"{""b"":""Z_magic_idle"",""d"":11,""noise"":""SI1000"",""p"":0.003,""q"":241,""r"":3,""rb"":0}"
1000000000,        15,         0, 13814.8,internal_correlated,e00b914fc540dae2428aa2378a26f4e7eab1e7de21c654aab4beb0207339054a,"{""b"":""Z_magic_idle"",""d"":13,""noise"":""SI1000"",""p"":0.001,""q"":337,""r"":3,""rb"":0}"
  18170410,      1011,         0,  2747.7,internal_correlated,26e50a6c12eff29d88fb52002441e7391add4b1a62265f337cd431cc28d98687,"{""b"":""Z_magic_idle"",""d"":13,""noise"":""SI1000"",""p"":0.003,""q"":337,""r"":3,""rb"":0}"
1000000000,         0,         0, 19315.0,internal_correlated,387e08dfb3a6ea71f8514924a9ea31509f920c9e74fb460a188b13707b435ded,"{""b"":""Z_magic_idle"",""d"":15,""noise"":""SI1000"",""p"":0.001,""q"":449,""r"":3,""rb"":0}"
  63036170,      1012,         0, 13033.0,internal_correlated,ded3697c2b36dc8e0bf54aea15df70579744ab9d26932de132d49938b3f502cb,"{""b"":""Z_magic_idle"",""d"":15,""noise"":""SI1000"",""p"":0.003,""q"":449,""r"":3,""rb"":0}"
1000000000,         0,         0, 24699.3,internal_correlated,dbf55de8baa0b276bcb41cca6f8dd46d9ceaf5560026274670a57d3fd4cfb4a2,"{""b"":""Z_magic_idle"",""d"":17,""noise"":""SI1000"",""p"":0.001,""q"":577,""r"":3,""rb"":0}"
 206631518,      1003,         0, 34428.7,internal_correlated,915f09f676896ad8be009fd55dc9fac5fbde2ae8dd7d47066d7f570046764cc4,"{""b"":""Z_magic_idle"",""d"":17,""noise"":""SI1000"",""p"":0.003,""q"":577,""r"":3,""rb"":0}"
""".strip()


# Data from table 1 of https://arxiv.org/abs/2409.07707
lee_et_al_2024_csv = """
                 Scheme, Output infidelity, Failure rate, Space cost, Time cost, Effective spacetime cost,   Tm,   Tintv, Tidle
"sng-(17,8,12,7)",                 1.72E-5,      5.37E-2,       2149,       512,                   1.16E6,     ,        ,
"sng-(23,16,14,9)",                1.13E-6,      2.52E-2,       3701,       640,                   2.43E6,     ,        ,
"sng-(29,20,18,11)",               1.27E-7,      1.82E-2,       5909,       768,                   4.62E6,     ,        ,
"sng-(47,24,26,23)",               3.58E-8,      1.51E-2,     1.31E4,      1536,                   2.04E7,     ,        ,
"cmb-(47,26,30,17,7,5)",           1.15E-8,      6.99E-3,     3.06E4,    2.28E5,                   7.03E9,    3,    3559,   693
"cmb-(53,38,34,19,7,5)",           1.04E-9,      4.21E-3,     3.99E4,    2.29E5,                   9.18E9,    3,    3579,   674
"cmb-(67,38,34,33,7,5)",          1.82E-10,      2.59E-3,     7.06E4,    2.26E5,                  1.60E10,    3,    3531,   638
""".strip()


def parse_field(f: Any) -> int | float | str:
    if isinstance(f, str):
        try:
            return int(f)
        except ValueError:
            try:
                return float(f)
            except ValueError:
                return f
    if isinstance(f, float):
        if f == int(f):
            return int(f)
    if isinstance(f, (int, float)):
        return f
    return str(f)


def select_rep_stats(stats: list[sinter.TaskStats], circuit: stim.Circuit) -> list[tuple[sinter.TaskStats, float]]:
    stats = [stat for stat in stats if stat.shots > stat.discards]
    vols = []
    errs = []
    baseline = cultiv.compute_expected_injection_growth_volume(
        circuit,
        discard_rate=0,
    )
    for stat in stats:
        keep_rate = (stat.shots - stat.discards) / stat.shots
        vols.append(baseline / keep_rate)
        errs.append(stat.errors / (stat.shots - stat.discards))
    indices = sorted(range(len(stats)), key=lambda e: (errs[e], vols[e]))
    vols = [vols[k] for k in indices]
    errs = [errs[k] for k in indices]
    stats = [stats[k] for k in indices]

    result = []
    prev_vol = None
    for k in range(len(stats)):
        if prev_vol is not None and vols[k] > prev_vol * 0.9 and errs[k] != 0:
            continue
        prev_vol = vols[k]
        result.append((stats[k], vols[k]))

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", nargs='+', type=str, default=())
    args = parser.parse_args()
    new_stats = sinter.read_stats_from_csv_files(*getattr(args, 'in'))
    print(sinter.CSV_HEADER)

    k = 0
    reader = csv.DictReader(io.StringIO(lee_et_al_2024_csv))
    reader.fieldnames = [e.strip() for e in reader.fieldnames]
    for item in reader:
        error_rate = float(item['Output infidelity'].strip())
        discard_rate = float(item['Failure rate'].strip())
        qubits = int(float(item['Space cost']))

        # > The time cost is quantified by the number of
        # > time steps required for each attempt of the scheme. Note
        # > that a single syndrome extraction round can be done in
        # > eight time steps by selecting the entangling gate schedule
        # > appropriate"
        TIME_ADJUSTMENT_FACTOR = 8
        rounds = int(float(item['Time cost'])) / TIME_ADJUSTMENT_FACTOR
        volume = float(item['Effective spacetime cost'].strip()) / TIME_ADJUSTMENT_FACTOR

        print(sinter.TaskStats(
            strong_id=f'''anon{k}''',
            decoder="estimated",
            json_metadata={
                'q': qubits,
                'r': rounds,
                'p': 1e-3,
                'noise': 'uniform',
                'c': '2024 Lee et al',
                'v': volume,
                'state': 'T',
            },
            shots=round(100 / error_rate / (1 - discard_rate)),
            errors=100,
            discards=round(100 / error_rate / (1 - discard_rate) * discard_rate),
        ))
        k += 1

    def s2p(stat: sinter.TaskStats) -> complex:
        q = stat.json_metadata['post_q']
        r = stat.json_metadata['post_r']
        assert r
        x = q * r * (stat.shots + 2) / (stat.shots - stat.discards + 1)
        y = stat.errors / (stat.shots - stat.discards)
        return x + 1j*y

    for group in sinter.group_by(sinter.read_stats_from_csv_files(io.StringIO(hook_data_csv)), key=lambda stat: stat.json_metadata['b']).values():
        points = [s2p(stat) for stat in group]
        for stat in group:
            p = s2p(stat)
            if any(q.real < p.real and q.imag < p.imag * 1.1 for q in points):
                continue
            row = stat.json_metadata
            if row['b'] == 'hook_inject_Y_magic_verify':
                c = '2023 Gidney'
            elif row['b'] == 'li_inject_Y_magic_verify':
                c = '2014 Li'
            elif row['b'] == 'zz_tweaked_inject_Y_magic_verify':
                c = '2022 Singh et al'
            else:
                continue
            print(sinter.TaskStats(
                strong_id=f'''anon{k}''',
                decoder="estimated",
                json_metadata={
                    'q': row['post_q'],
                    'r': row['post_r'],
                    'p': row['p'],
                    'noise': row['noise'],
                    'c': c,
                    'lbl': '',
                    'state': 'T',
                },
                shots=stat.shots,
                errors=stat.errors,
                discards=stat.discards,
            ))
            k += 1

    for row in csv.DictReader(io.StringIO(litinsky_data_csv)):
        row = {k: parse_field(v) for k, v in row.items()}
        print(sinter.TaskStats(
            strong_id=f'''anon{k}''',
            decoder="estimated",
            json_metadata={
                'q': row['qubits'],
                'r': row['cycles'] / (2 if row['state'] == '2T' else 1),
                'p': row['p_phys'],
                'noise': 'uniform',
                'c': row['c'],
                'lbl': row['lbl'],
                'state': 'T' if row['state'] == '2T' else row['state'],
            },
            shots=round(10 / row['p_out'] * (2 if row['state'] == '2T' else 1)),
            errors=10,
            discards=0,
        ))
        k += 1

    # Cryptography cost reference lines.
    # for v in [2, 5e8]:
    #     print(sinter.TaskStats(
    #         strong_id=f'''ref{k}''',
    #         decoder="estimated",
    #         json_metadata={'q': 1, 'r': v, 'p': 1e-3, 'noise': 'SI1000', 'c': f'''Sufficient to break RSA2048 (Gidney et al 2019)''', 'state': 'CCZ'},
    #         shots=100000000000,
    #         errors=round(100000000000/2.7e9),
    #         discards=0,
    #     ))
    #     k += 1
    # for v in [2, 5e8]:
    #     print(sinter.TaskStats(
    #         strong_id=f'''ref{k}''',
    #         decoder="estimated",
    #         json_metadata={'q': 1, 'r': v, 'p': 1e-3, 'noise': 'SI1000', 'c': f'''Sufficient to break ECDLP256 (Häner et al 2020)''', 'state': 'T'},
    #         shots=100000000000,
    #         errors=round(100000000000/(2**34 * 1.4)),
    #         discards=0,
    #     ))
    #     k += 1

    # Inplace Y reference (actual data).
    for row in csv.DictReader(io.StringIO(y_data_csv)):
        row = {k: parse_field(v) for k, v in row.items()}
        row.update(json.loads(row['json_metadata']))
        if row['b'] == 'Y' and row['p'] == 1e-3 and row['rb'] == row['d'] // 2 and row['r'] == row['d']:
            print(sinter.TaskStats(
                strong_id=f'''ref{k}''',
                decoder=row['decoder'],
                json_metadata={
                    'q': row['q'],
                    'r': row['rb'] + 1,
                    'p': row['p'],
                    'noise': row['noise'],
                    'c': f'''Preparing S|+⟩, for comparison''',
                    'lbl': '',
                    'state': 'Y',
                },
                shots=row['shots'] * 2,  # *2 to counter double-counting errors due to RY and MY both being in the experiment
                errors=row['errors'],
                discards=row['discards'],
            ))
        else:
            continue
        k += 1
    # Inplace Y reference (extrapolated).
    k += 1
    for v in range(19, 50, 2):
        print(sinter.TaskStats(
            strong_id=f'''ref{k}''',
            decoder="projected",
            json_metadata={
                'q': v * v * 2 + v - 1,
                'r': v // 2 + 1,
                'p': 1e-3,
                'noise': 'SI1000',
                'c': f'''Preparing S|+⟩, for comparison''',
                'state': 'Y',
                'projected': 1,
            },
            shots=round(2.5**v * 13),
            errors=1,
            discards=0,
        ))
        k += 1

    DOUBLING_S_GIVES_T_ASSUMPTION = 2
    match_stats = [
        stat
        for stat in new_stats
        if stat.decoder == 'perfectionist'
        if stat.json_metadata.get('d1') == 3
        if stat.json_metadata.get('c') == 'inject[unitary]+cultivate'
        if stat.json_metadata.get('p') == 1e-3
        if stat.json_metadata.get('noise') == 'uniform'
    ]
    if len(match_stats) == 1:
        stat, = match_stats
        print(sinter.TaskStats(
            strong_id=f'''ref{k}''',
            json_metadata={**stat.json_metadata, 'c': f'''2024 This Work (d1=3)''', 'state': 'T[d=3]'},
            decoder=stat.decoder,
            shots=stat.shots,
            errors=stat.errors * DOUBLING_S_GIVES_T_ASSUMPTION,
            discards=stat.discards,
        ))
        k += 1
    else:
        print(f"WARNING: found {len(match_stats)} instead of one d1=3 c=inject[unitary]+cultivate p=1e-3 for historical data plot", file=sys.stderr)

    match_stats = [
        stat
        for stat in new_stats
        if stat.decoder == 'perfectionist'
        if stat.json_metadata.get('d1') == 5
        if stat.json_metadata.get('c') == 'inject[unitary]+cultivate'
        if stat.json_metadata.get('p') == 1e-3
        if stat.json_metadata.get('noise') == 'uniform'
    ]
    if len(match_stats) == 1:
        stat, = match_stats
        print(sinter.TaskStats(
            strong_id=f'''ref{k}''',
            json_metadata={**stat.json_metadata, 'c': f'''2024 This Work (d1=5)''', 'state': 'T[d=5]'},
            decoder=stat.decoder,
            shots=stat.shots,
            errors=stat.errors * DOUBLING_S_GIVES_T_ASSUMPTION,
            discards=stat.discards,
        ))
        k += 1
    else:
        print("WARNING: didn't find d1=5 c=unitary p=1e-3 for historical data plot", file=sys.stderr)

    for p in [5e-4, 1e-3, 2e-3]:
        c3 = cultiv.make_end2end_cultivation_circuit(dcolor=3, dsurface=15, basis='Y', r_growing=3, r_end=1,
                                                     inject_style='unitary')
        match_stats = [
            stat
            for stat in new_stats
            if stat.decoder == 'desaturation'
            if stat.json_metadata.get('d1') == 3
            if stat.json_metadata.get('d2') == 15
            if stat.json_metadata.get('c') == 'end2end-inplace-distillation'
            if stat.json_metadata.get('p') == p
            if stat.json_metadata.get('r1') == 3
            if stat.json_metadata.get('r2') == 5
            if stat.json_metadata.get('noise') == 'uniform'
        ]
        if len(match_stats) == 1:
            gs = cultiv.stat_to_gap_stats(match_stats, rounding=1, func=lambda arg: sinter.AnonTaskStats(
                shots=arg.source.shots,
                discards=arg.at_least.discards + arg.less.shots,
                errors=arg.at_least.errors,
            ))
            c3n = gen.NoiseModel.uniform_depolarizing(p).noisy_circuit_skipping_mpp_boundaries(c3)
            for stat, cur_vol in select_rep_stats(gs, c3n):
                print(sinter.TaskStats(
                    strong_id=f'''ref{k}''',
                    json_metadata={**stat.json_metadata, 'c': f'''2024 This Work (d1=3)''', 'state': 'T', 'v': cur_vol},
                    decoder=stat.decoder,
                    shots=stat.shots,
                    errors=stat.errors * DOUBLING_S_GIVES_T_ASSUMPTION,
                    discards=stat.discards,
                ))
                k += 1
        else:
            print(f"WARNING: didn't find d1=3 d2=15 p={p} for historical data plot", file=sys.stderr)

    for p in [5e-4, 1e-3, 2e-3]:
        c5 = cultiv.make_end2end_cultivation_circuit(
            dcolor=5,
            dsurface=15,
            basis='Y',
            r_growing=5,
            r_end=1,
            inject_style='unitary',
        )
        match_stats = [
            stat
            for stat in new_stats
            if stat.decoder == 'desaturation'
            if stat.json_metadata.get('d1') == 5
            if stat.json_metadata.get('d2') == 15
            if stat.json_metadata.get('c') == 'end2end-inplace-distillation'
            if stat.json_metadata.get('p') == p
            if stat.json_metadata.get('r1') == 5
            if stat.json_metadata.get('r2') == 5
            if stat.json_metadata.get('noise') == 'uniform'
        ]
        if len(match_stats) == 1:
            gs = cultiv.stat_to_gap_stats(match_stats, rounding=1, func=lambda arg: sinter.AnonTaskStats(
                shots=arg.source.shots,
                discards=arg.at_least.discards + arg.less.shots,
                errors=arg.at_least.errors,
            ))
            c5n = gen.NoiseModel.uniform_depolarizing(p).noisy_circuit_skipping_mpp_boundaries(c5)
            for stat, cur_vol in select_rep_stats(gs, c5n):
                print(sinter.TaskStats(
                    strong_id=f'''ref{k}''',
                    json_metadata={**stat.json_metadata, 'c': f'''2024 This Work (d1=5)''', 'state': 'T', 'v': cur_vol},
                    decoder=stat.decoder,
                    shots=stat.shots,
                    errors=stat.errors * DOUBLING_S_GIVES_T_ASSUMPTION,
                    discards=stat.discards,
                ))
                k += 1
        else:
            print(f"WARNING: didn't find d1=5 d2=15 p={p} for historical data plot", file=sys.stderr)

    # CNOT reference (actual data).
    for p in [1e-3]:
        match_groups = sinter.group_by([
            stat
            for stat in new_stats
            if stat.decoder == 'sparse_blossom_correlated'
            if stat.json_metadata.get('c') == 'surface-code-cnot'
            if stat.json_metadata.get('p') == p
            if stat.json_metadata.get('noise') == 'uniform'
        ], key=lambda e: e.json_metadata['d2'])
        available_cnot = set()
        for _, (stat_x, stat_z) in match_groups.items():
            assert {stat_x.json_metadata['b'], stat_z.json_metadata['b']} == {'X', 'Z'}
            assert stat_x.discards == stat_z.discards == 0
            err_x = stat_x.errors / stat_x.shots
            err_z = stat_z.errors / stat_z.shots
            err_xz = 1 - (1 - err_x) * (1 - err_z)
            q = stat_x.json_metadata['q']
            d = stat_x.json_metadata['d2']
            r = stat_x.json_metadata['r']
            assert q == 2*d*d*3 + 4*d - 1  # sanity check size
            assert r == 2*d + 4
            if min(stat_x.errors, stat_z.errors) < 3:
                continue
            available_cnot.add(d)
            shots = min(stat_x.shots, stat_z.shots)
            print(sinter.TaskStats(
                strong_id=f'''ref{k}''',
                json_metadata={
                    **stat_x.json_metadata,
                    'c': f'''Lattice Surgery CNOT, for comparison''',
                    'state': 'CNOT',
                    'v': q * r,
                    'b': 'XZ',
                },
                decoder=stat_x.decoder,
                shots=shots,
                discards=0,
                errors=round(shots * err_xz)
            ))
            k += 1
    # CNOT reference (extrapolated).
    k += 1
    for d in range(3, 50, 2):
        if d in available_cnot:
            continue
        projected_error = 1.68 / 10 ** (d * 0.559)
        q = 2*d*d*3 + 4*d - 1
        r = 2*d + 4
        print(sinter.TaskStats(
            strong_id=f'''ref{k}''',
            decoder="projected",
            json_metadata={
                'q': q,
                'r': r,
                'p': 1e-3,
                'noise': 'uniform',
                'c': f'''Lattice Surgery CNOT, for comparison''',
                'state': 'CNOT',
                'v': q * r,
                'b': 'XZ',
                'projected': 1,
            },
            shots=round(100 / projected_error),
            errors=100,
            discards=0,
        ))
        k += 1


if __name__ == '__main__':
    main()

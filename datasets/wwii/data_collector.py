import sys
import os

HOME, USER = os.getenv('HOME'), os.getenv('USER')
IMACCESS_PROJECT_WORKSPACE = os.path.join(HOME, "WS_Farid", "ImACCESS")

CLIP_DIR = os.path.join(IMACCESS_PROJECT_WORKSPACE, "clip")
sys.path.insert(0, CLIP_DIR)

MISC_DIR = os.path.join(IMACCESS_PROJECT_WORKSPACE, "misc")
sys.path.insert(0, MISC_DIR)

for p in sys.path:
	print(p)

from utils import *
import visualize as viz
from nlp_utils import get_enriched_description, validate_text_cleaning_pipeline
from data_prep import get_single_label_stratified_split

DATASET_BASE_URL = "https://www.worldwarphotos.info/gallery"
URLs = { # key: url : val: user_query
	f"{DATASET_BASE_URL}/usa/pacific/kwajalein/": None,
	f"{DATASET_BASE_URL}/usa/pacific/okinawa/": None,
	f"{DATASET_BASE_URL}/usa/pacific/peleliu/": None,
	f"{DATASET_BASE_URL}/usa/pacific/philippines/": None,
	f"{DATASET_BASE_URL}/usa/pacific/biak/": None, # small
	f"{DATASET_BASE_URL}/usa/pacific/makin/": None, # small
	f"{DATASET_BASE_URL}/usa/pacific/new-guinea/": None, # small
	f"{DATASET_BASE_URL}/usa/pacific/tarawa/": None,
	f"{DATASET_BASE_URL}/usa/pacific/gloucester/": None,
	f"{DATASET_BASE_URL}/usa/pacific/tinian/": None,
	f"{DATASET_BASE_URL}/usa/pacific/saipan/": None,
	f"{DATASET_BASE_URL}/usa/pacific/bougainville/": None,
	f"{DATASET_BASE_URL}/usa/pacific/eniwetok/": None,
	f"{DATASET_BASE_URL}/usa/pacific/guadalcanal/": None,
	f"{DATASET_BASE_URL}/usa/pacific/guam/": None,
	f"{DATASET_BASE_URL}/usa/pacific/iwo-jima/": None,
	f"{DATASET_BASE_URL}/usa/pacific/iwo-jima2/": None,
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/a-17/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/a-18/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/a-19/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/a-20-havoc-boston/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/a-20/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/a20/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/a-26/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/a-36/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/b-17/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/b-17-flying-fortress/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/b-17b/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/b-17g/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/b17/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/b-17raf/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/b-18/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/b-23/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/b-24/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/b-24-liberator/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/b-24-bomber/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/b24/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/b-25-mitchell/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/b-25/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/b25/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/b-26-marauder/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/b-29/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/b-32-dominator/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/bt/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/c-106/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/c-109/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/c-46/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/c-47/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/c47/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/c-54/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/c-69/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/c-73/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/c-76/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/c-87/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/f2a/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/f4f-wildcat/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/f4f/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/f4u-corsair/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/f4u/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/f6f-hellcat/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/f7f/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/f8f/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/fr1/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/lodestar/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/o-38/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/o-46/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/o-47/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/o-52/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/os2u/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/ose/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/p-26/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/p-35/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/p-36/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/p-38-lightning/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/p-38/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/p-39/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/p-39-2/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/p-40-warhawk/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/p-40raf/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/p-40/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/p-43/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/p-47/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/p-47-thunderbolt/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/p47/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/p-47d/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/p-51-mustang/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/p-51/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/p51-raf/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/p-59-airacomet/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/p-61/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/p-63/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/p-66/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/p-70/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/xp-75/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/p-80-shooting-star/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/pb2y/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/pb4y/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/pbm/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/pby/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/pby5/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/pq-14/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/pv/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/r-4/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/sb2a/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/sb2c/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/sb2u-vindicator/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/sbc/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/sbd/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/sbd1/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/sc-seahawk/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/so3c/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/soc/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/tbd/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/tbf/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/tbm/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/tbu-tby/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/uc-61/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/xa-21/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/xa-38/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/b-15/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/b-19/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/xb-38/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/xb-39/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/xb-42/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/xb-43/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/xf-12/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/xf8b/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/xfl/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/xp-42/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/xp-46/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/xp-54/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/xp-55/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/xp-56/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/xp-58/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/xp-83/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/xpb2m/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/pbb/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/tb2f/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/yb-40/": "aircraft",
	f"{DATASET_BASE_URL}/usa/aircrafts-2-3/yfm/": "aircraft",
	f"{DATASET_BASE_URL}/usa/armoured-vehicles-2/m12/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/usa/armoured-vehicles-2/m2-half-track/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/usa/armoured-vehicles-2/m3_halftrack-2/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/usa/armoured-vehicles-2/m3_scout/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/usa/armoured-vehicles-2/m31/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/usa/armoured-vehicles-2/m32/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/usa/armoured-vehicles-2/m7_priest/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/usa/armoured-vehicles-2/m8_greyhound/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/usa/tanks/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/usa/tanks/m1/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/usa/tanks/m10-wolverine/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/usa/tanks/m18-hellcat/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/usa/tanks/m2/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/usa/tanks/m2-medium/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/usa/tanks/m24/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/usa/tanks/m26/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/usa/tanks/m3_lee/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/usa/tanks/m3_stuart/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/usa/tanks/m3_m5/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/usa/tanks/m36-jackson/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/usa/tanks/m4_sherman/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/usa/tanks/m4-sherman-tank/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/usa/tanks/sherman/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/usa/tanks/sherman-tank/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/usa/tanks/m6-tank/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/usa/tanks/m8/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/usa/us-navy/": "naval ship",
	f"{DATASET_BASE_URL}/usa/vehicles/g506/": "military vehicles",
	f"{DATASET_BASE_URL}/usa/vehicles/m29/": "military vehicles",
	f"{DATASET_BASE_URL}/italy/spg2/75-18/" : "armored fighting vehicles", # https://en.wikipedia.org/wiki/Armoured_fighting_vehicle
	f"{DATASET_BASE_URL}/italy/spg2/l40/" : "armored fighting vehicles", # https://en.wikipedia.org/wiki/Armoured_fighting_vehicle
	f"{DATASET_BASE_URL}/france/tanks-france/" : "armored fighting vehicles", # French Tanks of World War II
	f"{DATASET_BASE_URL}/france/normandy-1944/": "normandy invasion", # Invasion of Normandy 1944 photo gallery
	f"{DATASET_BASE_URL}/japan/aircrafts/b7a/": "aircraft", #
	f"{DATASET_BASE_URL}/japan/aircrafts/d3a/": "aircraft", #
	f"{DATASET_BASE_URL}/japan/aircrafts/e13a/": "aircraft", #
	f"{DATASET_BASE_URL}/japan/aircrafts/e16a": "aircraft", #
	f"{DATASET_BASE_URL}/japan/aircrafts/m6a/": "aircraft", #
	f"{DATASET_BASE_URL}/japan/aircrafts/h8k/": "aircraft", #
	f"{DATASET_BASE_URL}/japan/aircrafts/ki-100/": "aircraft", #
	f"{DATASET_BASE_URL}/japan/aircrafts/ki-45/": "aircraft", #
	f"{DATASET_BASE_URL}/japan/aircrafts/ki-48/": "aircraft", #
	f"{DATASET_BASE_URL}/japan/aircrafts/ki-60/": "aircraft", #
	f"{DATASET_BASE_URL}/japan/aircrafts/ki-61-hien/": "aircraft", #
	f"{DATASET_BASE_URL}/japan/aircrafts/q1w/": "aircraft", #
	f"{DATASET_BASE_URL}/japan/aircrafts/l2d/": "aircraft", #
	f"{DATASET_BASE_URL}/japan/aircrafts/a5m/": "aircraft", #
	f"{DATASET_BASE_URL}/japan/aircrafts/a6m-zero/": "aircraft", #
	f"{DATASET_BASE_URL}/japan/aircrafts/g3m/": "aircraft", #
	f"{DATASET_BASE_URL}/japan/aircrafts/g4m/": "aircraft", #
	f"{DATASET_BASE_URL}/japan/aircrafts/j2m-raiden/": "aircraft", #
	f"{DATASET_BASE_URL}/japan/aircrafts/ki-21/": "aircraft", #
	f"{DATASET_BASE_URL}/japan/aircrafts/ki-46/": "aircraft", #
	f"{DATASET_BASE_URL}/japan/aircrafts/ki-57/": "aircraft", #
	f"{DATASET_BASE_URL}/japan/aircrafts/ki-67/": "aircraft", #
	f"{DATASET_BASE_URL}/japan/aircrafts/a2n/": "aircraft", #
	f"{DATASET_BASE_URL}/japan/aircrafts/b5n/": "aircraft", #
	f"{DATASET_BASE_URL}/japan/aircrafts/b6n/": "aircraft", #
	f"{DATASET_BASE_URL}/japan/aircrafts/c6n/": "aircraft", #
	f"{DATASET_BASE_URL}/japan/aircrafts/g5n/": "aircraft", #
	f"{DATASET_BASE_URL}/japan/aircrafts/g8n/": "aircraft", #
	f"{DATASET_BASE_URL}/japan/aircrafts/ki-115/": "aircraft", #
	f"{DATASET_BASE_URL}/japan/aircrafts/ki-43/": "aircraft", #
	f"{DATASET_BASE_URL}/japan/aircrafts/ki-44/": "aircraft", #
	f"{DATASET_BASE_URL}/japan/aircrafts/ki-84-hayate/": "aircraft", #
	f"{DATASET_BASE_URL}/japan/aircrafts/ki-54/": "aircraft", #
	f"{DATASET_BASE_URL}/japan/aircrafts/wrecks/": "wreck", #
	f"{DATASET_BASE_URL}/japan/aircrafts/d4y/": "aircraft", #
	f"{DATASET_BASE_URL}/japan/aircrafts/yokosuka_mxy7_ohka/": "aircraft", #
	f"{DATASET_BASE_URL}/japan/aircrafts/p1y/": "aircraft",
	f"{DATASET_BASE_URL}/japan/ijn/midget/": "submarine",
	f"{DATASET_BASE_URL}/japan/japanese-tanks/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/uk/british-tanks/cruiser-mk-iii-a13-mk-i-cruiser-mk-iv-a13-mk-ii/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/uk/british-tanks/challenger/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/uk/british-tanks/churchill-a22/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/uk/british-tanks/comet/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/uk/british-tanks/covenanter/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/uk/british-tanks/a9-tank/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/uk/british-tanks/cruiser-mk-ii-a10/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/uk/british-tanks/crusader-tank/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/uk/british-tanks/vickers/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/uk/british-tanks/matilda-i-a11-tank/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/uk/british-tanks/matilda-ii-a12/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/uk/british-tanks/matilda-a12/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/uk/british-tanks/tetrarch/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/uk/armoured-vehicles/aec_dorchester/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/uk/armoured-vehicles/humber/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/uk/armoured-vehicles/marmon_herrington_-armoured_car/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/uk/armoured-vehicles/universal-carrier-bren-gun-carrier/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/uk/raf/aw23/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/albacore/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/baltimore/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/barracuda/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/fairey-battle/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/beaufighter/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/beau/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/beaufort/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/blenheim1/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/blenheim/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/brigand/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/buckingham/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/buckmaster/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/defiant/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/firebrand/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/dh95/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/halifax/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/hamilcar/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/harrow/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/hudson/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/hurricane/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/hurricane2/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/hurricane1/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/lancaster/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/lanc/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/lincoln/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/london/": "water-based aircraft",
	f"{DATASET_BASE_URL}/uk/raf/lysander/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/manchester/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/maryland/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/monitor/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/mosquito/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/mosquito2/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/mossie/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/roc/": "aircraft", # remove 2 flying boat
	f"{DATASET_BASE_URL}/uk/raf/seafang/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/seafire/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/shetland/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/singapore/": "water-based aircraft",
	f"{DATASET_BASE_URL}/uk/raf/skua/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/spiteful/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/spitfire/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/spitfire2/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/spitfire5/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/spitfire9/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/spit/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/short-stirling/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/stirling/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/sunderland/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/sund/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/swordfish/": "water-based aircraft",
	f"{DATASET_BASE_URL}/uk/raf/tempest/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/tornado/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/typhoon/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/vickers432/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/welkin/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/wellington/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/wellington1/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/whirlwind/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/whitley/": "aircraft",
	f"{DATASET_BASE_URL}/uk/raf/windsor/": "aircraft",
	f"{DATASET_BASE_URL}/ussr/vvs/ar-2/": "aircraft",
	f"{DATASET_BASE_URL}/ussr/vvs/i153/": "aircraft",
	f"{DATASET_BASE_URL}/ussr/vvs/il2-sturmovik/": "aircraft",
	f"{DATASET_BASE_URL}/ussr/vvs/il2/": "aircraft",
	f"{DATASET_BASE_URL}/ussr/vvs/lagg3/": "aircraft",
	f"{DATASET_BASE_URL}/ussr/vvs/li2/": "aircraft",
	f"{DATASET_BASE_URL}/ussr/vvs/mig/": "aircraft",
	f"{DATASET_BASE_URL}/ussr/vvs/pe8/": "aircraft",
	f"{DATASET_BASE_URL}/ussr/vvs/po-2/": "aircraft",
	f"{DATASET_BASE_URL}/ussr/vvs/r-10/": "aircraft",
	f"{DATASET_BASE_URL}/ussr/vvs/su-2/": "aircraft",
	f"{DATASET_BASE_URL}/ussr/armoured-vehicles-2-3/ba-10/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/ussr/armoured-vehicles-2-3/ba-20/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/ussr/armoured-vehicles-2-3/ba-27/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/ussr/spg/isu-122/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/ussr/spg/isu-152/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/ussr/spg/su-100/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/ussr/spg/su-122/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/ussr/spg/su-152/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/ussr/spg/su-85/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/ussr/tanks-2/bt-2-bt-5-bt-7-tank/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/ussr/tanks-2/is-2/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/ussr/tanks-2/kv-1/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/ussr/tanks-2/kv-1-tank/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/ussr/tanks-2/kv-1s/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/ussr/tanks-2/kv-2/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/ussr/tanks-2/t-26/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/ussr/tanks-2/t-27/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/ussr/tanks-2/t-28-tank/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/ussr/tanks-2/t-34/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/ussr/tanks-2/t-34_tank/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/ussr/tanks-2/t-34-85/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/ussr/tanks-2/t-35-tank/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/ussr/tanks-2/t-37-tank/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/ussr/tanks-2/t-38-tank/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/ussr/tanks-2/t-40/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/ussr/tanks-2/t-50/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/ussr/tanks-2/t-60/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/ussr/tanks-2/t-70/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/ussr/artillery_tractor/": "military vehicles",
	f"{DATASET_BASE_URL}/ussr/rkka/red_army/": "military personnel",
	f"{DATASET_BASE_URL}/germany/armored_vehicles/adgz/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/germany/armored_vehicles/kfz13/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/germany/armored_vehicles/sdkfz_221_222_223/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/germany/armored_vehicles/sdkfz_231_232_233/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/germany/armored_vehicles/sdkfz_247/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/germany/armored_vehicles/sdkfz_263/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/germany/kriegsmarine/": "kriegsmarine",
	f"{DATASET_BASE_URL}/germany/german_army_soldiers/": "military personnel",
	f"{DATASET_BASE_URL}/germany/wehrmacht_trucks/bussing-nag/": "military vehicles",
	f"{DATASET_BASE_URL}/germany/wehrmacht_trucks/einheitsdiesel/": "military vehicles",
	f"{DATASET_BASE_URL}/germany/wehrmacht_trucks/faun/": "military vehicles",
	f"{DATASET_BASE_URL}/germany/wehrmacht_trucks/ford-lkw/": "military vehicles",
	f"{DATASET_BASE_URL}/germany/wehrmacht_trucks/ford-pkw/": "military vehicles",
	f"{DATASET_BASE_URL}/germany/wehrmacht_trucks/hanomag/": "military vehicles",
	f"{DATASET_BASE_URL}/germany/wehrmacht_trucks/henschel-33/": "military vehicles",
	f"{DATASET_BASE_URL}/germany/wehrmacht_trucks/horch_830/": "military vehicles",
	f"{DATASET_BASE_URL}/germany/wehrmacht_trucks/horch-901/": "military vehicles",
	f"{DATASET_BASE_URL}/germany/wehrmacht_trucks/krupp/": "military vehicles",
	f"{DATASET_BASE_URL}/germany/wehrmacht_trucks/krupp_protze_l2h_143/": "military vehicles",
	f"{DATASET_BASE_URL}/germany/wehrmacht_trucks/kubelwagen/": "military vehicles",
	f"{DATASET_BASE_URL}/germany/wehrmacht_trucks/mercedes-benz/": "military vehicles",
	f"{DATASET_BASE_URL}/germany/wehrmacht_trucks/opel_blitz/": "military vehicles",
	f"{DATASET_BASE_URL}/germany/wehrmacht_trucks/schwimmwagen/": "military vehicles",
	f"{DATASET_BASE_URL}/germany/aircrafts-2/ar-65/": "aircraft",
	f"{DATASET_BASE_URL}/germany/aircrafts-2/ar-66/": "aircraft",
	f"{DATASET_BASE_URL}/germany/aircrafts-2/arado_234/": "aircraft",
	f"{DATASET_BASE_URL}/germany/aircrafts-2/messerschmitt_bf_110/": "aircraft",
	f"{DATASET_BASE_URL}/germany/aircrafts-2/me_110/": "aircraft",
	f"{DATASET_BASE_URL}/germany/aircrafts-2/bf110/": "aircraft",
	f"{DATASET_BASE_URL}/germany/aircrafts-2/messerschmitt_bf109/": "aircraft",
	f"{DATASET_BASE_URL}/germany/aircrafts-2/messerschmitt-bf-109/": "aircraft",
	f"{DATASET_BASE_URL}/germany/aircrafts-2/bf109/": "aircraft",
	f"{DATASET_BASE_URL}/germany/aircrafts-2/bf_109/": "aircraft",
	f"{DATASET_BASE_URL}/germany/aircrafts-2/bv142/": "aircraft",
	f"{DATASET_BASE_URL}/germany/aircrafts-2/bv222/": "aircraft",
	f"{DATASET_BASE_URL}/germany/aircrafts-2/dornier_do_215/": "aircraft",
	f"{DATASET_BASE_URL}/germany/aircrafts-2/do217/": "aircraft",
	f"{DATASET_BASE_URL}/germany/aircrafts-2/do_335/": "aircraft",
	f"{DATASET_BASE_URL}/germany/aircrafts-2/dornier_do17/": "aircraft",
	f"{DATASET_BASE_URL}/germany/aircrafts-2/fw_189/": "aircraft",
	f"{DATASET_BASE_URL}/germany/aircrafts-2/fw190/": "aircraft",
	f"{DATASET_BASE_URL}/germany/aircrafts-2/focke_wulf_fw_190/": "aircraft",
	f"{DATASET_BASE_URL}/germany/aircrafts-2/fw190d/": "aircraft",
	f"{DATASET_BASE_URL}/germany/aircrafts-2/focke_wulf_fw200/": "aircraft",
	f"{DATASET_BASE_URL}/germany/aircrafts-2/he115/": "aircraft",
	f"{DATASET_BASE_URL}/germany/aircrafts-2/he116/": "aircraft",
	f"{DATASET_BASE_URL}/germany/aircrafts-2/heinkel_he111/": "aircraft",
	f"{DATASET_BASE_URL}/germany/aircrafts-2/he-112/": "aircraft",
	f"{DATASET_BASE_URL}/germany/aircrafts-2/he_162/": "aircraft",
	f"{DATASET_BASE_URL}/germany/aircrafts-2/he_177/": "aircraft",
	f"{DATASET_BASE_URL}/germany/aircrafts-2/hs123/": "aircraft",
	f"{DATASET_BASE_URL}/germany/aircrafts-2/hs_129/": "aircraft",
	f"{DATASET_BASE_URL}/germany/aircrafts-2/junkers-ju87-stuka/": "aircraft",
	f"{DATASET_BASE_URL}/germany/aircrafts-2/ju87/": "aircraft",
	f"{DATASET_BASE_URL}/germany/aircrafts-2/junkers_ju188/": "aircraft",
	f"{DATASET_BASE_URL}/germany/aircrafts-2/ju-290/": "aircraft",
	f"{DATASET_BASE_URL}/germany/aircrafts-2/junkers_ju_52/": "aircraft",
	f"{DATASET_BASE_URL}/germany/aircrafts-2/junkers_ju88/": "aircraft",
	f"{DATASET_BASE_URL}/germany/aircrafts-2/ju-88/": "aircraft",
	f"{DATASET_BASE_URL}/germany/aircrafts-2/ju-90/": "aircraft",
	f"{DATASET_BASE_URL}/germany/aircrafts-2/me261/": "aircraft",
	f"{DATASET_BASE_URL}/germany/aircrafts-2/me321/": "aircraft",
	f"{DATASET_BASE_URL}/germany/aircrafts-2/messerschmitt-me323-gigant/": "aircraft",
	f"{DATASET_BASE_URL}/germany/aircrafts-2/me-323-gigant/": "aircraft",
	f"{DATASET_BASE_URL}/germany/aircrafts-2/messerschmitt-me262/": "aircraft",
	f"{DATASET_BASE_URL}/germany/aircrafts-2/mistel/": "aircraft",
	f"{DATASET_BASE_URL}/germany/artillery/sturmpanzer_iii/": "artillery",
	f"{DATASET_BASE_URL}/germany/artillery/17-cm-k18/": "artillery",
	f"{DATASET_BASE_URL}/germany/artillery/flak-105/": "artillery",
	f"{DATASET_BASE_URL}/germany/artillery/flak-88/": "artillery",
	f"{DATASET_BASE_URL}/germany/artillery/flakpanzer-38/": "artillery",
	f"{DATASET_BASE_URL}/germany/artillery/grille/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/germany/artillery/hummel/": "artillery",
	f"{DATASET_BASE_URL}/germany/artillery/karl-gerat/": "artillery",
	f"{DATASET_BASE_URL}/germany/artillery/lorraine-schlepper/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/germany/artillery/pak43/": "artillery",
	f"{DATASET_BASE_URL}/germany/artillery/sig33b/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/germany/artillery/sig33-bison/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/germany/artillery/sturmpanzer_ii/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/germany/artillery/wespe/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/germany/armored-trains/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/germany/railway_gun/": "armored fighting vehicles",
	f"{DATASET_BASE_URL}/germany/units/afrika_korps/" : "military unit",
	f"{DATASET_BASE_URL}/germany/units/waffen-ss/" : "military unit",
	f"{DATASET_BASE_URL}/germany/units/grossdeutschland/" : "military unit",
	f"{DATASET_BASE_URL}/germany/units/sturmgeschutz_brigade_244/" : "armored fighting vehicles",
}

# how to run in local:
# $ nohup python -u data_collector.py -ddir $HOME/datasets/WW_DATASETs -nw 12 --img_mean_std --thumbnail_size 512,512 -v > logs/wwii_dataset_collection.out &

# run in Pouta:
# $ python data_collector.py -ddir /media/volume/ImACCESS/datasets/WW_DATASETs -sdt 1900-01-01 -edt 1960-12-31
# $ nohup python -u data_collector.py -ddir /media/volume/ImACCESS/datasets/WW_DATASETs -nw 24 --img_mean_std --thumbnail_size 512,512 -v > /media/volume/ImACCESS/trash/wwii_dataset_collection.out &

dataset_name = "WWII".upper()
parser = argparse.ArgumentParser(description=f"{dataset_name} ARCHIVE data colletion")
parser.add_argument('--dataset_dir', '-ddir', type=str, required=True, help='Dataset DIR')
parser.add_argument('--start_date', '-sdt', type=str, default="1939-09-01", help='Start Date')
parser.add_argument('--end_date', '-edt', type=str, default="1945-09-02", help='End Date')
parser.add_argument('--num_workers', '-nw', type=int, default=8, help='Number of CPUs')
parser.add_argument('--batch_size', '-bs', type=int, default=128, help='batch_size')
parser.add_argument('--historgram_bin', '-hb', type=int, default=60, help='Histogram Bins')
parser.add_argument('--img_mean_std', action='store_true', help='calculate image mean & std')
parser.add_argument('--val_split_pct', '-vsp', type=float, default=0.35, help='Validation Split Percentage')
parser.add_argument('--thumbnail_size', type=parse_tuple, default=None, help='Thumbnail size (width, height) in pixels')
parser.add_argument('--seed', '-s', type=int, default=42, help='Random seed')
parser.add_argument('--verbose', '-v', action='store_true', help='Verbose mode')

args, unknown = parser.parse_known_args()
args.dataset_dir = os.path.normpath(args.dataset_dir)
print(args)
print_args_table(args=args, parser=parser)
set_seeds(seed=args.seed, debug=False)

meaningless_words_fpth = os.path.join(IMACCESS_PROJECT_WORKSPACE, 'misc', 'meaningless_words.txt')
# STOPWORDS = nltk.corpus.stopwords.words(nltk.corpus.stopwords.fileids())
STOPWORDS = list()
with open(meaningless_words_fpth, 'r') as file_:
	customized_meaningless_words=[line.strip().lower() for line in file_]
STOPWORDS.extend(customized_meaningless_words)
STOPWORDS = set(STOPWORDS)
# print(STOPWORDS, type(STOPWORDS))

START_DATE = args.start_date
END_DATE = args.end_date

os.makedirs(os.path.join(args.dataset_dir, f"{dataset_name}_{START_DATE}_{END_DATE}"), exist_ok=True)
DATASET_DIRECTORY = os.path.join(args.dataset_dir, f"{dataset_name}_{START_DATE}_{END_DATE}")

os.makedirs(os.path.join(DATASET_DIRECTORY, "images"), exist_ok=True)
IMAGE_DIRECTORY = os.path.join(DATASET_DIRECTORY, "images")

os.makedirs(os.path.join(DATASET_DIRECTORY, "hits"), exist_ok=True)
HITs_DIR = os.path.join(DATASET_DIRECTORY, "hits")

os.makedirs(os.path.join(DATASET_DIRECTORY, "outputs"), exist_ok=True)
OUTPUT_DIRECTORY = os.path.join(DATASET_DIRECTORY, "outputs")

img_rgb_mean_fpth:str = os.path.join(DATASET_DIRECTORY, "img_rgb_mean.gz")
img_rgb_std_fpth:str = os.path.join(DATASET_DIRECTORY, "img_rgb_std.gz")

FIGURE_SIZE = (12, 9)
DPI = 250
# Define regex pattern for WWII years: 1939–1945
YEAR_PATTERN = re.compile(r'\b(19[3][9]|[1][9]4[0-5])\b')
headers = {
	'Content-type': 'application/json',
	'Accept': 'application/json; text/plain; */*',
	'Cache-Control': 'no-cache',
	'Connection': 'keep-alive',
	'Pragma': 'no-cache',
}

HEADERS = {
	"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
	"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
	"Accept-Language": "en-US,en;q=0.5",
	"Accept-Encoding": "gzip, deflate",
	"Connection": "keep-alive",
	"Upgrade-Insecure-Requests": "1",
}

def _download_and_process_image(
	img_url: str,
	img_fpath: str,
	thumbnail_size: tuple = None,
	max_retries: int = 3,
	verbose: bool = False,
) -> bool:
	"""download, verify, and process an image with retries + timeout"""
	for attempt in range(1, max_retries + 1):
		try:
			resp = requests.get(
				img_url, 
				headers=headers,
				stream=True,
			)
			resp.raise_for_status()

			with open(img_fpath, "wb") as f:
				f.write(resp.content)

			with Image.open(img_fpath) as img:
				img.verify()

			if not process_image_for_storage(
				img_path=img_fpath,
				thumbnail_size=thumbnail_size,
				verbose=verbose,
			):
				if verbose:
					print(f"Failed to process image {img_fpath}")
				return False

			if verbose:
				print(f"{img_fpath} downloaded and processed successfully")

			return True
		except Exception as e:
			if verbose:
				print(f"[Attempt {attempt}/{max_retries}] Failed to download {img_url}: {e}")
			
			# Remove partially written file if any
			if os.path.exists(img_fpath):
				try:
					os.remove(img_fpath)
				except OSError:
					pass
			
			if attempt == max_retries:
				# Give up
				return False
			
			# Small backoff before retrying
			time.sleep(1.0)

def extract_year(text):
	match = YEAR_PATTERN.search(str(text))
	return match.group(1) if match else None

def extract_url_info(url:str)-> Dict:
	parsed_url = urllib.parse.urlparse(url)
	url_base = f"{parsed_url.scheme}://{parsed_url.netloc}/gallery" # Extract the base URL
	path_components = parsed_url.path.strip('/').split('/') # Split the path into components		

	# Extract country, main_label, and type
	country = path_components[1] if len(path_components) > 1 else None
	main_label = path_components[2] if len(path_components) > 2 else None
	type_ = path_components[3] if len(path_components) > 3 else None

	# Decode URL-encoded characters (if any)
	if main_label:
		main_label = urllib.parse.unquote(main_label)
		main_label = re.sub(r'[^a-zA-Z\s]', ' ', main_label) # Remove special characters and digits
		main_label = re.sub(r'\s+', ' ', main_label)  # Remove extra whitespace

	if type_:
		type_ = urllib.parse.unquote(type_)

	return {
		"url_base": url_base,
		"country": country,
		"main_label": main_label,
		"type": type_
	}

def get_dframe(
	doc_idx: int,
	doc_url: str,
	user_query: str,
	num_workers: int = 8,
	thumbnail_size: tuple = None,
	verbose: bool = False,
) -> pd.DataFrame:

	# ── 0. Setup & cache key ──────────────────────────────────────────────
	content_to_hash = f"{doc_url}_{START_DATE}_{END_DATE}"
	hash_digest = hashlib.md5(content_to_hash.encode('utf-8')).hexdigest()
	query_prefix = user_query.replace(' ', '_') + '_' if user_query else ''
	df_fpth = os.path.join(HITs_DIR, f"df_{query_prefix}{hash_digest}.gz")
	if verbose:
		print(f"\n[EXTRACTING DOCUMENT {doc_idx+1:3d}/{len(URLs)}]")
		print(f"  ├─ DOC_URL         : {doc_url}")
		print(f"  ├─ user_query      : « {user_query} »")
		print(f"  ├─ thumbnail_size  : {thumbnail_size}")
		print(f"  ├─ content_to_hash : {content_to_hash}")
		print(f"  ├─ df_fpath        : {df_fpth}")
		print(f"  └─ num_workers     : {num_workers}")

	# ── 1. Cache path ─────────────────────────────────────────────────────
	if os.path.exists(df_fpth):
			if verbose:
					print(f"  [CACHE] {df_fpth} exists => loading...")
			df = load_pickle(fpath=df_fpth)
			if df.shape[0] == 0:
					print(f"  [WARNING] Cached DF is empty {df.shape} => ignoring cache, re-scraping...")
			else:
					# Normalize legacy root-relative img_urls
					rel_mask = df['img_url'].fillna('').astype(str).str.startswith('/')
					if rel_mask.any():
							print(f"  Normalizing {int(rel_mask.sum())} root-relative img_url(s) in cached DF...")
							df.loc[rel_mask, 'img_url'] = [
									urllib.parse.urljoin(base, url)
									for base, url in zip(df.loc[rel_mask, 'doc_url'], df.loc[rel_mask, 'img_url'])
							]
					img_paths = df['img_path'].tolist()
					missing_indices = [i for i, p in enumerate(img_paths) if not os.path.exists(p)]
					missing_paths = [img_paths[i] for i in missing_indices]
					if missing_indices:
							if verbose:
								print(f"Downloading {len(missing_indices)} missing images ({num_workers} workers)...")
								print(f"Missing paths: {len(missing_paths)}")
							def download_task(idx: int):
									return idx, df['img_url'].iloc[idx], _download_and_process_image(
											img_url=df['img_url'].iloc[idx],
											img_fpath=df['img_path'].iloc[idx],
											thumbnail_size=thumbnail_size,
											verbose=verbose,
									)
							failed = []
							with ThreadPoolExecutor(max_workers=num_workers) as ex:
									futures = {ex.submit(download_task, idx): idx for idx in missing_indices}
									for fut in tqdm(as_completed(futures), total=len(futures),
																	desc="Downloading missing images", ncols=100):
											idx, url, ok = fut.result()
											if not ok:
													failed.append((idx, url))
							if failed and verbose:
									print(f"  Failed to download {len(failed)} image(s).")
									for i, (idx, url) in enumerate(failed):
											print(f"    {i}: idx={idx} url={url}")
					return df

	# ── 2. Scrape gallery index page ──────────────────────────────────────
	doc_url_info = extract_url_info(doc_url)
	print(json.dumps(doc_url_info, indent=4, ensure_ascii=False))
	session = requests.Session()
	session.headers.update(HEADERS)
	df_st_time = time.time()
	try:
			response = session.get(doc_url, timeout=30)
			response.raise_for_status()
			soup = BeautifulSoup(response.text, 'html.parser')
			header = None
			header_el = soup.find('h1')
			if not header_el:
					header_el = soup.find('h2', class_="entry-title")
			if header_el:
					header = header_el.get_text(strip=True)
			if not header:
					print(f"  [WARNING] Could not find title in {doc_url}")
					header = doc_url_info.get('type', 'Unknown')
			hits = soup.find_all('article', class_='photo-card')
			if not hits:
					hits = soup.find_all('img', class_='attachment-thumbnail')
	except Exception as e:
			print(f"  [ERROR] Failed to retrieve or parse {doc_url}: {e}")
			return None
	print(f"\nGallery header: {header}")

	# Gallery-level description (used as fallback)
	gallery_description = ""
	caption_element = soup.find('div', class_='folder-description')

	if not caption_element:
		caption_element = soup.find('div', class_='entry-caption')

	if caption_element:
		gallery_description = caption_element.get_text(strip=True)
		gallery_description = re.sub(r'\s+', ' ', gallery_description).strip()

	# if gallery_description.lower() and header.lower() not in gallery_description.lower():
	# 	gallery_description = header + " " + gallery_description
	# elif not gallery_description.strip():
	# 	gallery_description = header

	print(f"\nGallery description:\n{gallery_description}\n")

	# Old-layout caption map
	caption_map = {}
	for p in soup.find_all('p', class_='wp-caption-text gallery-caption'):
			cid = p.get('id')
			if cid:
					txt = p.get_text(strip=True)
					if txt:
							caption_map[cid] = txt
	print(f"{len(caption_map)} Caption Map(s): {json.dumps(caption_map, indent=2, ensure_ascii=False)}")


	# ── 3. Helper: extract lightweight metadata from a hit (no I/O) ───────
	def _extract_hit_meta(vdoc):
		"""Return {img_url_raw, doc_title, doc_doc_url} or None."""
		if vdoc.name == 'article':
			img_tag = vdoc.find('img')
			if not img_tag:
				return None
			img_url = img_tag.get('src') or img_tag.get('data-src')
			caption_el = vdoc.find('h3')
			doc_title = caption_el.get_text(strip=True) if caption_el else (img_tag.get('alt') or '')
			parent_a = vdoc.find('a')
			doc_doc_url = parent_a.get('href') if parent_a else None
		else:
			img_tag = vdoc
			img_url = img_tag.get('data-src')
			if not img_url:
				return None
			parent_a = img_tag.find_parent('a')
			doc_doc_url = parent_a.get('href') if parent_a else None
			doc_title = img_tag.get("alt")
			if doc_title == "Folder Icon":
				doc_title = None
			aria_id = img_tag.get("aria-describedby")
			if aria_id:
				ct = caption_map.get(aria_id)
				if ct:
					doc_title = ct
		return {
			'img_url_raw': img_url,
			'doc_title': doc_title,
			'doc_doc_url': doc_doc_url,
		}

	# ── 4. First pass: collect hit metadata & unique photo page URLs ──────
	print(f"Found {len(hits)} image(s) on index page")
	hit_metas = []
	unique_photo_urls = set()
	for vdoc in hits:
		meta = _extract_hit_meta(vdoc)
		if meta is None:
			continue
		specific_doc_url = urllib.parse.urljoin(doc_url, meta['doc_doc_url']) if meta['doc_doc_url'] else doc_url
		meta['specific_doc_url'] = specific_doc_url
		hit_metas.append(meta)
		if specific_doc_url != doc_url:
			unique_photo_urls.add(specific_doc_url)

	# ── 5. Parallel fetch: photo-specific descriptions ────────────────────
	photo_descriptions = {}
	def _fetch_photo_desc(url: str):
			"""Scrape a single photo page and return its caption text."""
			try:
					# Each thread gets its own short-lived session to avoid
					# race conditions on the shared `session` object.
					s = requests.Session()
					s.headers.update(HEADERS)
					r = s.get(url, timeout=15)
					r.raise_for_status()
					bs = BeautifulSoup(r.text, 'html.parser')
					# Try multiple common caption containers (theme-dependent)
					el = (
							bs.find('div', class_='photo-description') or
							bs.find('div', class_='entry-caption') or
							bs.find('figcaption') or
							bs.find('div', class_='wp-caption-text') or
							bs.find('p', class_='wp-caption-text')
					)
					if el:
							txt = el.get_text(strip=True)
							txt = re.sub(r'\s+', ' ', txt).strip()
							if txt:
									return url, txt
					# Fallback: meta description
					meta_tag = bs.find('meta', attrs={'name': 'description'})
					if meta_tag and meta_tag.get('content'):
							return url, meta_tag['content'].strip()
			except Exception as e:
					if verbose:
							print(f"  [WARN] Could not fetch photo description from {url}: {e}")
			return url, None
	if unique_photo_urls:
		if verbose:
			print(f"  Fetching descriptions from {len(unique_photo_urls)} photo page(s)...")
		with ThreadPoolExecutor(max_workers=min(num_workers, len(unique_photo_urls))) as ex:
			futures = [ex.submit(_fetch_photo_desc, url) for url in unique_photo_urls]
			for fut in as_completed(futures):
			# for fut in tqdm(as_completed(futures), total=len(futures), desc="Photo descriptions", ncols=100, disable=not verbose):
				url, desc = fut.result()
				if desc:
					photo_descriptions[url] = desc

	# ── 6. Second pass: download images & build rows ──────────────────────
	data = []
	def _fetch_image(preferred_url, img_fpath, fallback_url=None):
			"""Try preferred URL, then fallback if it fails."""
			if _download_and_process_image(
					img_url=preferred_url, img_fpath=img_fpath,
					thumbnail_size=thumbnail_size, verbose=verbose
			):
					return preferred_url
			if fallback_url and fallback_url != preferred_url:
					if verbose:
							print(f"  Primary URL failed, retrying with original: {fallback_url}")
					if _download_and_process_image(
							img_url=fallback_url, img_fpath=img_fpath,
							thumbnail_size=thumbnail_size, verbose=verbose
					):
							return fallback_url
			return None

	for idoc, meta in enumerate(hit_metas):
		print(f"\n[{idoc+1:4d}/{len(hit_metas)}] {meta['doc_title'] or 'Untitled'}")
		img_url = meta['img_url_raw']
		if not img_url:
				print(f"    No image URL found, skipping...")
				continue
		# Absolutize
		img_url = urllib.parse.urljoin(doc_url, img_url)
		original_img_url = img_url
		# Clean
		img_url = img_url.replace("_cache/", "")
		img_url = re.sub(r'-\d+x\d+\.jpg$', '.jpg', img_url)
		img_url = re.sub(r'_hu_[a-f0-9]+\.jpg$', '.jpg', img_url)
		filename = os.path.basename(img_url)
		img_fpath = os.path.join(IMAGE_DIRECTORY, filename)
		specific_doc_url = meta['specific_doc_url']
		# Extract year
		extracted_year = None
		for src in [meta['doc_title'], specific_doc_url, img_url, filename, gallery_description]:
				if src:
						y = extract_year(src)
						if y:
								extracted_year = y
								break
		# ── Description: photo-specific > gallery fallback ─────────────────
		row_description = photo_descriptions.get(specific_doc_url, gallery_description)
		# Download / process image
		if not os.path.exists(img_fpath):
				working_url = _fetch_image(img_url, img_fpath, fallback_url=original_img_url)
				if working_url is None:
					if verbose:
						print(f"[FAILED] downloading {img_url} => Skipping...")
					continue
				img_url = working_url
		else:
				if not process_image_for_storage(
						img_path=img_fpath, thumbnail_size=thumbnail_size, verbose=verbose
				):
						if os.path.exists(img_fpath):
								if verbose:
										print(f"    Existing image {img_fpath} failed re-processing. Re-downloading...")
								os.remove(img_fpath)
						working_url = _fetch_image(img_url, img_fpath, fallback_url=original_img_url)
						if working_url is None:
							if verbose:
								print(f"[FAILED] re-download {img_url} => Skipping...")
							continue
						img_url = working_url
				else:
						if verbose:
							print(f"[SUCCESS] Existing image {img_fpath} re-processed!")
		row = {
			'id': filename,
			'date': extracted_year,
			'doc_url': specific_doc_url,
			'img_url': img_url,
			'title': meta['doc_title'],
			'description': row_description,
			'country': doc_url_info.get("country"),
			'user_query': [user_query] if user_query else None,
			'label': user_query if user_query else None,
			'img_path': img_fpath,
		}
		if verbose:
			print(f"Appending row:")
			print(json.dumps(row, indent=6, ensure_ascii=False))
			print("-" * 120)

		data.append(row)

	# ── 7. Build DataFrame ────────────────────────────────────────────────
	if verbose:
		print(f"  Creating DataFrame from {len(data)} row(s)...")
	df = pd.DataFrame(data)
	print(f"  DF: {df.shape} {type(df)} Elapsed time: {time.time()-df_st_time:.1f} sec")
	if df.shape[0] > 0:
			print(f"  Saving DF to {df_fpth}")
			save_pickle(pkl=df, fname=df_fpth)
	else:
			print(f"  [WARNING] Scraped DF is empty {df.shape} — NOT caching to {df_fpth}")

	return df

def get_dframe_old(
	doc_idx: int,
	doc_url: str,
	user_query: str,
	num_workers: int,
	thumbnail_size: tuple = None,
	verbose: bool = False,
) -> pd.DataFrame:
	print(f"\n>> Extracting DF for user_query[{doc_idx}]: « {user_query} » from {doc_url} with thumbnail_size={thumbnail_size}")
	content_to_hash = f"{doc_url}_{START_DATE}_{END_DATE}"
	print(f"content_to_hash: {content_to_hash}")
	hash_digest = hashlib.md5(content_to_hash.encode('utf-8')).hexdigest()
	query_prefix = user_query.replace(' ', '_') + '_' if user_query else ''
	df_fpth = os.path.join(HITs_DIR, f"df_{query_prefix}{hash_digest}.gz")
	print(f"df_fpth: {df_fpth}")

	# ── CACHE PATH ──────────────────────────────────────────────────────────
	if os.path.exists(df_fpth):
			df = load_pickle(fpath=df_fpth)
			if df.shape[0] == 0:
					raise ValueError(f"Empty DF: {df.shape} => Exit...")
			print(df[['id', 'img_path']].head(10))
			print()
			# FIX #3: Reconstruct path from filename instead of string-replace
			df['img_path'] = df['img_path'].apply(
					lambda x: os.path.join(IMAGE_DIRECTORY, os.path.basename(x))
			)
			print(df[['id', 'img_path']].head(10))
			print("#" * 160)
			# Identify missing images
			missing_indices = [
					i
					for i, path in enumerate(df['img_path'].tolist())
					if not os.path.exists(path)
			]
			missing_paths = [
					path
					for path in df['img_path'].tolist()
					if not os.path.exists(path)
			]
			if missing_indices:
					if verbose:
							print(f"Downloading {len(missing_indices)} missing images using {num_workers} workers...")
							print(f"Missing paths:\n{missing_paths}\n")
					def download_task(idx: int):
							img_path = df['img_path'].iloc[idx]
							img_url = df['img_url'].iloc[idx]
							success = _download_and_process_image(
									img_url=img_url,
									img_fpath=img_path,
									thumbnail_size=thumbnail_size,
									verbose=verbose,
							)
							return idx, img_url, success
					failed = []
					with ThreadPoolExecutor(max_workers=num_workers) as ex:
							futures = {ex.submit(download_task, idx): idx for idx in missing_indices}
							for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading missing images", ncols=100):
									idx, url, ok = fut.result()
									if not ok:
											failed.append((idx, url))
					if failed and verbose:
							print(f"Failed to download {len(failed)} images.")
							for i, (idx, url) in enumerate(failed):
									print(f"{i} {idx} {url}")
			return df
	# ── FETCH DOCUMENT ──────────────────────────────────────────────────────
	doc_url_info = extract_url_info(doc_url)
	print(json.dumps(doc_url_info, indent=4, ensure_ascii=False))
	session = requests.Session()
	session.headers.update(HEADERS)
	df_st_time = time.time()
	try:
			response = session.get(doc_url, timeout=30)
			response.raise_for_status()
			soup = BeautifulSoup(response.text, 'html.parser')
			# Try new layout first, fallback to old layout
			header = None
			header_el = soup.find('h1')
			if header_el:
					header = header_el.get_text(strip=True)
			else:
					header_el = soup.find('h2', class_="entry-title")
					if header_el:
							header = header_el.get_text(strip=True)
			if not header:
					print(f"[WARNING] Could not find title in {doc_url}")
					header = doc_url_info.get('type', 'Unknown')
			# Try new layout images first
			hits = soup.find_all('article', class_='photo-card')
			if not hits:
					hits = soup.find_all('img', class_='attachment-thumbnail')
	except Exception as e:
			print(f"[ERROR] Failed to retrieve or parse {doc_url}: {e}")
			return None
	print("-" * 150)
	print(f"\nDoc header:\n{header}")
	# ── DESCRIPTION ─────────────────────────────────────────────────────────
	doc_description = ""
	caption_element = soup.find('div', class_='folder-description')
	if not caption_element:
			caption_element = soup.find('div', class_='entry-caption')
	if caption_element:
			doc_description = caption_element.get_text(strip=True)
			doc_description = re.sub(r'\s+', ' ', doc_description).strip()
	if doc_description.lower() and header.lower() not in doc_description.lower():
			doc_description = header + " " + doc_description
	elif not doc_description.strip():
			doc_description = header
	print(f"\nDoc Description:\n{doc_description}\n")
	# ── CAPTION MAP (old layout only) ───────────────────────────────────────
	caption_map = {}
	for p in soup.find_all('p', class_='wp-caption-text gallery-caption'):
			cid = p.get('id')
			if cid:
					caption_text = p.get_text(strip=True)
					if caption_text:
							caption_map[cid] = caption_text
	print(f"{len(caption_map)} Caption Map(s):\n{json.dumps(caption_map, indent=4, ensure_ascii=False)}")
	print(f"Found {len(hits)} Document(s) => Extracting information [might take a while]")
	data = []
	for idoc, vdoc in enumerate(hits):
			print(f"[{idoc + 1}/{len(hits)}] {vdoc}")
			# ── Extract metadata depending on layout ──────────────────────────
			if vdoc.name == 'article':
					# NEW layout
					img_tag = vdoc.find('img')
					if not img_tag:
							continue
					img_url = img_tag.get('src')
					caption_el = vdoc.find('h3')
					doc_title = caption_el.get_text(strip=True) if caption_el else img_tag.get('alt', '')
					parent_a = vdoc.find('a')
					doc_doc_url = parent_a.get('href') if parent_a else None
			else:
					# OLD layout
					img_tag = vdoc
					img_url = img_tag.get('data-src')
					if not img_url:
							print(f"[WARNING] No data-src found, skipping...")
							continue
					parent_a = img_tag.find_parent('a')
					doc_doc_url = parent_a.get('href') if parent_a else None
					doc_title = img_tag.get("alt")
					if doc_title == "Folder Icon":
							doc_title = None
					aria_id = img_tag.get("aria-describedby")
					if aria_id:
							caption_title = caption_map.get(aria_id)
							if caption_title:
									doc_title = caption_title
			if not img_url:
					print(f"[WARNING] No image URL found, skipping...")
					continue
			# ── FIX #1 & #2: Absolutize & clean URL ───────────────────────────
			img_url = urllib.parse.urljoin(doc_url, img_url)
			img_url = img_url.replace("_cache/", "")
			img_url = re.sub(r'-\d+x\d+\.jpg$', '.jpg', img_url)
			# NOTE: Do NOT strip Hugo _hu_ hash suffix — the hashed file is usually the only one deployed.
			# If you need the original, add a fallback inside _download_and_process_image().
			filename = os.path.basename(img_url)
			img_fpath = os.path.join(IMAGE_DIRECTORY, filename)
			specific_doc_url = urllib.parse.urljoin(doc_url, doc_doc_url) if doc_doc_url else doc_url
			# ── Extract year ──────────────────────────────────────────────────
			date_sources = [doc_title, specific_doc_url, img_url, filename, doc_description]
			extracted_year = None
			for src in date_sources:
					if src:
							year = extract_year(src)
							if year:
									extracted_year = year
									break
			if verbose:
					print(f"extracted_year: {extracted_year}")
			# ── Download & process image ────────────────────────────────────────
			if not os.path.exists(img_fpath):
					ok = _download_and_process_image(img_url, img_fpath, thumbnail_size, verbose)
					if not ok:
							if verbose:
									print(f"Failed to download {img_url} => Skipping...")
							continue
			else:
					ok = process_image_for_storage(img_path=img_fpath, thumbnail_size=thumbnail_size, verbose=verbose)
					if not ok:
							if verbose:
									print(f"Existing image {img_fpath} failed re-processing. Attempting re-download...")
							os.remove(img_fpath)
							ok = _download_and_process_image(img_url, img_fpath, thumbnail_size, verbose)
							if not ok:
									if verbose:
											print(f"Failed to re-download {img_url} => Skipping...")
									continue
					else:
							if verbose:
									print(f"Existing image {img_fpath} re-processed successfully")
			# ── Build row ───────────────────────────────────────────────────────
			row = {
					'id': filename,
					'date': extracted_year,
					'doc_url': specific_doc_url,
					'img_url': img_url,
					'title': doc_title,
					'description': doc_description,
					'country': doc_url_info.get("country"),
					'user_query': user_query if user_query else None,   # FIX #5: string, not list
					'label': user_query if user_query else None,
					'img_path': img_fpath,
			}
			if verbose:
					print(f"Appending Row[{idoc + 1}/{len(hits)}]:")
					print(f"{json.dumps(row, indent=4, ensure_ascii=False)}")
					print("-" * 120)
			data.append(row)
	# ── FIX #6: Guard against empty data ──────────────────────────────────
	if not data:
			print("[WARNING] No images successfully downloaded or processed.")
			return pd.DataFrame()
	if verbose:
			print(f"Creating DataFrame from {len(data)} rows...")
	df = pd.DataFrame(data)
	print(f"DF: {df.shape} {type(df)} Elapsed time: {time.time() - df_st_time:.1f} sec")
	print(f"Saving DF to {df_fpth}")
	save_pickle(pkl=df, fname=df_fpth)
	return df

@measure_execution_time
def main():
	# slice[:N] URLs [JUST FOR TESTING]:
	# URLs = {k:v for i, (k, v) in enumerate(URLs.items()) if i < 3}

	dfs_fname = os.path.join(HITs_DIR, f"{dataset_name}_{len(URLs)}_dfs.gz")
	
	try:
		dfs = load_pickle(fpath=dfs_fname,)
		print(f"Loaded {len(dfs)} dfs from {os.path.join(OUTPUT_DIRECTORY, f'{dataset_name}_dfs.gz')}")
	except Exception as e:
		print(f"<!> {e}")
		print(f"Scraping {len(URLs)} URLs...")
		dfs = [
			get_dframe(
				doc_idx=i, 
				doc_url=k, 
				user_query=v,
				num_workers=args.num_workers,
				thumbnail_size=args.thumbnail_size,
				verbose=args.verbose,
			) for i, (k, v) in enumerate(URLs.items())
		]
		dfs = [df for df in dfs if df is not None]
		# save_pickle(pkl=dfs, fname=dfs_fname,)
		# print(f"Saved {len(dfs)} dfs to {dfs_fname}")

	total_searched_labels = len(dfs)
	print(f"Concatinating {total_searched_labels} x {type(dfs[0])} dfs...")
	wwii_df = pd.concat(dfs, ignore_index=True)
	print(f"wwii_df {type(wwii_df)} {wwii_df.shape} {list(wwii_df.columns)}")
	print(wwii_df.info(verbose=True, memory_usage="deep"))
	print(wwii_df.head(5))
	print("="*100)

	# 1: multi label:
	print(f"[MULTI-LABEL]")
	multi_label_synched_df = wwii_df.copy()
	multi_label_final_df = get_enriched_description(
		df=multi_label_synched_df, 
		check_english=True, 
		verbose=args.verbose
	)
	# validate_text_cleaning_pipeline(df=multi_label_final_df, text_column='enriched_document_description')

	multi_label_fpath = os.path.join(DATASET_DIRECTORY, "metadata_multi_label.csv")
	multi_label_final_df.to_csv(multi_label_fpath, index=False)
	try:
		multi_label_final_df.to_excel(multi_label_fpath.replace('.csv', '.xlsx'), index=False)
	except Exception as e:
		print(f"Failed to write Excel file: {e}")

	if args.verbose:
		print(f"[SAVED] {multi_label_fpath}")
		print(f"-"*100)

	# 2: single label:
	print(f"\n[SINGLE-LABEL]")
	# a) drop None from labels:
	print(f"Checking for None labels: {wwii_df['label'].isna().sum()} None labels / {wwii_df.shape[0]} total samples")
	single_label_final_df = wwii_df.dropna(subset=['label'])
	# b) save
	single_label_fpath = multi_label_fpath.replace('multi_label', 'single_label')
	unique_labels = single_label_final_df['label'].unique()
	print(f"{len(unique_labels)} Unique labels ({type(unique_labels)}):\n{unique_labels}\n")
	if unique_labels.size == 0:
		print(f"[ERROR] No unique labels found in the dataset. => stopping.")
		return

	print(f"Saving SINGLE-LABEL dataset in {single_label_fpath}...")
	single_label_final_df.to_csv(single_label_fpath, index=False)
	try:
		single_label_final_df.to_excel(single_label_fpath.replace('.csv', '.xlsx'), index=False)
	except Exception as e:
		print(f"Failed to write Excel file: {e}")
	print(f"[SUCCESS] saved {single_label_fpath}")

	print(single_label_final_df['label'].value_counts())

	label_dirstribution_fname = os.path.join(
		OUTPUT_DIRECTORY, 
		f"{dataset_name}_single_label_distribution_{wwii_df.shape[0]}_x_{unique_labels.shape[0]}.png"
	)
	viz.plot_label_distribution(
		df=single_label_final_df,
		fpth=label_dirstribution_fname,
		FIGURE_SIZE=(14, 8),
		DPI=260,
		label_column='label',
	)

	# stratified splitting [single-label]:
	train_df, val_df = get_single_label_stratified_split(
		csv_file=single_label_fpath,
		val_split_pct=args.val_split_pct,
		seed=args.seed,
		verbose=args.verbose,
	)

	viz.plot_train_val_label_distribution(
		train_df=train_df,
		val_df=val_df,
		dataset_name=dataset_name,
		VAL_SPLIT_PCT=args.val_split_pct,
		fname=os.path.join(OUTPUT_DIRECTORY, f'simple_random_split_stratified_single_label_distribution_train_val_{args.val_split_pct}_pct.png'),
	)

	if args.img_mean_std and os.listdir(IMAGE_DIRECTORY):
		try:
			img_rgb_mean = load_pickle(fpath=img_rgb_mean_fpth)
			img_rgb_std = load_pickle(fpath=img_rgb_std_fpth)
		except Exception as e:
			print(f"{e}")
			img_rgb_mean, img_rgb_std = get_mean_std_rgb_img_multiprocessing(
				source=IMAGE_DIRECTORY, 
				num_workers=args.num_workers,
				batch_size=args.batch_size,
				img_rgb_mean_fpth=img_rgb_mean_fpth,
				img_rgb_std_fpth=img_rgb_std_fpth,
				verbose=args.verbose,
			)

if __name__ == '__main__':
	print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(160, " "))
	get_ip_info()
	main()
	print(f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ".center(160, " "))

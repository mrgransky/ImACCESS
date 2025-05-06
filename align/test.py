# from PIL import Image
# from transformers import AlignProcessor, AlignModel
# import torch
# from torch.nn.functional import cosine_similarity
# import requests
# from io import BytesIO

# # Load processor and model
# processor = AlignProcessor.from_pretrained("kakaobrain/align-base")
# model = AlignModel.from_pretrained("kakaobrain/align-base")

# # Load your dataset
# text_data = ["A cat sitting on a mat", "A dog playing with a ball"]
# image_urls = [
#     "https://i.postimg.cc/7hJL9KK4/fluffy-siberian-cat-sitting-on-600nw-2150187551.png",
#     "https://i.postimg.cc/4xgzKc3y/images.jpg",
# ]

# # Fetch images from URLs and convert them to PIL images
# image_data = []
# for url in image_urls:
#     response = requests.get(url)
#     if response.status_code == 200:
#         img = Image.open(BytesIO(response.content)).convert("RGB")
#         image_data.append(img)
#     else:
#         print(f"Failed to fetch image from URL: {url}")

# # Preprocess the data
# inputs = processor(
#     images=image_data, 
#     text=text_data, 
#     return_tensors="pt", 
#     padding="max_length", 
#     max_length=128, 
#     truncation=True
# )

# # Move inputs to GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)
# inputs = {k: v.to(device) for k, v in inputs.items()}

# # Forward pass
# outputs = model(**inputs)

# # Extract embeddings for images and text
# image_embeds = outputs.image_embeds  # Shape: (batch_size, embedding_dim)
# text_embeds = outputs.text_embeds    # Shape: (batch_size, embedding_dim)

# # Compute cosine similarity between image and text embeddings
# similarity_scores = cosine_similarity(image_embeds, text_embeds)

# # Print similarity scores
# for i, score in enumerate(similarity_scores):
#     print(f"Similarity score for pair {i + 1}: {score.item()}")

import torch
from PIL import Image
from transformers import AlignProcessor, AlignModel
from urllib.request import urlopen
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
processor = AlignProcessor.from_pretrained("kakaobrain/align-base")
model = AlignModel.from_pretrained("kakaobrain/align-base")
model.to("cuda:0") # Add this line to move the model to the GPU
model.eval()
image = Image.open(
	urlopen(
		# 'https://upload.wikimedia.org/wikipedia/commons/thumb/1/1f/8inchHowitzerTowedByScammellPioneer12Jan1940.jpg/632px-8inchHowitzerTowedByScammellPioneer12Jan1940.jpg'
		# "https://truck-encyclopedia.com/ww1/img/photos/Liberty_B2_Truck.jpg"
		# "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Military_Parade_in_Cairo_%285%29.tif/lossy-page1-2316px-Military_Parade_in_Cairo_%285%29.tif.jpg"
		"https://digitalcollections.smu.edu/digital/api/singleitem/image/ryr/2455/default.jpg"
	)
).convert("RGB")

print(image.size, image.mode)

# # Define category sets for different aspects of visual content
# object_categories = [
# 	# Military vehicles
# 	"tank", "jeep", "armored car", "truck", "military aircraft", "helicopter",
# 	"submarine", "battleship", "aircraft carrier", "fighter jet", "bomber aircraft",
	
# 	# Military personnel
# 	"soldier", "officer", "military personnel", "pilot", "sailor", "cavalry",
	
# 	# Weapons
# 	"gun", "rifle", "machine gun", "artillery", "cannon", "missile", "bomb",
	
# 	# Other military objects
# 	"military base", "bunker", "trench", "fortification", "flag", "military uniform"
# ]

# scene_categories = [
# 	# Terrain types
# 	"desert", "forest", "urban area", "beach", "mountain", "field", "ocean", "river",
	
# 	# Military scenes
# 	"battlefield", "military camp", "airfield", "naval base", "military parade",
# 	"military exercise", "war zone", "training ground", "military factory"
# ]

# era_categories = [
# 	"World War I era", "World War II era", "Cold War era", "modern military",
# 	"1910s style", "1940s style", "1960s style", "1980s style", "2000s style"
# ]

# activity_categories = [
# 	"driving", "flying", "marching", "fighting", "training", "maintenance",
# 	"loading equipment", "unloading equipment", "towing", "firing weapon",
# 	"military parade", "crossing terrain", "naval operation"
# ]

object_categories = [
		# Tanks & Armored Vehicles (WWI-WWII)
		"tank", "light tank", "medium tank", "heavy tank", "super-heavy tank", 
		"tank destroyer", "self-propelled gun", "armored car", "half-track", 
		"armored personnel carrier", "armored train", "reconnaissance vehicle",
		"Mark IV tank", "Tiger tank", "Panther tank", "T-34 tank", "Sherman tank",
		"Churchill tank", "KV-1 tank", "Panzer IV", "Panzer III", "Stuart tank",
		"SdKfz armored vehicle", "Kettenkrad", "M4 Sherman", "T-34/85", "IS-2 tank",
		
		# Light & Utility Vehicles
		"jeep", "staff car", "command car", "ambulance", "motorcycle", 
		"military truck", "supply truck", "fuel truck", "artillery tractor", 
		"amphibious vehicle", "scout car", "Willys Jeep", "Kubelwagen", 
		"Dodge WC series", "Opel Blitz", "Zis truck", "weapons carrier",
		
		# Aircraft
		"military aircraft", "fighter aircraft", "bomber aircraft", "reconnaissance aircraft", 
		"dive bomber", "torpedo bomber", "transport aircraft", "seaplane", "flying boat",
		"biplane", "monoplane", "fighter-bomber", "ground attack aircraft", "night fighter",
		"Spitfire", "Messerschmitt Bf 109", "P-51 Mustang", "Focke-Wulf Fw 190", 
		"B-17 Flying Fortress", "Lancaster bomber", "Heinkel He 111", "Junkers Ju 87 Stuka",
		"Mitsubishi Zero", "Il-2 Sturmovik", "P-47 Thunderbolt", "Hurricane fighter", "helicopter",
		
		# Naval Vessels
		"submarine", "U-boat", "destroyer", "cruiser", "battleship", "aircraft carrier", 
		"battlecruiser", "corvette", "frigate", "minesweeper", "torpedo boat", 
		"landing craft", "PT boat", "pocket battleship", "gunboat", "escort carrier",
		"liberty ship", "merchant vessel", "hospital ship", "troop transport",
		
		# Military Personnel
		"soldier", "infantryman", "officer", "NCO", "general", "field marshal",
		"pilot", "bomber crew", "tanker", "artilleryman", "sailor", "marine", 
		"paratrooper", "commando", "sniper", "medic", "military police", 
		"cavalry", "SS officer", "Wehrmacht soldier", "Red Army soldier", 
		"Desert Rat", "Afrika Korps soldier", "Luftwaffe personnel", "naval officer",
		
		# Weapons & Ordnance
		"rifle", "machine gun", "submachine gun", "pistol", "bayonet", "flamethrower", 
		"mortar", "artillery piece", "howitzer", "field gun", "anti-tank gun", "cannon", 
		"anti-aircraft gun", "rocket launcher", "grenade", "hand grenade", "rifle grenade",
		"landmine", "naval mine", "depth charge", "torpedo", "aerial bomb", "incendiary bomb",
		"Thompson submachine gun", "MG-42", "Karabiner 98k", "M1 Garand", "Sten gun",
		"Luger pistol", "PIAT", "Bazooka", "Panzerfaust", "88mm gun",
		
		# Military Infrastructure
		"bunker", "pillbox", "gun emplacement", "observation post", "barbed wire", 
		"trenches", "foxhole", "dugout", "fortification", "coastal defense", 
		"anti-tank obstacle", "dragon's teeth", "minefield", "pontoon bridge",
		"Bailey bridge", "military headquarters", "command post", "communications center",
		
		# Military Insignia & Symbols
		"military flag", "swastika flag", "rising sun flag", "Soviet flag", "Union Jack", 
		"American flag", "regimental colors", "military insignia", "rank insignia", 
		"unit patch", "medal", "military decoration", "Iron Cross", "Victoria Cross",
		"Medal of Honor", "military helmet", "steel helmet", "Brodie helmet",
		"Stahlhelm", "Adrian helmet", "gas mask",
		
		# Military Equipment
		"military uniform", "combat uniform", "field equipment", "backpack", "mess kit", 
		"entrenching tool", "canteen", "ammunition belt", "bandolier", "map case", 
		"binoculars", "field telephone", "radio equipment", "signal equipment",
		"parachute", "life vest", "fuel drum", "jerry can", "ration box",
		"military stretcher", "field kitchen", "anti-gas equipment"
]

scene_categories = [
		# European Theaters
		"Western Front", "Eastern Front", "Italian Front", "North African Front",
		"Normandy beaches", "French countryside", "Belgian forest", "Dutch canal",
		"Russian steppe", "Ukrainian wheat field", "Alpine mountain", "Mediterranean coast",
		"Sicilian town", "German city ruins", "English Channel", "Atlantic Wall",
		"Ardennes Forest", "Rhineland", "Soviet urban ruins", "Berlin streets",
		
		# Pacific & Asian Theaters
		"Pacific island", "jungle battlefield", "Pacific beach landing", "atoll",
		"tropical forest", "coral reef", "bamboo grove", "rice paddy",
		"Burmese jungle", "Chinese village", "Philippine beach", "volcanic island",
		"Japanese homeland", "Pacific airfield", "jungle airstrip", "coconut plantation",
		
		# Military Settings
		"prisoner of war camp", "concentration camp", "military hospital", "field hospital",
		"military cemetery", "aircraft factory", "tank factory", "shipyard",
		"military depot", "ammunition dump", "fuel depot", "supply dump",
		"military port", "embarkation point", "submarine pen", "naval dry dock",
		
		# Terrain Types
		"desert", "desert oasis", "desert dunes", "rocky desert", "forest",
		"dense forest", "winter forest", "urban area", "bombed city", "city ruins",
		"beach", "landing beach", "rocky beach", "mountain", "mountain pass",
		"field", "farm field", "snow-covered field", "ocean", "open ocean",
		"coastal waters", "river", "river crossing", "flooded river", "bridge",
		
		# Military Infrastructure
		"airfield", "temporary airstrip", "bomber base", "fighter base",
		"naval base", "submarine base", "army barracks", "training camp",
		"military headquarters", "command bunker", "coastal defense",
		"fortified line", "defensive position", "artillery position",
		
		# Military Activities
		"battlefield", "active battlefield", "battlefield aftermath",
		"military parade", "victory parade", "surrender ceremony",
		"military exercise", "amphibious landing exercise", "tank maneuvers",
		"war zone", "civilian evacuation", "occupation zone", "frontline",
		"military checkpoint", "border checkpoint", "military convoy route",
		
		# Home Front
		"war factory", "armaments factory", "aircraft assembly line",
		"vehicle assembly line", "shipbuilding yard", "munitions factory",
		"civilian air raid shelter", "bombed civilian area", "rationing center",
		"recruitment office", "propaganda poster display", "war bonds office",
		"civil defense drill", "air raid aftermath", "victory celebration"
]

era_categories = [
		# Pre-War & Early War
		"pre-World War I era", "World War I era", "interwar period", "early 1930s",
		"Spanish Civil War era", "pre-1939 military", "early World War II",
		"Phoney War period", "Blitzkrieg era", "1939-1940 equipment",
		
		# World War II Specific Periods
		"Battle of Britain era", "North African campaign", "Eastern Front 1941",
		"Pearl Harbor era", "Midway period", "Stalingrad era", "Normandy invasion",
		"Operation Barbarossa", "Battle of the Bulge", "Italian campaign",
		"D-Day preparations", "Market Garden operation", "Fall of Berlin",
		"Island-hopping campaign", "Battle of the Atlantic", "V-E Day era",
		"Pacific War late stage", "atomic bomb era", "Japanese surrender period",
		
		# Post-War Periods
		"immediate post-war", "occupation period", "early Cold War",
		"Korean War era", "1950s military", "Vietnam era", "late Cold War",
		
		# Visual Style Markers
		"1910s style", "1920s style", "1930s style", "1940s style", "1940s military aesthetic",
		"wartime propaganda style", "black and white photography era",
		"wartime color photography", "Technicolor film era", "wartime newsreel style",
		"press photography style", "military documentation style",
		
		# Military Technology Eras
		"early tank warfare", "biplane era", "early radar period", "monoplane transition",
		"early jet aircraft", "V-weapon period", "heavy bomber era", "aircraft carrier warfare",
		"submarine warfare golden age", "amphibious assault development",
		"mechanized warfare", "combined arms doctrine", "early nuclear era",
		
		# National Military Period Styles
		"Wehrmacht prime", "Soviet military buildup", "British Empire forces",
		"American war production peak", "Imperial Japanese forces",
		"Nazi Germany zenith", "Allied powers ascendancy", "Axis powers decline",
		"Red Army resurgence", "American military dominance"
]

activity_categories = [
		# Combat Activities
		"fighting", "tank battle", "infantry assault", "naval engagement",
		"aerial dogfight", "bombing run", "strafing run", "artillery barrage",
		"firing weapon", "machine gun firing", "mortar firing", "shelling",
		"anti-aircraft firing", "sniper activity", "flamethrower attack",
		"bayonet charge", "hand-to-hand combat", "urban combat", "house-to-house fighting",
		
		# Movement & Transportation
		"driving", "convoy movement", "tank column", "troop transport",
		"marching", "infantry advance", "tactical retreat", "military withdrawal",
		"flying", "air patrol", "reconnaissance flight", "bombing mission",
		"parachute drop", "airborne landing", "glider landing", "air resupply",
		"crossing terrain", "river crossing", "beach landing", "amphibious assault",
		"fording stream", "mountain climbing", "moving through jungle",
		"naval convoy", "fleet movement", "submarine patrol", "naval blockade",
		
		# Military Operations
		"digging trenches", "building fortifications", "laying mines", "clearing mines",
		"constructing bridge", "demolishing bridge", "breaching obstacles",
		"setting up artillery", "camouflaging position", "establishing perimeter",
		"setting up command post", "establishing field hospital", "creating airstrip",
		
		# Logistics & Support
		"loading equipment", "unloading equipment", "loading ammunition", "refueling",
		"resupplying troops", "distributing rations", "loading wounded", "evacuating casualties",
		"loading ships", "unloading landing craft", "airdrop receiving", "gathering supplies",
		"towing disabled vehicle", "vehicle recovery", "aircraft maintenance", "tank repair",
		"weapon cleaning", "equipment maintenance", "vehicle maintenance",
		
		# Military Life & Routines
		"training", "infantry drill", "weapons training", "tank crew training", "pilot training",
		"field exercise", "receiving briefing", "map reading", "radio communication",
		"standing guard", "sentry duty", "prisoner handling", "military inspection",
		"cooking field rations", 'military rations', "eating meal", "resting between battles", "writing letters",
		"medical treatment", "field surgery", "distributing supplies", "receiving orders",
		
		# Ceremonial & Administrative
		"military parade", "award ceremony", "flag raising", "surrender ceremony", 
		"prisoner processing", "military funeral", "military wedding", "religious service",
		"officer briefing", "signing documents", "military trial", "propaganda filming",
		"press conference", "VIP visit", "civilian interaction", "occupation duty",
		"war crime investigation", "reconnaissance reporting"
]

candidate_labels = object_categories + scene_categories + era_categories + activity_categories

inputs = processor(images=image, text=candidate_labels, return_tensors="pt", padding="max_length").to("cuda:0")

with torch.no_grad():
	outputs = model(**inputs)

# this is the image-text similarity score
logits_per_image = outputs.logits_per_image

# we can take the softmax to get the label probabilities
probs = logits_per_image.softmax(dim=1)
print(probs.shape)
print(probs)
topk_indices = probs[0].topk(5)
print(topk_indices)
print(topk_indices.indices)
print(topk_indices.values)
print([candidate_labels[i] for i in topk_indices.indices])



# Get probabilities for the first image (assuming batch size is 1)
image_probs = probs[0]

# Sort probabilities in descending order and get the sorted indices
sorted_probs, sorted_indices = torch.sort(image_probs, descending=True)

# Collect the top K unique labels
k = 5 # Number of unique labels to retrieve
unique_top_results = [] # To store (label, probability) tuples
seen_labels = set() # To keep track of labels already added

# Iterate through the sorted results
for i in range(len(sorted_indices)):
		prob = sorted_probs[i].item() # Get probability as a Python number
		idx = sorted_indices[i].item() # Get original index as a Python number
		label = candidate_labels[idx] # Get the label string

		# If the label hasn't been seen yet, add it to our results
		if label not in seen_labels:
				unique_top_results.append((label, prob))
				seen_labels.add(label)

				# Stop once we have collected K unique labels
				if len(unique_top_results) == k:
						break

# Print the top K unique labels and their probabilities
print(f"\nTop {k} unique labels and their probabilities:")
for label, prob in unique_top_results:
		print(f"- {label}: {prob:.4f}")

# You can also print just the list of unique labels if preferred:
print(f"\nTop {k} unique labels:")
print([label for label, prob in unique_top_results])
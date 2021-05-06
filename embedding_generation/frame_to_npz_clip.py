#!/usr/bin/env python
# coding: utf-8


# Import arguments
import argparse
import sys

# Arg parse function.
def parse_args():
	parser = argparse.ArgumentParser(description='')
	parser.add_argument("--index", help="Float Argument).", type=float, default=0)
	parser.add_argument("--gpus", help="GPU's to be used.",
						type=str, default="8")
	parser.add_argument('--set', help='String Argument',
						default = 'training', type=str)
	# parser.add_argument('--set', help='String Argument',
	# 					default = 'validation', type=str)

	
	args = parser.parse_args()
	return args

# Check shell
try:
	shell = get_ipython().__class__.__name__
	sys.argv = sys.argv[0:1]
except:
	pass

# Parse the args.
args = parse_args()
args.index = args.index % 9


# GPU setup
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpus


from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import numpy as np
import torch
import clip
from tqdm.notebook import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
import random
import os
import numpy as np
from PIL import Image
import torchvision   
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sklearn
from torch import nn
from torch.nn.utils.rnn import *
import time
cuda = torch.cuda.is_available()
print("cuda", cuda)
num_workers = 8 if cuda else 0
print(num_workers)
print("Torch version:", torch.__version__)


# # Load CLIP Model


print("Avaliable Models: ", clip.available_models())
model, preprocess = clip.load("RN50", jit=False) # clip.load("ViT-B/32") #

input_resolution = model.input_resolution #.item()
context_length = model.context_length #.item()
vocab_size = model.vocab_size #.item()

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)


# # Selected classes and mapping for kinetics dataset


labels = ['abseiling',
	'acting_in_play',
	'adjusting_glasses',
	'air_drumming',
	'alligator_wrestling',
	'answering_questions',
	'applauding',
	'applying_cream',
	'archaeological_excavation',
	'archery',
	'arguing',
	'arm_wrestling',
	'arranging_flowers',
	'assembling_bicycle',
	'assembling_computer',
	'attending_conference',
	'auctioning',
	'baby_waking_up',
	'backflip_(human)',
	'baking_cookies',
	'balloon_blowing',
	'bandaging',
	'barbequing',
	'bartending',
	'base_jumping',
	'bathing_dog',
	'battle_rope_training',
	'beatboxing',
	'bee_keeping',
	'belly_dancing',
	'bench_pressing',
	'bending_back',
	'bending_metal',
	'biking_through_snow',
	'blasting_sand',
	'blowdrying_hair',
	'blowing_bubble_gum',
	'blowing_glass',
	'blowing_leaves',
	'blowing_nose',
	'blowing_out_candles',
	'bobsledding',
	'bodysurfing',
	'bookbinding',
	'bottling',
	'bouncing_on_bouncy_castle',
	'bouncing_on_trampoline',
	'bowling',
	'braiding_hair',
	'breading_or_breadcrumbing',
	'breakdancing',
	'breaking_boards',
	'breathing_fire',
	'brushing_hair',
	'brushing_teeth',
	'brush_painting',
	'building_cabinet',
	'building_lego',
	'building_sandcastle',
	'building_shed',
	'bulldozing',
	'bull_fighting',
	'bungee_jumping',
	'burping',
	'busking',
	'calculating',
	'calligraphy',
	'canoeing_or_kayaking',
	'capoeira',
	'capsizing',
	'card_stacking',
	'card_throwing',
	'carrying_baby',
	'cartwheeling',
	'carving_ice',
	'carving_pumpkin',
	'casting_fishing_line',
	'catching_fish',
	'catching_or_throwing_baseball',
	'catching_or_throwing_frisbee',
	'catching_or_throwing_softball',
	'celebrating',
	'changing_gear_in_car',
	'changing_oil',
	'changing_wheel',
	'changing_wheel_(not_on_bike)',
	'checking_tires',
	'cheerleading',
	'chewing_gum',
	'chiseling_stone',
	'chiseling_wood',
	'chopping_meat',
	'chopping_vegetables',
	'chopping_wood',
	'clam_digging',
	'clapping',
	'clay_pottery_making',
	'clean_and_jerk',
	'cleaning_floor',
	'cleaning_gutters',
	'cleaning_pool',
	'cleaning_shoes',
	'cleaning_toilet',
	'cleaning_windows',
	'climbing_a_rope',
	'climbing_ladder',
	'climbing_tree',
	'coloring_in',
	'combing_hair',
	'contact_juggling',
	'contorting',
	'cooking_chicken',
	'cooking_egg',
	'cooking_on_campfire',
	'cooking_sausages',
	'cooking_sausages_(not_on_barbeque)',
	'cooking_scallops',
	'cosplaying',
	'counting_money',
	'country_line_dancing',
	'cracking_back',
	'cracking_knuckles',
	'cracking_neck',
	'crawling_baby',
	'crossing_eyes',
	'crossing_river',
	'crying',
	'cumbia',
	'curling_hair',
	'curling_(sport)',
	'cutting_nails',
	'cutting_orange',
	'cutting_pineapple',
	'cutting_watermelon',
	'dancing_ballet',
	'dancing_charleston',
	'dancing_gangnam_style',
	'dancing_macarena',
	'deadlifting',
	'decorating_the_christmas_tree',
	'delivering_mail',
	'digging',
	'dining',
	'directing_traffic',
	'disc_golfing',
	'diving_cliff',
	'docking_boat',
	'dodgeball',
	'doing_aerobics',
	'doing_jigsaw_puzzle',
	'doing_laundry',
	'doing_nails',
	'drawing',
	'dribbling_basketball',
	'drinking',
	'drinking_beer',
	'drinking_shots',
	'driving_car',
	'driving_tractor',
	'drooling',
	'drop_kicking',
	'drumming_fingers',
	'dumpster_diving',
	'dunking_basketball',
	'dyeing_eyebrows',
	'dying_hair',
	'eating_burger',
	'eating_cake',
	'eating_carrots',
	'eating_chips',
	'eating_doughnuts',
	'eating_hotdog',
	'eating_ice_cream',
	'eating_spaghetti',
	'eating_watermelon',
	'egg_hunting',
	'embroidering',
	'exercising_arm',
	'exercising_with_an_exercise_ball',
	'extinguishing_fire',
	'faceplanting',
	'falling_off_bike',
	'falling_off_chair',
	'feeding_birds',
	'feeding_fish',
	'feeding_goats',
	'fencing_(sport)',
	'fidgeting',
	'filling_eyebrows',
	'finger_snapping',
	'fixing_bicycle',
	'fixing_hair',
	'flint_knapping',
	'flipping_pancake',
	'flying_kite',
	'fly_tying',
	'folding_clothes',
	'folding_napkins',
	'folding_paper',
	'front_raises',
	'frying_vegetables',
	'garbage_collecting',
	'gargling',
	'geocaching',
	'getting_a_haircut',
	'getting_a_piercing',
	'getting_a_tattoo',
	'giving_or_receiving_award',
	'gold_panning',
	'golf_chipping',
	'golf_driving',
	'golf_putting',
	'gospel_singing_in_church',
	'grinding_meat',
	'grooming_dog',
	'grooming_horse',
	'gymnastics_tumbling',
	'hammer_throw',
	'hand_washing_clothes',
	'headbanging',
	'headbutting',
	'head_stand',
	'high_jump',
	'high_kick',
	'historical_reenactment',
	'hitting_baseball',
	'hockey_stop',
	'holding_snake',
	'home_roasting_coffee',
	'hopscotch',
	'hoverboarding',
	'huddling',
	'hugging',
	'hugging_baby',
	'hula_hooping',
	'hurdling',
	'hurling_(sport)',
	'ice_climbing',
	'ice_fishing',
	'ice_skating',
	'ice_swimming',
	'inflating_balloons',
	'installing_carpet',
	'ironing',
	'ironing_hair',
	'javelin_throw',
	'jaywalking',
	'jetskiing',
	'jogging',
	'juggling_balls',
	'juggling_fire',
	'juggling_soccer_ball',
	'jumping_bicycle',
	'jumping_into_pool',
	'jumping_jacks',
	'jumpstyle_dancing',
	'karaoke',
	'kicking_field_goal',
	'kicking_soccer_ball',
	'kissing',
	'kitesurfing',
	'knitting',
	'krumping',
	'laughing',
	'lawn_mower_racing',
	'laying_bricks',
	'laying_concrete',
	'laying_stone',
	'laying_tiles',
	'leatherworking',
	'licking',
	'lifting_hat',
	'lighting_fire',
	'lock_picking',
	'longboarding',
	'long_jump',
	'looking_at_phone',
	'luge',
	'lunge',
	'making_a_cake',
	'making_a_sandwich',
	'making_balloon_shapes',
	'making_bed',
	'making_bubbles',
	'making_cheese',
	'making_horseshoes',
	'making_jewelry',
	'making_paper_aeroplanes',
	'making_pizza',
	'making_snowman',
	'making_sushi',
	'making_tea',
	'making_the_bed',
	'marching',
	'marriage_proposal',
	'massaging_back',
	'massaging_feet',
	'massaging_legs',
	'massaging_person\'s_head',
	'milking_cow',
	'moon_walking',
	'mopping_floor',
	'mosh_pit_dancing',
	'motorcycling',
	'mountain_climber_(exercise)',
	'moving_furniture',
	'mowing_lawn',
	'mushroom_foraging',
	'needle_felting',
	'news_anchoring',
	'opening_bottle',
	'opening_door',
	'opening_present',
	'opening_refrigerator',
	'paragliding',
	'parasailing',
	'parkour',
	'passing_American_football_(in_game)',
	'passing_American_football_(not_in_game)',
	'passing_soccer_ball',
	'peeling_apples',
	'peeling_potatoes',
	'person_collecting_garbage',
	'petting_animal_(not_cat)',
	'petting_cat',
	'photobombing',
	'photocopying',
	'picking_fruit',
	'pillow_fight',
	'pinching',
	'pirouetting',
	'planing_wood',
	'planting_trees',
	'plastering',
	'playing_accordion',
	'playing_badminton',
	'playing_bagpipes',
	'playing_basketball',
	'playing_bass_guitar',
	'playing_beer_pong',
	'playing_blackjack',
	'playing_cards',
	'playing_cello',
	'playing_chess',
	'playing_clarinet',
	'playing_controller',
	'playing_cricket',
	'playing_cymbals',
	'playing_darts',
	'playing_didgeridoo',
	'playing_dominoes',
	'playing_drums',
	'playing_field_hockey',
	'playing_flute',
	'playing_gong',
	'playing_guitar',
	'playing_hand_clapping_games',
	'playing_harmonica',
	'playing_harp',
	'playing_ice_hockey',
	'playing_keyboard',
	'playing_kickball',
	'playing_laser_tag',
	'playing_lute',
	'playing_maracas',
	'playing_marbles',
	'playing_monopoly',
	'playing_netball',
	'playing_ocarina',
	'playing_organ',
	'playing_paintball',
	'playing_pan_pipes',
	'playing_piano',
	'playing_pinball',
	'playing_ping_pong',
	'playing_poker',
	'playing_polo',
	'playing_recorder',
	'playing_rubiks_cube',
	'playing_saxophone',
	'playing_scrabble',
	'playing_squash_or_racquetball',
	'playing_tennis',
	'playing_trombone',
	'playing_trumpet',
	'playing_ukulele',
	'playing_violin',
	'playing_volleyball',
	'playing_with_trains',
	'playing_xylophone',
	'poking_bellybutton',
	'pole_vault',
	'polishing_metal',
	'popping_balloons',
	'pouring_beer',
	'preparing_salad',
	'presenting_weather_forecast',
	'pull_ups',
	'pumping_fist',
	'pumping_gas',
	'punching_bag',
	'punching_person_(boxing)',
	'pushing_car',
	'pushing_cart',
	'pushing_wheelbarrow',
	'pushing_wheelchair',
	'push_up',
	'putting_in_contact_lenses',
	'putting_on_eyeliner',
	'putting_on_foundation',
	'putting_on_lipstick',
	'putting_on_mascara',
	'putting_on_sari',
	'putting_on_shoes',
	'raising_eyebrows',
	'reading_book',
	'reading_newspaper',
	'recording_music',
	'repairing_puncture',
	'riding_a_bike',
	'riding_camel',
	'riding_elephant',
	'riding_mechanical_bull',
	'riding_mountain_bike',
	'riding_mule',
	'riding_or_walking_with_horse',
	'riding_scooter',
	'riding_snow_blower',
	'riding_unicycle',
	'ripping_paper',
	'roasting_marshmallows',
	'roasting_pig',
	'robot_dancing',
	'rock_climbing',
	'rock_scissors_paper',
	'roller_skating',
	'rolling_pastry',
	'rope_pushdown',
	'running_on_treadmill',
	'sailing',
	'salsa_dancing',
	'sanding_floor',
	'sausage_making',
	'sawing_wood',
	'scrambling_eggs',
	'scrapbooking',
	'scrubbing_face',
	'scuba_diving',
	'separating_eggs',
	'setting_table',
	'sewing',
	'shaking_hands',
	'shaking_head',
	'shaping_bread_dough',
	'sharpening_knives',
	'sharpening_pencil',
	'shaving_head',
	'shaving_legs',
	'shearing_sheep',
	'shining_flashlight',
	'shining_shoes',
	'shooting_basketball',
	'shooting_goal_(soccer)',
	'shopping',
	'shot_put',
	'shoveling_snow',
	'shredding_paper',
	'shucking_oysters',
	'shuffling_cards',
	'shuffling_feet',
	'side_kick',
	'sign_language_interpreting',
	'singing',
	'sipping_cup',
	'situp',
	'skateboarding',
	'skiing_crosscountry',
	'skiing_mono',
	'skiing_(not_slalom_or_crosscountry)',
	'skiing_slalom',
	'ski_jumping',
	'skipping_rope',
	'skipping_stone',
	'skydiving',
	'slacklining',
	'slapping',
	'sled_dog_racing',
	'sleeping',
	'smashing',
	'smoking',
	'smoking_hookah',
	'smoking_pipe',
	'snatch_weight_lifting',
	'sneezing',
	'sniffing',
	'snorkeling',
	'snowboarding',
	'snowkiting',
	'snowmobiling',
	'somersaulting',
	'spelunking',
	'spinning_poi',
	'spraying',
	'spray_painting',
	'springboard_diving',
	'square_dancing',
	'squat',
	'standing_on_hands',
	'staring',
	'steer_roping',
	'sticking_tongue_out',
	'stomping_grapes',
	'stretching_arm',
	'stretching_leg',
	'strumming_guitar',
	'sucking_lolly',
	'surfing_crowd',
	'surfing_water',
	'sweeping_floor',
	'swimming_backstroke',
	'swimming_breast_stroke',
	'swimming_butterfly_stroke',
	'swimming_front_crawl',
	'swing_dancing',
	'swinging_baseball_bat',
	'swinging_legs',
	'swinging_on_something',
	'sword_fighting',
	'sword_swallowing',
	'tackling',
	'tagging_graffiti',
	'tai_chi',
	'taking_a_shower',
	'talking_on_cell_phone',
	'tango_dancing',
	'tap_dancing',
	'tapping_guitar',
	'tapping_pen',
	'tasting_beer',
	'tasting_food',
	'tasting_wine',
	'testifying',
	'texting',
	'threading_needle',
	'throwing_axe',
	'throwing_ball',
	'throwing_discus',
	'throwing_knife',
	'throwing_snowballs',
	'throwing_tantrum',
	'throwing_water_balloon',
	'tickling',
	'tie_dying',
	'tightrope_walking',
	'tiptoeing',
	'tobogganing',
	'tossing_coin',
	'tossing_salad',
	'training_dog',
	'trapezing',
	'trimming_or_shaving_beard',
	'trimming_shrubs',
	'trimming_trees',
	'triple_jump',
	'twiddling_fingers',
	'tying_bow_tie',
	'tying_knot_(not_on_a_tie)',
	'tying_shoe_laces',
	'tying_tie',
	'unboxing',
	'unloading_truck',
	'using_a_microscope',
	'using_a_paint_roller',
	'using_a_power_drill',
	'using_a_sledge_hammer',
	'using_atm',
	'using_a_wrench',
	'using_bagging_machine',
	'using_circular_saw',
	'using_computer',
	'using_inhaler',
	'using_puppets',
	'using_remote_controller_(not_gaming)',
	'using_segway',
	'vacuuming_floor',
	'vault',
	'visiting_the_zoo',
	'wading_through_mud',
	'wading_through_water',
	'waiting_in_line',
	'waking_up',
	'walking_the_dog',
	'walking_through_snow',
	'washing_dishes',
	'washing_feet',
	'washing_hair',
	'washing_hands',
	'watching_tv',
	'watering_plants',
	'water_skiing',
	'water_sliding',
	'waving_hand',
	'waxing_back',
	'waxing_chest',
	'waxing_eyebrows',
	'waxing_legs',
	'weaving_basket',
	'weaving_fabric',
	'welding',
	'whistling',
	'windsurfing',
	'winking',
	'wood_burning_(art)',
	'wrapping_present',
	'wrestling',
	'writing',
	'yawning',
	'yoga',
	'zumba',
]

sub_labels = ['making_tea',
		'shaking_head',
		'skiing_slalom',
		'bobsledding',
		'high_kick',
		'scrambling_eggs',
		'bee_keeping',
		'swinging_on_something',
		'washing_hands',
		'laying_bricks',
		'push_up',
		'doing_nails',
		'massaging_legs',
		'using_computer',
		'clapping',
		'drinking_beer',
		'eating_chips',
		'riding_mule',
		'petting_animal_(not_cat)',
		'frying_vegetables',
		'skiing_(not_slalom_or_crosscountry)',
		'snowkiting',
		'massaging_person\'s_head',
		'cutting_nails',
		'picking_fruit']
map_id = {}
i=0
for label in labels:
	map_id[label]=i
	i+=1


import os
import pdb
ROOT = "/data3/puppala/data/kinetics_jpg/{}".format(args.set)
DEST = "/data3/puppala/data/kinetics_embeddings/{}".format(args.set)
print(ROOT)
print(DEST)
cnt = 1
for filename in labels:
	if filename not in os.listdir(ROOT):
		print(filename," - Missing")
		
labels = os.listdir(ROOT)


class_names = os.listdir(ROOT)
len_classes = len(labels)
split_var = np.ceil(len_classes/8)
num_samples_per_class = 100
print("Creating embeddings from {} - {}".format(args.index * split_var, args.index*split_var + split_var))
for cls_idx, cls_name in enumerate(labels):
	if cls_idx < args.index*split_var or cls_idx >= (args.index + 1)*split_var:
		continue
	class_file = os.path.join(ROOT,cls_name)
	random_tags = os.listdir(class_file)
	save_dir = os.path.join(DEST, cls_name)
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	len_random_tags = len(random_tags)
	avg_time = []
	for rand_id, random_tag in enumerate(random_tags):
		if rand_id >= num_samples_per_class:
			continue
		try:
			start_time = time.time()
			video_name = os.path.join(class_file,random_tag)
			save_dir = os.path.join(DEST, cls_name)
			file_save = os.path.join(save_dir, random_tag + '.npz')
			if os.path.exists(file_save):
				continue
			count = 0
			frames = sorted([ x for x in os.listdir(video_name) if x.endswith('.jpg')])
			N = len(frames)
			if N >=100:
				n = N//100
			else:
				n = 1
			selected_frames = np.arange(0,N,n).tolist()[0:100]
			images = []
			for frame in frames:
				if count in selected_frames:
					tmp = str(count)
					image = Image.open(os.path.join(video_name,frame))
					image = preprocess(image)
# 					image = torch.unsqueeze(image, 0)
					image = image.cuda()
					images.append(image)
				count+=1

			images_batch = torch.stack(images, dim=0)
			image_features, preattention_features = model.encode_image(images_batch , feat=True)
# 			print(image_features.shape, preattention_features.shape)
			image_features /= (image_features.norm(dim=-1, keepdim=True)  + 10e-8)
			image_features = image_features.detach().cpu().numpy()

			label = map_id[cls_name]*np.ones(image_features.shape[0])
			save_dir = os.path.join(DEST, cls_name)
			file_save = os.path.join(save_dir, random_tag)
			np.savez(file_save, data=image_features,label=label)
			# np.savez(file_save + '_preattention_features', data=preattention_features.detach().cpu().numpy(), label=label)
			end_time = time.time()
			avg_time.append(end_time - start_time)
			print(cls_idx - args.index*split_var, '/', np.ceil(len_classes/8), '-', rand_id, '/', 
				  len_random_tags, '\t time taken - ', np.round(np.mean(avg_time), 3), end='\r')
		except Exception as e:
			print(e)



def get_experiments_list(dataset):
	if dataset == 'inbreast':
		experiments = {
			'Default': 'Default'
		}
	elif dataset == 'bcdr':
		experiments = {
			'Plain': 'BCDR_segmentation_test',
			'Plain_full': 'BCDR_seg_full_set',
			'Plain_half': 'BCDR_seg_half_set',
			'LE': 'BCDR_LE_segmentation_test',
		}
	elif dataset == 'mnm':
		experiments = {
			'Defualt': 'Defualt',
		}
	else:
		raise ValueError(f"{dataset} not implemented")
	return experiments